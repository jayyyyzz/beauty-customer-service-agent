# -*- coding: utf-8 -*-
"""
纯 numpy 实现的 bge-small-zh-v1.5 编码器(无需 torch/transformers)
= safetensors 权重解析 + WordPiece 分词 + BERT 前向传播 + CLS pooling + L2 归一化
教学注释版:每一步都标注了它对应 transformers 库里的哪个环节。
"""
import json, struct, unicodedata
import numpy as np

# ---------------------------------------------------------------
# 1. 权重加载:safetensors 就是 [8字节头长度][json头][原始张量字节] 的平铺文件
# ---------------------------------------------------------------
def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        data = f.read()
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        s, e = info["data_offsets"]
        dt = {"F32": np.float32, "I64": np.int64}[info["dtype"]]
        out[name] = np.frombuffer(data[s:e], dtype=dt).reshape(info["shape"]).copy()
    return out

# ---------------------------------------------------------------
# 2. WordPiece 分词器(等价 BertTokenizer, do_lower_case=True)
# ---------------------------------------------------------------
class WordPieceTokenizer:
    def __init__(self, vocab_path):
        self.vocab = {}
        with open(vocab_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                self.vocab[line.rstrip("\n")] = i
        self.unk, self.cls, self.sep, self.pad = (
            self.vocab["[UNK]"], self.vocab["[CLS]"],
            self.vocab["[SEP]"], self.vocab["[PAD]"])

    @staticmethod
    def _is_cjk(ch):
        cp = ord(ch)
        return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
                0xF900 <= cp <= 0xFAFF or 0x20000 <= cp <= 0x2A6DF)

    @staticmethod
    def _is_punct(ch):
        cp = ord(ch)
        if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
            return True
        return unicodedata.category(ch).startswith("P")

    def _basic_tokenize(self, text):
        text = text.lower()
        # 去重音(BERT strip_accents 随 lowercase 默认开启)
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        out, buf = [], []
        for ch in text:
            if unicodedata.category(ch).startswith("C") and ch not in "\t\n\r":
                continue  # 控制字符
            if ch.isspace():
                if buf: out.append("".join(buf)); buf = []
            elif self._is_cjk(ch) or self._is_punct(ch):
                if buf: out.append("".join(buf)); buf = []
                out.append(ch)  # 中文逐字、标点单独成词
            else:
                buf.append(ch)
        if buf: out.append("".join(buf))
        return out

    def _wordpiece(self, word):
        if len(word) > 100:
            return [self.unk]
        ids, start = [], 0
        while start < len(word):
            end, cur = len(word), None
            while start < end:
                sub = word[start:end]
                if start > 0: sub = "##" + sub
                if sub in self.vocab: cur = self.vocab[sub]; break
                end -= 1
            if cur is None:
                return [self.unk]
            ids.append(cur); start = end
        return ids

    def encode(self, text, max_len=512):
        ids = [self.cls]
        for w in self._basic_tokenize(text):
            ids.extend(self._wordpiece(w))
            if len(ids) >= max_len - 1:
                ids = ids[:max_len - 1]; break
        ids.append(self.sep)
        return ids

# ---------------------------------------------------------------
# 3. BERT 前向传播(与 BertModel 逐层等价)
# ---------------------------------------------------------------
def layer_norm(x, w, b, eps=1e-12):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b

def erf(x):  # Abramowitz-Stegun 7.1.26 近似, 精度 ~1e-7, 免 scipy 依赖
    s = np.sign(x); x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y

def gelu(x):  # transformers 的 gelu (erf 版)
    return x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)

class BgeEncoder:
    def __init__(self, model_dir):
        self.w = load_safetensors(f"{model_dir}/model.safetensors")
        self.tok = WordPieceTokenizer(f"{model_dir}/vocab.txt")
        cfg = json.load(open(f"{model_dir}/config.json"))
        self.n_layers = cfg["num_hidden_layers"]   # 4
        self.n_heads  = cfg["num_attention_heads"] # 8
        self.dim      = cfg["hidden_size"]         # 512

    def _embed(self, ids_batch, mask):
        w = self.w
        seq = ids_batch.shape[1]
        x = (w["embeddings.word_embeddings.weight"][ids_batch]
             + w["embeddings.position_embeddings.weight"][:seq][None]
             + w["embeddings.token_type_embeddings.weight"][0][None, None])
        return layer_norm(x, w["embeddings.LayerNorm.weight"], w["embeddings.LayerNorm.bias"])

    def _layer(self, x, mask, i):
        w, H = self.w, self.n_heads
        B, S, D = x.shape
        hd = D // H
        p = f"encoder.layer.{i}."
        def lin(name, v):
            return v @ w[p + name + ".weight"].T + w[p + name + ".bias"]
        # --- 多头自注意力 ---
        q = lin("attention.self.query", x).reshape(B, S, H, hd).transpose(0, 2, 1, 3)
        k = lin("attention.self.key",   x).reshape(B, S, H, hd).transpose(0, 2, 1, 3)
        v = lin("attention.self.value", x).reshape(B, S, H, hd).transpose(0, 2, 1, 3)
        att = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd)
        att += (1.0 - mask[:, None, None, :]) * -10000.0  # padding 位打负无穷
        ctx = (softmax(att) @ v).transpose(0, 2, 1, 3).reshape(B, S, D)
        x = layer_norm(lin("attention.output.dense", ctx) + x,
                       w[p + "attention.output.LayerNorm.weight"],
                       w[p + "attention.output.LayerNorm.bias"])
        # --- FFN ---
        h = gelu(lin("intermediate.dense", x))
        x = layer_norm(lin("output.dense", h) + x,
                       w[p + "output.LayerNorm.weight"],
                       w[p + "output.LayerNorm.bias"])
        return x

    def encode(self, texts, batch_size=16, max_len=512, verbose=False):
        """返回 L2 归一化后的 float32 向量 (N, 512)。bge 用 CLS 位做句向量。"""
        all_ids = [self.tok.encode(t, max_len) for t in texts]
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for s in range(0, len(texts), batch_size):
            batch = all_ids[s:s + batch_size]
            L = max(len(i) for i in batch)
            ids = np.full((len(batch), L), self.tok.pad, dtype=np.int64)
            mask = np.zeros((len(batch), L), dtype=np.float32)
            for j, seq in enumerate(batch):
                ids[j, :len(seq)] = seq
                mask[j, :len(seq)] = 1.0
            x = self._embed(ids, mask)
            for i in range(self.n_layers):
                x = self._layer(x, mask, i)
            cls = x[:, 0, :]                       # CLS pooling
            cls /= np.linalg.norm(cls, axis=1, keepdims=True)  # 归一化 -> 点积=余弦
            out[s:s + len(batch)] = cls
            if verbose:
                print(f"  {min(s + batch_size, len(texts))}/{len(texts)}", flush=True)
        return out
