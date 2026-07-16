# 多源知识接入 RAG

当前统一知识库包含五类数据：

| `document_type` | 来源 | 数量（当前数据） |
|---|---|---:|
| `conversation` | 清洗后的主题聚合customer_dialogues Chunk | 1,904 |
| `product` | 商品知识 | 1,034 |
| `faq` | FAQ 标准问答（含补充规范问答） | 22,576 |
| `policy` | 品牌售后政策 | 3 |
| `shipping` | 品牌物流规则 | 3 |

订单数据不进入 RAG，仍由业务 API 按订单号查询，避免向量检索返回过期或错误的订单状态。

## 1. 配置

在项目根目录的 `.env` 中确认：

```dotenv
ES_URL=http://127.0.0.1:9200
ES_INDEX=customer_service_knowledge_v1
ES_SEARCH_MODE=hybrid
ES_VECTOR_FIELD=content_vector
```

如使用账号密码或 API Key，再配置 `ES_USER`、`ES_PASSWORD` 或 `ES_API_KEY`。只有本地自签名 HTTPS 才设置 `ES_INSECURE=true`。

## 2. 入库前校验

```powershell
.venv\Scripts\python.exe es_store\es_ingest.py --dry-run --sources all
```

该命令只校验字段转换、文档 ID 和数据源，不连接 Elasticsearch，也不会生成向量。

## 3. 首次全量入库

```powershell
.venv\Scripts\python.exe es_store\es_ingest.py --sources all --recreate
```

当前共 25,520 条文档。FAQ 与对话采用“标题 + 问题”生成检索向量，答案保留为召回内容；商品、政策和物流规则使用完整知识文本生成向量。全量首次向量化耗时取决于本机 CPU，可以先用小范围数据打通链路：

```powershell
.venv\Scripts\python.exe es_store\es_ingest.py --sources conversation,product,policy,shipping --recreate
```

确认检索正常后，再追加 FAQ：

```powershell
.venv\Scripts\python.exe es_store\es_ingest.py --sources faq
```

当前在线索引已用 `policy,shipping` 的 6 条数据完成建库、增量比较和来源检索验证；继续执行 `--sources all` 即可补齐其余数据，不需要再次删除索引。

## 4. 增量更新

源 CSV 或 Chunk 更新后，直接再次运行，不加 `--recreate`：

```powershell
.venv\Scripts\python.exe es_store\es_ingest.py --sources all
```

脚本会读取 ES 中现有 `content_hash`，只为新增或内容发生变化的文档重新生成向量。若源文件删除了记录，请使用 `--recreate` 做一次全量同步，清除 ES 中残留的旧文档。

## 5. 带元数据过滤的检索

```powershell
# 只检索售后政策和 FAQ
.venv\Scripts\python.exe es_store\es_search.py 退货政策 --document-type policy,faq

# 只检索 ColourPop 的政策
.venv\Scripts\python.exe es_store\es_search.py 退货政策 --document-type policy --brand ColourPop

# 只检索商品知识
.venv\Scripts\python.exe es_store\es_search.py 油皮适合什么产品 --document-type product
```

Agent 会依据一级意图自动选择知识类型。例如商品咨询优先检索 `product/faq/conversation`，售后问题优先检索 `policy/faq/conversation`，物流问题优先检索 `shipping/faq/conversation`。

## 6. 来源引用

召回结果包含 `source_name`、`source_url`、`document_type` 和 `title`。生成答案时使用 `[S1]`、`[S2]` 标记知识来源，`handle_user_question()` 的返回值同时包含结构化 `citations` 数组，便于前端展示可点击链接。

## 7. 自动化验证

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖多源转换、稳定 ID、元数据过滤、检索去重和引用透传，不需要调用 DeepSeek 或 Elasticsearch。
