const state = {
  conversationId: `demo-${crypto.randomUUID().slice(0, 8)}`,
  messages: [],
  pendingToolContext: null,
  loading: false,
};

const $ = (selector) => document.querySelector(selector);
const messagesEl = $("#messages");
const form = $("#chatForm");
const input = $("#messageInput");
const sendButton = $("#sendButton");
const confirmButton = $("#confirmButton");
const toast = $("#toast");

function setConversationLabel() {
  $("#conversationId").textContent = state.conversationId;
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  window.setTimeout(() => toast.classList.remove("show"), 2800);
}

function addMessage(role, content, meta = "") {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  const text = document.createElement("div");
  text.textContent = content;
  bubble.appendChild(text);
  if (meta) {
    const metadata = document.createElement("div");
    metadata.className = "message-meta";
    metadata.textContent = meta;
    bubble.appendChild(metadata);
  }
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return row;
}

function addTyping() {
  const row = addMessage("agent", "");
  row.dataset.typing = "true";
  row.querySelector(".message-bubble > div").innerHTML = '<span class="typing" aria-label="Agent 正在处理"><i></i><i></i><i></i></span>';
  return row;
}

function setLoading(loading) {
  state.loading = loading;
  sendButton.disabled = loading;
  input.disabled = loading;
  sendButton.querySelector("span:first-child").textContent = loading ? "处理中" : "发送";
}

function toolContextFromResponse(data) {
  if (data?.api_data?.next_tool_context) return data.api_data.next_tool_context;
  const items = data?.api_data?.task_results || [];
  for (const item of items) {
    if (item?.data?.next_tool_context) return item.data.next_tool_context;
  }
  return null;
}

function updateTrace(data) {
  const intent = data.intent || data.intents?.[0] || {};
  const confidence = Number(intent.intent_confidence || 0);
  $("#routeBadge").textContent = data.route || "unknown";
  $("#intentName").textContent = intent.intent_level1 || "未识别";
  $("#confidence").textContent = intent.intent_confidence == null ? "—" : `${Math.round(confidence * 100)}%`;
  $("#intentLogic").textContent = intent.intent_logic || "本轮未调用意图模型。";
  $("#latency").textContent = data.trace?.latency_ms ? `${data.trace.latency_ms} ms` : "— ms";

  const sources = data.citations || [];
  $("#sourceCount").textContent = String(sources.length);
  const sourceList = $("#sourceList");
  sourceList.replaceChildren();
  if (!sources.length) {
    const empty = document.createElement("li");
    empty.className = "empty-row";
    empty.textContent = "本轮未使用知识来源";
    sourceList.appendChild(empty);
  } else {
    sources.forEach((source) => {
      const item = document.createElement("li");
      const title = document.createElement(source.source_url ? "a" : "strong");
      title.textContent = `${source.citation_id || "S?"} · ${source.title || source.source_name || "知识片段"}`;
      if (source.source_url) {
        title.href = source.source_url;
        title.target = "_blank";
        title.rel = "noreferrer";
      }
      const meta = document.createElement("p");
      meta.textContent = `${source.document_type || "knowledge"} · ${source.source_name || "内部知识库"}`;
      item.append(title, meta);
      sourceList.appendChild(item);
    });
  }

  const safety = data.safety || {};
  $("#safetyStatus").textContent = safety.level && safety.level !== "normal" ? safety.level : "正常";
  const toolAction = data.api_data?.action || data.api_data?.task_results?.map((item) => item.data?.action).filter(Boolean).join("、") || "未触发";
  const pii = data.pii_redacted?.length ? data.pii_redacted.join("、") : "未检测";
  const handoff = data.handoff?.ticket_id || "未创建";
  $("#detailList").innerHTML = `
    <div><dt>工具调用</dt><dd>${escapeHtml(toolAction)}</dd></div>
    <div><dt>敏感信息</dt><dd>${escapeHtml(pii)}</dd></div>
    <div><dt>人工工单</dt><dd>${escapeHtml(handoff)}</dd></div>
    <div><dt>请求追踪</dt><dd>${escapeHtml(data.trace?.request_id || "—")}</dd></div>`;

  state.pendingToolContext = toolContextFromResponse(data);
  confirmButton.classList.toggle("hidden", !state.pendingToolContext);
  $("#rawResponse").textContent = JSON.stringify(data, null, 2);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function sendMessage(question, toolContext = null) {
  if (!question.trim() || state.loading) return;
  if (!toolContext) {
    addMessage("user", question.trim(), "客户");
  } else {
    addMessage("user", "确认执行上述操作", "客户确认");
  }
  const history = state.messages.slice(-20);
  state.messages.push({ role: "buyer", content: toolContext ? "确认执行上述操作" : question.trim() });
  const typing = addTyping();
  setLoading(true);
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversation_id: state.conversationId,
        question: toolContext ? "确认执行上述操作" : question.trim(),
        messages: history,
        knowledge_top_k: 3,
        tool_context: toolContext,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "请求失败");
    typing.remove();
    addMessage("agent", data.answer || "没有生成回复", `${data.route || "unknown"} · Agent`);
    state.messages.push({ role: "seller", content: data.answer || "" });
    updateTrace(data);
  } catch (error) {
    typing.remove();
    addMessage("agent", `暂时无法完成请求：${error.message}`, "系统错误");
    showToast(error.message);
  } finally {
    setLoading(false);
    input.focus();
  }
}

async function requestHandoff() {
  const summary = state.messages.at(-1)?.content || input.value.trim() || "用户主动请求人工客服";
  try {
    const response = await fetch("/api/handoff", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversation_id: state.conversationId,
        reason: "user_requested",
        summary,
        priority: "normal",
      }),
    });
    const ticket = await response.json();
    if (!response.ok) throw new Error(ticket.detail || "转接失败");
    addMessage("agent", `已创建人工客服工单 ${ticket.ticket_id}，客服会继续处理。`, "人工转接");
    showToast(`已转人工：${ticket.ticket_id}`);
  } catch (error) {
    showToast(error.message);
  }
}

function resetConversation() {
  state.conversationId = `demo-${crypto.randomUUID().slice(0, 8)}`;
  state.messages = [];
  state.pendingToolContext = null;
  messagesEl.innerHTML = '<div class="welcome-note"><p>新会话已创建。</p><small>可以从上方示例问题开始，也可以直接输入客户诉求。</small></div>';
  confirmButton.classList.add("hidden");
  setConversationLabel();
  input.focus();
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    const online = response.ok && data.elasticsearch?.available && data.deepseek_configured;
    $("#healthDot").className = `status-dot ${online ? "online" : "offline"}`;
    $("#healthText").textContent = online ? "服务就绪" : "部分依赖未就绪";
  } catch {
    $("#healthDot").className = "status-dot offline";
    $("#healthText").textContent = "服务不可用";
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value;
  input.value = "";
  sendMessage(question);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.prompt;
    input.focus();
  });
});

confirmButton.addEventListener("click", () => {
  if (state.pendingToolContext) sendMessage("确认执行上述操作", state.pendingToolContext);
});
$("#handoffButton").addEventListener("click", requestHandoff);
$("#newChatButton").addEventListener("click", resetConversation);

setConversationLabel();
checkHealth();
input.focus();
