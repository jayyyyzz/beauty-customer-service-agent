import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


_configured_model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip()
if _configured_model in {"", "deepseek-chat"}:
    _configured_model = "deepseek-v4-flash"


LLM_deepseek_config = {
    "api_key": os.getenv("DEEPSEEK_API_KEY", "").strip(),
    "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    "model": _configured_model,
    "thinking": os.getenv("DEEPSEEK_THINKING", "disabled"),
    "intent_temperature": float(os.getenv("DEEPSEEK_INTENT_TEMPERATURE", "0")),
    "answer_temperature": float(os.getenv("DEEPSEEK_ANSWER_TEMPERATURE", "0")),
}

ES_search_config = {
    "url": os.getenv("ES_URL", "http://127.0.0.1:9200"),
    "api_key": os.getenv("ES_API_KEY", "").strip(),
    "user": os.getenv("ES_USER", ""),
    "password": os.getenv("ES_PASSWORD", ""),
    "index": os.getenv("ES_INDEX", "customer_service_knowledge_v1"),
    "mode": os.getenv("ES_SEARCH_MODE", "hybrid"),
    "insecure": os.getenv("ES_INSECURE", "false").strip().lower() in {"1", "true", "yes"},
    "num_candidates": int(os.getenv("ES_NUM_CANDIDATES", "120")),
    "text_boost": float(os.getenv("ES_TEXT_BOOST", "0.2")),
    "vector_boost": float(os.getenv("ES_VECTOR_BOOST", "1.0")),
    "vector_field": os.getenv("ES_VECTOR_FIELD", "content_vector"),
    "candidate_k": int(os.getenv("ES_CANDIDATE_K", "12")),
    "rrf_k": int(os.getenv("ES_RRF_K", "60")),
    "mmr_lambda": float(os.getenv("ES_MMR_LAMBDA", "0.70")),
    "dedup_threshold": float(os.getenv("ES_DEDUP_THRESHOLD", "0.82")),
    "bm25_fast_path": os.getenv("ES_BM25_FAST_PATH", "false").strip().lower() in {"1", "true", "yes"},
}

BUSINESS_TOOL_config = {
    "db_path": os.getenv("BUSINESS_TOOL_DB", str(ROOT / "runtime" / "business_tools.db")),
    "confirmation_ttl_seconds": int(os.getenv("BUSINESS_CONFIRM_TTL_SECONDS", "600")),
    "max_retries": int(os.getenv("BUSINESS_TOOL_MAX_RETRIES", "3")),
}

AGENT_RUNTIME_config = {
    "max_concurrent_requests": int(os.getenv("AGENT_MAX_CONCURRENCY", "4")),
    "request_timeout_seconds": float(os.getenv("AGENT_REQUEST_TIMEOUT_SECONDS", "90")),
    "rate_limit_requests": int(os.getenv("AGENT_RATE_LIMIT_REQUESTS", "20")),
    "rate_limit_window_seconds": int(os.getenv("AGENT_RATE_LIMIT_WINDOW_SECONDS", "60")),
    "clarify_confidence_threshold": float(
        os.getenv("AGENT_CLARIFY_CONFIDENCE_THRESHOLD", "0.50")
    ),
    "handoff_confidence_threshold": float(
        os.getenv("AGENT_HANDOFF_CONFIDENCE_THRESHOLD", "0.45")
    ),
    "handoff_db_path": os.getenv(
        "AGENT_HANDOFF_DB", str(ROOT / "runtime" / "handoffs.db")
    ),
    "log_path": os.getenv("AGENT_LOG_PATH", str(ROOT / "runtime" / "agent.log")),
}

RERANK_config = {
    "enabled": os.getenv("RERANK_ENABLED", "false").strip().lower() in {"1", "true", "yes"},
    "conditional": os.getenv("RERANK_CONDITIONAL", "true").strip().lower() in {"1", "true", "yes"},
    "candidate_k": int(os.getenv("RERANK_CANDIDATE_K", "5")),
    "max_doc_chars": int(os.getenv("RERANK_MAX_DOC_CHARS", "700")),
}
