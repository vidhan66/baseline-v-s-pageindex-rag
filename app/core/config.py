import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    pageindex_api_key: str = field(default_factory=lambda: os.getenv("PAGEINDEX_API_KEY", ""))

    chat_model: str = field(default_factory=lambda: os.getenv("CHAT_MODEL", "gpt-4o-mini"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))

    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200")))
    retrieval_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_K", "6")))
    retrieval_fetch_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_FETCH_K", "25")))

    pageindex_ready_timeout_sec: int = field(default_factory=lambda: int(os.getenv("PAGEINDEX_READY_TIMEOUT_SEC", "300")))
    pageindex_poll_sec: int = field(default_factory=lambda: int(os.getenv("PAGEINDEX_READY_POLL_SEC", "5")))

    tree_batch_size: int = field(default_factory=lambda: int(os.getenv("TREE_BATCH_SIZE", "80")))
    max_summary_chars: int = field(default_factory=lambda: int(os.getenv("MAX_SUMMARY_CHARS", "280")))
    max_selected_nodes: int = field(default_factory=lambda: int(os.getenv("MAX_SELECTED_NODES", "12")))
    max_context_chars: int = field(default_factory=lambda: int(os.getenv("MAX_CONTEXT_CHARS", "12000")))

    # Simple per-strategy conversation memory (turns)
    memory_turns: int = field(default_factory=lambda: int(os.getenv("MEMORY_TURNS", "4")))

    # Timeout per strategy answer call
    answer_timeout_sec: int = field(default_factory=lambda: int(os.getenv("ANSWER_TIMEOUT_SEC", "45")))

