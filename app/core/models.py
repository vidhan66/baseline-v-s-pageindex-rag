from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, prompt: int = 0, completion: int = 0, total: int = 0) -> None:
        self.prompt_tokens += int(prompt or 0)
        self.completion_tokens += int(completion or 0)
        self.total_tokens += int(total or 0)


@dataclass
class RagResult:
    ok: bool
    answer: str
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    elapsed_ms: int = 0
    error: Optional[str] = None
    meta: Dict[str, str] = field(default_factory=dict)


@dataclass
class MemoryTurn:
    question: str
    answer: str


def memory_to_text(turns: List[MemoryTurn]) -> str:
    if not turns:
        return "No previous conversation."
    blocks: List[str] = []
    for idx, t in enumerate(turns, start=1):
        blocks.append(f"Turn {idx} - User: {t.question}\nTurn {idx} - Assistant: {t.answer}")
    return "\n\n".join(blocks)

