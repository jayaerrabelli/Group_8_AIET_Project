from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Context:
    text: str
    source: str = ""
    score: float | None = None


@dataclass
class RagResult:
    answer: str
    contexts: list[Context] = field(default_factory=list)
    domain: str = ""
