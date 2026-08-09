from dataclasses import dataclass


@dataclass
class SourceReference:
    source: str
    page: int | str
    score: float


@dataclass
class RAGResponse:
    answer: str
    sources: list[SourceReference]