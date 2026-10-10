"""API contract (schemas). Giữ ổn định: AI pipeline thật phải trả đúng các trường này."""
from typing import List
from pydantic import BaseModel, Field


class Filters(BaseModel):
    occasion: List[str] = Field(default_factory=list)
    color: List[str] = Field(default_factory=list)
    style: List[str] = Field(default_factory=list)
    pattern: List[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str = ""
    filters: Filters = Field(default_factory=Filters)
    top_k: int = Field(5, ge=1, le=50)


class SearchResult(BaseModel):
    id: str
    name: str                      # trường bổ sung so với mẫu ban đầu, để UI hiện tiêu đề
    image: str
    description: str
    score: float                   # 0..1, giảm dần trong danh sách
    occasion: List[str]
    color: List[str]
    style: List[str]
    pattern: List[str]
    reason: str
    cultural_context: str
    warnings: List[str]
    visual: dict = Field(default_factory=dict)
    # Trường mở rộng (tùy chọn, frontend cũ bỏ qua được)
    warning_details: List[dict] = Field(default_factory=list)   # [{warning, severity, rule_id}]
    cultural_sources: List[str] = Field(default_factory=list)    # id mục KB đã dùng


class SearchResponse(BaseModel):
    results: List[SearchResult]
