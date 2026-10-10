"""API layer: chỉ nhận request, gọi service, trả response."""

import json
import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .schemas import SearchRequest, SearchResponse
from .search_service import BaseSearchService, MockSearchService

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vietphuc.api")

BASE_DIR = Path(__file__).resolve().parent
MOCK_CATALOG_PATH = BASE_DIR / "mock_catalog.json"  # giữ làm fallback/test


def load_mock_catalog():
    """Đọc mock catalog (fallback)."""
    if not MOCK_CATALOG_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy catalog: {MOCK_CATALOG_PATH}")
    with open(MOCK_CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


load_catalog = load_mock_catalog  # tên cũ, giữ để tương thích


@lru_cache
def get_search_service() -> BaseSearchService:
    """Ưu tiên AI Search. Nếu model/catalog lỗi thì dùng Mock và ghi log rõ ràng.
    Đặt biến môi trường SEARCH_BACKEND=mock để ép dùng mock."""
    if os.environ.get("SEARCH_BACKEND", "ai").lower() != "mock":
        try:
            from .ai_search_service import AISearchService
            return AISearchService()
        except Exception as e:  # noqa: BLE001
            log.error("AI Search không khởi động được: %r. Tạm dùng MockSearchService.", e)
    return MockSearchService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    svc = get_search_service()  # khởi tạo sớm để biết ngay chế độ nào đang chạy
    log.info("Search backend: %s", type(svc).__name__)
    yield


app = FastAPI(title="Diện Việt API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, service: BaseSearchService = Depends(get_search_service)):
    return SearchResponse(results=service.search(req.query, req.filters, req.top_k))


def _real_catalog():
    from .embedding_utils import CATALOG_PATH, load_catalog as load_real
    return load_real(CATALOG_PATH) if CATALOG_PATH.exists() else []


@app.get("/images/{image_id}")
def get_image(image_id: str):
    """Ảnh thật từ data/raw/ao_dai nếu có; không thì SVG placeholder."""
    clean_id = image_id
    for ext in (".svg", ".png", ".jpg", ".jpeg", ".webp"):
        if clean_id.endswith(ext):
            clean_id = clean_id[: -len(ext)]
    from .embedding_utils import resolve_image
    from .placeholder import dress_svg

    item = next((i for i in _real_catalog() if i["id"] == clean_id), None)
    if item:
        path = resolve_image(item["image"])  # chặn đường dẫn ngoài data/raw/ao_dai
        if path:
            return FileResponse(path)
    else:
        item = next((i for i in load_mock_catalog() if i.get("id") == clean_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Không tìm thấy trang phục")
    return Response(content=dress_svg(item.get("name", "Áo dài")), media_type="image/svg+xml")


@app.get("/catalog")
def get_catalog(source: str = "real"):
    """Catalog thật (mặc định). ?source=mock để xem mock catalog."""
    if source == "mock":
        return load_mock_catalog()
    real = _real_catalog()
    return real if real else load_mock_catalog()


@app.get("/health")
def health():
    svc = get_search_service()
    info = svc.info() if hasattr(svc, "info") else {}
    return {"status": "ok", "service": "dien-viet-api",
            "search_backend": "ai" if info else "mock", **info}


# Frontend: serve từ ../frontend/
FRONTEND = BASE_DIR.parent / "frontend"


@app.get("/")
def serve_frontend():
    index = FRONTEND / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Frontend chưa được build. Mở backend/index.html trực tiếp."}


if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
