"""Tiện ích dùng chung cho scripts/build_embeddings.py và AISearchService:
cùng model, cùng cách ghép văn bản, cùng cách đọc catalog, cùng cách tìm ảnh."""
import hashlib
import json
import logging
import os
import re
import unicodedata
from pathlib import Path

import numpy as np

log = logging.getLogger("vietphuc.embedding")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
FIELDS = ("occasion", "color", "style", "pattern")

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("VIETPHUC_DATA_DIR", PROJECT_DIR / "data"))
CATALOG_PATH = DATA_DIR / "catalog" / "catalog.json"
EMBED_DIR = DATA_DIR / "embeddings"
RAW_DIR = DATA_DIR / "raw" / "ao_dai"
EMB_FILE, IDS_FILE = "catalog_embeddings.npy", "catalog_embedding_ids.json"


def norm(s) -> str:
    """Bỏ dấu, hạ chữ thường, đệm khoảng trắng hai đầu để so khớp từ nguyên vẹn."""
    s = unicodedata.normalize("NFD", str(s).lower().replace("đ", "d"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip() + " "


def hit(a: str, b: str) -> bool:
    """a, b đã norm. 'đỏ' khớp 'đỏ tươi' (chứa nhau theo từ)."""
    return bool(a.strip()) and bool(b.strip()) and (a in b or b in a)


def _list(v) -> list[str]:
    if isinstance(v, str):
        v = re.split(r"[,;|]", v)
    return [str(x).strip() for x in (v or []) if str(x).strip()]


def load_catalog(path=None) -> list[dict]:
    path = Path(path or CATALOG_PATH)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        data = data.get("items", [])
    items, seen = [], set()
    for raw in data:
        if not isinstance(raw, dict) or not raw.get("id") or str(raw["id"]) in seen:
            continue
        it = {
            "id": str(raw["id"]), "name": str(raw.get("name") or raw["id"]),
            "image": str(raw.get("image") or ""), "description": str(raw.get("description") or "").strip(),
            **{f: _list(raw.get(f)) for f in FIELDS},
            "cultural_context": str(raw.get("cultural_context") or "").strip(),
            "warnings": _list(raw.get("warnings")),
        }
        if it["description"] or any(it[f] for f in FIELDS):
            seen.add(it["id"]); items.append(it)
    if not items:
        raise ValueError(f"Catalog rỗng hoặc sai cấu trúc: {path}")
    return items


def resolve_image(rel: str) -> Path | None:
    """Đường dẫn tương đối trong catalog -> file thật. Chỉ cho phép trong data/raw/ao_dai."""
    if not rel:
        return None
    p = (PROJECT_DIR / rel.replace("\\", "/")).resolve()
    try:
        p.relative_to(RAW_DIR.resolve())
    except ValueError:
        return None
    return p if p.is_file() else None


def build_text(it: dict) -> str:
    parts = [it["description"]] + [
        f"{lab}: {', '.join(it[f])}" for lab, f in
        (("Dịp", "occasion"), ("Màu", "color"), ("Phong cách", "style"), ("Họa tiết", "pattern")) if it[f]]
    return ". ".join(p for p in parts if p)


def fingerprint(model: str, items) -> str:
    h = hashlib.sha256(model.encode())
    for it in items:
        h.update(f"\n{it['id']}\t{build_text(it)}".encode())
    return h.hexdigest()


class SbertEncoder:
    """Sentence Transformer chạy local. Lần đầu tải model về máy, sau đó dùng cache."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name, self._m = model_name, None

    def encode(self, texts):
        if self._m is None:
            from sentence_transformers import SentenceTransformer
            self._m = SentenceTransformer(self.model_name)
        return np.asarray(self._m.encode(list(texts), normalize_embeddings=True, show_progress_bar=False),
                          dtype=np.float32)


def _atomic(path: Path, writer):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        writer(f)
    os.replace(tmp, path)


def build(encoder=None, catalog_path=None, out_dir=None, force=False, model_name=MODEL_NAME) -> dict:
    """Tạo embeddings. Chạy lại nhiều lần vẫn ra đúng một bộ dữ liệu (ghi đè nguyên tử, không nhân đôi)."""
    out = Path(out_dir or EMBED_DIR)
    out.mkdir(parents=True, exist_ok=True)
    items = load_catalog(catalog_path)
    fp = fingerprint(model_name, items)
    meta_p, emb_p = out / IDS_FILE, out / EMB_FILE
    if not force and meta_p.exists() and emb_p.exists():
        m = json.loads(meta_p.read_text(encoding="utf-8"))
        if m.get("fingerprint") == fp and np.load(emb_p).shape[0] == len(items):
            return {"status": "up_to_date", "count": len(items), "dim": m.get("dim"), "dir": str(out)}
    emb = (encoder or SbertEncoder(model_name)).encode([build_text(i) for i in items]).astype(np.float32)
    meta = {"model": model_name, "count": len(items), "dim": int(emb.shape[1]),
            "fingerprint": fp, "ids": [i["id"] for i in items]}
    _atomic(emb_p, lambda f: np.save(f, emb))
    _atomic(meta_p, lambda f: f.write(json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8")))
    return {"status": "built", "count": len(items), "dim": meta["dim"], "dir": str(out)}


def load_embeddings(items, encoder, out_dir=None, model_name=MODEL_NAME) -> tuple[np.ndarray, str]:
    """Trả về (ma trận embedding, nguồn: 'file' hoặc 'memory')."""
    out = Path(out_dir or EMBED_DIR)
    try:
        meta = json.loads((out / IDS_FILE).read_text(encoding="utf-8"))
        emb = np.load(out / EMB_FILE)
        if meta.get("fingerprint") == fingerprint(model_name, items) and emb.shape[0] == len(items):
            return emb, "file"
        log.warning("Embeddings cũ so với catalog. Hãy chạy: py scripts/build_embeddings.py. Tạm tính trong bộ nhớ.")
    except FileNotFoundError:
        log.warning("Chưa có embeddings. Hãy chạy: py scripts/build_embeddings.py. Tạm tính trong bộ nhớ.")
    return encoder.encode([build_text(i) for i in items]).astype(np.float32), "memory"
