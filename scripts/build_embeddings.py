"""Tạo embedding OFFLINE cho data/catalog/catalog.json (Sentence Transformer chạy local, không gọi Gemini).
Chạy lại thoải mái: catalog không đổi thì bỏ qua; ghi file nguyên tử nên không bị hỏng/nhân đôi.

  py scripts/build_embeddings.py          # bỏ qua nếu không đổi
  py scripts/build_embeddings.py --force  # buộc tính lại
Kết quả: data/embeddings/catalog_embeddings.npy + catalog_embedding_ids.json
"""
import argparse
import logging
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.embedding_utils import CATALOG_PATH, MODEL_NAME, build  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    print("Model  :", MODEL_NAME)
    print("Catalog:", CATALOG_PATH)
    r = build(force=a.force)
    print(f"[{r['status']}] {r['count']} mẫu, dim={r['dim']} -> {r['dir']}")
