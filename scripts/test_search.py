"""In Top 5 + score cho 5 query tiếng Việt và test filter.
  py scripts/test_search.py             # gọi thẳng AISearchService với model thật
  py scripts/test_search.py --api       # gọi POST http://localhost:8000/search (backend đang chạy)
  py scripts/test_search.py --offline   # encoder GIẢ (kiểm tra luồng, KHÔNG phản ánh chất lượng ngữ nghĩa)
"""
import argparse
import json
import sys
import tempfile
import urllib.request
import zlib
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUERIES = ["Tôi muốn áo dài đỏ trẻ trung để đi chơi Tết", "Áo dài thanh lịch cho đám cưới",
           "Tôi muốn áo dài truyền thống màu hồng", "Áo dài có hoa sen", "Áo dài hiện đại cho Gen Z"]
FILTER_TESTS = [("color=red", {"color": ["red"]}), ("occasion=tet", {"occasion": ["tet"]}),
                ("color=red + occasion=tet", {"color": ["red"], "occasion": ["tet"]}),
                ("(đỏ OR hồng) AND Tết", {"color": ["đỏ", "hồng"], "occasion": ["Tết"]})]


class HashEncoder:  # chỉ để test offline
    def encode(self, texts):
        from backend.embedding_utils import norm
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for r, t in enumerate(texts):
            t = norm(t)
            for k in range(len(t) - 2):
                out[r, zlib.crc32(t[k:k + 3].encode()) % 384] += 1
            out[r] /= (np.linalg.norm(out[r]) or 1)
        return out


def make_search(a):
    empty = {"occasion": [], "color": [], "style": [], "pattern": []}
    if a.api:
        def s(q, f=None, k=5):
            body = json.dumps({"query": q, "filters": {**empty, **(f or {})}, "top_k": k}).encode()
            req = urllib.request.Request("http://localhost:8000/search", body, {"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req))["results"]
        return s
    from backend.ai_search_service import AISearchService
    from backend.schemas import Filters
    if a.offline:
        from backend import embedding_utils as eu
        enc, tmp = HashEncoder(), tempfile.mkdtemp()
        print("build lần 1:", eu.build(enc, None, tmp)["status"], "| lần 2:", eu.build(enc, None, tmp)["status"])
        svc = AISearchService(enc, embeddings_dir=tmp)
    else:
        svc = AISearchService()
    return lambda q, f=None, k=5: [r.model_dump() for r in svc.search(q, Filters(**(f or {})), k)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", action="store_true"); ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    search = make_search(a)
    for q in QUERIES:
        print(f"\n=== {q}")
        for n, r in enumerate(search(q), 1):
            print(f"{n}. {r['score']:.3f} {r['id']} | {r['name'][:42]} | {r['color']} {r['style']} {r['occasion']}"
                  + (f" | ⚠ {len(r['warnings'])}" if r["warnings"] else ""))
    print("\n##### FILTER")
    for label, f in FILTER_TESTS:
        r = search("", f, 50)
        print(f"{label:28s} -> {len(r)} kết quả:", [x["id"] for x in r[:6]])
    print("\nThử top_k:", {k: len(search("áo dài", None, k)) for k in (1, 3, 5)})
