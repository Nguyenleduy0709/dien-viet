"""Cultural layer: Knowledge Base (JSON) + Guardrail (luật cố định). Không dùng LLM."""
import json
import logging

from .embedding_utils import DATA_DIR, FIELDS, hit, norm

log = logging.getLogger("vietphuc.cultural")
CULTURAL_DIR = DATA_DIR / "cultural"
UNKNOWN = "Chưa xác định"


def _read(name: str) -> dict:
    p = CULTURAL_DIR / name
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        log.warning("Thiếu %s, bỏ qua lớp này.", p)
        return {}


class CulturalLayer:
    def __init__(self):
        self.entries = _read("knowledge.json").get("entries", [])
        self.rules = _read("rules.json").get("rules", [])

    def notes_for(self, item: dict) -> tuple[str, list[str]]:
        """Thông tin văn hóa từ KB. Không có mục nào khớp thì dùng ghi chú từ ảnh, cuối cùng là 'Chưa xác định'."""
        vals = {f: [norm(v) for v in item[f]] for f in FIELDS}
        text = norm(item["name"] + " " + item["description"])
        picked = []
        for order, e in enumerate(self.entries):
            for f, wanted in e.get("match", {}).items():
                ok = (any(norm(w) in text for w in wanted) if f == "text_any"
                      else any(hit(norm(w), v) for w in wanted for v in vals.get(f, [])))
                if ok:
                    picked.append((-e.get("priority", 0), order, e)); break
        picked.sort(key=lambda t: t[:2])
        top = [e for _, _, e in picked[:2]]
        if top:
            return " ".join(e["text"] for e in top), [e["id"] for e in top]
        cc = item.get("cultural_context", "")
        return (cc if cc and norm(cc).strip() != "khong xac dinh" else UNKNOWN), []

    def check(self, context: str, item: dict) -> list[dict]:
        """context = câu hỏi + giá trị chip. Trả về cảnh báo (không loại kết quả)."""
        ctx = norm(context)
        out = []
        for r in self.rules:
            ok = True
            for key, vals in r.get("when", {}).items():
                if key == "query_any":
                    ok = any(norm(p) in ctx for p in vals)
                elif key.startswith("item_") and key.endswith("_any"):
                    have = [norm(v) for v in item.get(key[5:-4], [])]
                    ok = any(hit(norm(w), h) for w in vals for h in have)
                elif key.startswith("item_") and key.endswith("_all_in"):
                    have = [norm(v) for v in item.get(key[5:-7], [])]
                    ok = bool(have) and all(any(hit(norm(w), h) for w in vals) for h in have)
                else:
                    ok = False
                if not ok:
                    break
            if ok:
                out.append({"warning": r["message"], "severity": r.get("severity", "low"), "rule_id": r["id"]})
        return out
