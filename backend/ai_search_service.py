"""AI search: Sentence Transformer (local) + Hybrid Ranking + Cultural layer. Không dùng Gemini."""
import logging
import re

import numpy as np

from .cultural import CulturalLayer
from .embedding_utils import (CATALOG_PATH, EMBED_DIR, FIELDS, MODEL_NAME, SbertEncoder,
                              hit, load_catalog, load_embeddings, norm)
from .schemas import Filters, SearchResult
from .search_service import BaseSearchService

log = logging.getLogger("vietphuc.search")

W_SEM, W_ATTR = 0.6, 0.4
FIELD_W = {"occasion": 3.0, "color": 3.0, "style": 2.5, "pattern": 2.0}

# Áo dài nam / trẻ em có trong dataset nhưng ẩn khỏi kết quả mặc định (không xóa dữ liệu).
# Chỉ hiện khi câu hỏi nhắc đến rõ ràng.
HIDDEN = {
    "ao_dai_nam": ["ao dai nam", "nam gioi", "chu re", "dan ong", "con trai"],
    "ao_dai_tre_em": ["tre em", "cho be", "be gai", "be trai", "cho con", "thieu nhi"],
}


def _category(image: str) -> str:
    """Danh mục lấy từ tên file ảnh: 'ao_dai_trẻ_em__000001.jpg' -> 'ao_dai_tre_em'."""
    name = image.replace("\\", "/").split("/")[-1].split("__")[0]
    return norm(name).strip().replace(" ", "_")


# Chip/filter bằng tiếng Anh (ví dụ color=red) được quy về nhãn tiếng Việt của catalog
EN = {"red": "đỏ", "pink": "hồng", "yellow": "vàng", "white": "trắng", "blue": "xanh", "green": "xanh",
      "purple": "tím", "orange": "cam", "tet": "Tết", "wedding": "Cưới hỏi", "festival": "Lễ hội",
      "school": "Đi học", "traditional": "truyền thống", "modern": "cách tân", "youthful": "trẻ trung",
      "elegant": "thanh lịch", "luxury": "sang trọng"}

# Từ đồng nghĩa (không dấu) để nhận ra thuộc tính trong câu hỏi; màu xử lý riêng vì dễ trùng từ thường
SYN = {
    "occasion": {"Tết": ["tet", "xuan", "nam moi"],
                 "Cưới hỏi": ["dam cuoi", "le cuoi", "an hoi", "cuoi hoi", "ao dai cuoi", "co dau", "tiec cuoi"],
                 "Lễ hội": ["le hoi", "hoi xuan"], "Đi học": ["di hoc", "nu sinh", "hoc sinh"],
                 "Dạo phố": ["di choi", "dao pho"], "Chụp ảnh kỷ niệm": ["chup anh", "chup hinh", "ky yeu"],
                 "Sự kiện": ["su kien", "di tiec"]},
    "style": {"trẻ trung": ["tre trung", "nang dong", "ca tinh", "gen z", "genz"],
              "cách tân": ["cach tan", "hien dai", "gen z", "genz"],
              "thanh lịch": ["thanh lich", "nhe nhang", "thanh nha"], "sang trọng": ["sang trong", "sang chanh"],
              "truyền thống": ["truyen thong", "co dien", "xua"]},
    "pattern": {"hoa sen": ["hoa sen"], "hoa mai": ["hoa mai"], "hoa đào": ["hoa dao"], "thổ cẩm": ["tho cam"],
                "thêu hoa": ["theu hoa"], "hoa văn gấm": ["hoa van gam", "vai gam"], "trơn": ["tron", "don gian"]},
}
BASE_COLORS = ["đỏ", "vàng", "trắng", "hồng", "xanh", "tím", "cam", "đen", "nâu", "xám", "be"]


def _color_found(q_raw: str, label: str) -> bool:
    """Màu phải gõ có dấu ('đỏ', 'trắng') hoặc đi sau 'màu/áo dài'. Tránh 'trang phục' -> trắng, 'hỏng' -> hồng."""
    plain = norm(label).strip()
    return bool(re.search(rf"(?<!\w){re.escape(label)}(?!\w)|(?:màu|mau|áo dài|ao dai) {re.escape(plain)}(?!\w)", q_raw))


class AISearchService(BaseSearchService):
    def __init__(self, encoder=None, catalog_path=None, embeddings_dir=None, cultural=None):
        self.encoder = encoder or SbertEncoder(MODEL_NAME)
        self.items = load_catalog(catalog_path or CATALOG_PATH)
        self.emb, self.emb_source = load_embeddings(self.items, self.encoder, embeddings_dir or EMBED_DIR)
        self.encoder.encode(["khởi động"])  # nạp model ngay lúc khởi động để lỗi lộ ra sớm (main sẽ fallback)
        self.cultural = cultural or CulturalLayer()
        self.cats = [_category(it["image"]) for it in self.items]
        self.vals = [{f: [norm(v) for v in it[f]] for f in FIELDS} for it in self.items]
        self.colors = sorted({*BASE_COLORS, *(v for it in self.items for v in it["color"])})
        self.vocab = {f: {norm(v): v for it in self.items for v in it[f]} for f in ("occasion", "style", "pattern")}
        log.info("AISearchService: %d mẫu, embeddings từ %s", len(self.items), self.emb_source)

    def info(self) -> dict:
        return {"model": MODEL_NAME, "items": len(self.items), "embedding_dim": int(self.emb.shape[1]),
                "embedding_source": self.emb_source}

    @staticmethod
    def _canon(v: str) -> str:
        return EN.get(v.strip().lower(), v)

    def _soft_terms(self, query: str, filters: dict) -> dict:
        q_raw, q = query.lower(), norm(query)
        out = {f: {} for f in FIELDS}
        for f, table in SYN.items():
            for label, words in table.items():
                if any(f" {w} " in q for w in words):
                    out[f][norm(label)] = label
            for nv, label in self.vocab[f].items():
                if len(nv.split()) > 1 and nv in q:
                    out[f].setdefault(nv, label)
        for label in self.colors:
            if _color_found(q_raw, label):
                out["color"][norm(label)] = label
        for f in FIELDS:
            for v in filters[f]:
                out[f][norm(v)] = v
            t = out[f]
            out[f] = {k: v for k, v in t.items() if not any(k != u and k in u for u in t)}  # giữ cụm dài nhất
        return out

    def search(self, query: str, filters: Filters, top_k: int) -> list[SearchResult]:
        flt = {f: [self._canon(v) for v in getattr(filters, f)] for f in FIELDS}
        hard = {f: [norm(v) for v in flt[f]] for f in FIELDS}
        # Hard filter: OR trong một nhóm, AND giữa các nhóm
        cand = [i for i in range(len(self.items))
                if all(not hard[f] or any(hit(w, v) for w in hard[f] for v in self.vals[i][f]) for f in FIELDS)]
        qn = norm(query)
        shown = {c for c, words in HIDDEN.items() if any(f" {w} " in qn for w in words)}
        cand = [i for i in cand if self.cats[i] not in HIDDEN or self.cats[i] in shown]
        if not cand:
            return []
        soft = self._soft_terms(query, flt)
        has_q = bool(query.strip())
        sem = np.zeros(len(cand), dtype=np.float32)
        if has_q:
            qv = self.encoder.encode([query.strip()])[0]
            sem = np.clip(self.emb[cand] @ qv, 0.0, 1.0)  # embedding đã chuẩn hóa nên tích vô hướng = cosine
        ctx = query + " " + " ".join(v for f in FIELDS for v in flt[f])

        rows = []
        for n, i in enumerate(cand):
            got = tot = 0.0
            matched = []
            for f in FIELDS:
                for t, label in soft[f].items():
                    tot += FIELD_W[f]
                    if any(hit(t, v) for v in self.vals[i][f]):
                        got += FIELD_W[f]
                        matched.append(label)
            attr = got / tot if tot else None
            if has_q and attr is not None:
                final = W_SEM * float(sem[n]) + W_ATTR * attr
            elif has_q:
                final = float(sem[n])          # câu hỏi không có thuộc tính nhận ra: chỉ dùng ngữ nghĩa
            elif attr is not None:
                final = attr                   # chỉ có chip/filter
            else:
                final = 0.5                    # không có gì để xếp hạng
            rows.append((min(max(final, 0.0), 1.0), self.items[i]["id"], i, matched, float(sem[n])))
        rows.sort(key=lambda r: (-r[0], r[1]))

        out = []
        for final, _, i, matched, s in rows[:top_k]:
            it = self.items[i]
            bits = ([f"khớp {', '.join(matched)}"] if matched else []) + ([f"độ tương đồng ngữ nghĩa {s:.0%}"] if has_q else [])
            reason = ("Vì " + "; ".join(bits) + " với yêu cầu của bạn.") if bits else "Mẫu nổi bật để bạn tham khảo."
            culture, sources = self.cultural.notes_for(it)
            details = [{"warning": w, "severity": "low", "rule_id": "annotation"} for w in it["warnings"]]
            details += self.cultural.check(ctx, it)
            warnings = list(dict.fromkeys(d["warning"] for d in details))
            out.append(SearchResult(
                id=it["id"], name=it["name"], image=f"/images/{it['id']}", description=it["description"],
                score=round(final, 4), occasion=it["occasion"], color=it["color"], style=it["style"],
                pattern=it["pattern"], reason=reason, cultural_context=culture, warnings=warnings,
                warning_details=details, cultural_sources=sources))
        return out