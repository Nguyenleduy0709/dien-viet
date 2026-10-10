"""Search service cho bản Mock MVP."""

import json
import re
import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path

from .schemas import Filters, SearchResult


BASE_DIR = Path(__file__).resolve().parent
CATALOG_PATH = BASE_DIR / "mock_catalog.json"


def load_catalog():
    """Đọc mock catalog từ mock_catalog.json."""
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file catalog: {CATALOG_PATH}"
        )

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def norm(s: str) -> str:
    """Chuẩn hóa chuỗi để tìm kiếm không dấu."""
    s = unicodedata.normalize(
        "NFD",
        s.lower().replace("đ", "d")
    )

    s = "".join(
        c for c in s
        if unicodedata.category(c) != "Mn"
    )

    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)

    return " " + s.strip() + " "


class BaseSearchService(ABC):

    @abstractmethod
    def search(
        self,
        query: str,
        filters: Filters,
        top_k: int,
    ) -> list[SearchResult]:
        pass


FIELDS = (
    "occasion",
    "color",
    "style",
    "pattern",
)


WEIGHT = {
    "occasion": 3.0,
    "color": 3.0,
    "style": 2.5,
    "pattern": 2.0,
}


# Từ khóa mock.
# Đây chỉ là luật đơn giản cho MVP, chưa phải ML.
LEXICON = {

    "occasion": {
        "Tết": [
            "tet",
            "xuan",
            "nam moi",
        ],
        "Cưới hỏi": [
            "cuoi",
            "an hoi",
            "dam cuoi",
        ],
        "Lễ hội": [
            "le hoi",
            "hoi xuan",
            "le",
            "chua",
        ],
        "Đi học": [
            "di hoc",
            "nu sinh",
            "truong",
        ],
    },

    "color": {
        "đỏ": [
            "do",
            "do tuoi",
            "do do",
        ],
        "vàng": [
            "vang",
        ],
        "trắng": [
            "trang",
        ],
        "hồng": [
            "hong",
        ],
        "xanh": [
            "xanh",
            "xanh ngoc",
        ],
    },

    "style": {
        "trẻ trung": [
            "tre trung",
            "nang dong",
            "cach tan",
            "hien dai",
            "ca tinh",
        ],
        "thanh lịch": [
            "thanh lich",
            "nhe nhang",
            "sang trong",
            "thanh nha",
        ],
        "truyền thống": [
            "truyen thong",
            "co dien",
            "xua",
        ],
    },

    "pattern": {
        "hoa mai": [
            "hoa mai",
            "mai",
        ],
        "hoa đào": [
            "hoa dao",
            "dao",
        ],
        "hoa sen": [
            "hoa sen",
            "sen",
        ],
        "thổ cẩm": [
            "tho cam",
        ],
        "chim hạc": [
            "chim hac",
            "hac",
        ],
        "trơn": [
            "tron",
            "don gian",
        ],
        "hoa văn gấm": [
            "gam",
        ],
    },
}


class MockSearchService(BaseSearchService):

    def __init__(self, catalog=None):
        self.catalog = (
            catalog
            if catalog is not None
            else load_catalog()
        )

    @staticmethod
    def _parse_query(query: str):
        """
        Phân tích query thành các tiêu chí:
        occasion / color / style / pattern
        """

        q = norm(query)

        found = {
            field: set()
            for field in FIELDS
        }

        for field, table in LEXICON.items():

            for value, words in table.items():

                if any(
                    f" {word} " in q
                    for word in words
                ):
                    found[field].add(value)

        return found

    def search(
        self,
        query: str,
        filters: Filters,
        top_k: int,
    ) -> list[SearchResult]:

        # -----------------------------
        # 1. Parse filter từ frontend
        # -----------------------------

        wanted = {
            field: {
                norm(value)
                for value in getattr(filters, field)
            }
            for field in FIELDS
        }

        # -----------------------------
        # 2. Parse query tự nhiên
        # -----------------------------

        soft = self._parse_query(query)

        # Chip/filter cũng được tính vào điểm
        for field in FIELDS:

            for value in getattr(filters, field):
                soft[field].add(value)

        # Chuẩn hóa
        soft_n = {
            field: {
                norm(value): value
                for value in soft[field]
            }
            for field in FIELDS
        }

        has_criteria = any(
            soft_n[field]
            for field in FIELDS
        )

        # -----------------------------
        # 3. Search
        # -----------------------------

        rows = []

        for item in self.catalog:

            have = {
                field: {
                    norm(value)
                    for value in item.get(field, [])
                }
                for field in FIELDS
            }

            # -------------------------
            # Hard filter
            # -------------------------

            if any(
                wanted[field]
                and not (
                    wanted[field] & have[field]
                )
                for field in FIELDS
            ):
                continue

            # -------------------------
            # Scoring
            # -------------------------

            got = 0.0
            maximum = 0.0
            hits = []

            for field in FIELDS:

                for value, label in soft_n[field].items():

                    maximum += WEIGHT[field]

                    if value in have[field]:

                        got += WEIGHT[field]

                        hits.append(label)

            # Có query nhưng không match gì
            if has_criteria and got == 0:
                continue

            # -------------------------
            # Final score
            # -------------------------

            if has_criteria:

                score = round(
                    0.55 + 0.43 * got / maximum,
                    2,
                )

                reason = (
                    "Khớp với "
                    + ", ".join(hits)
                    + " trong yêu cầu của bạn."
                )

            else:

                score = 0.5

                reason = (
                    "Mẫu nổi bật để bạn tham khảo."
                )

            rows.append(
                (
                    score,
                    item["id"],
                    item,
                    reason,
                )
            )

        # -----------------------------
        # 4. Sort
        # -----------------------------

        rows.sort(
            key=lambda row: (
                -row[0],
                row[1],
            )
        )

        # -----------------------------
        # 5. Convert response
        # -----------------------------

        return [
            SearchResult(
                id=item["id"],
                name=item["name"],
                image=f"/images/{item['id']}.svg",
                description=item["description"],
                score=score,
                occasion=item["occasion"],
                color=item["color"],
                style=item["style"],
                pattern=item["pattern"],
                reason=reason,
                cultural_context=item["cultural_context"],
                warnings=item["warnings"],
                visual=item.get("visual", {}),
            )
            for score, _, item, reason
            in rows[:top_k]
        ]