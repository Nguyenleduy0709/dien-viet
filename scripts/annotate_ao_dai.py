"""Script sử dụng Gemini Vision API để phân tích các ảnh áo dài đã tải.

- Đọc ảnh từ: data/raw/ao_dai/
- Trích xuất thông tin có cấu trúc (JSON schema)
- Lưu kết quả vào: data/catalog/catalog.json
- Lưu lỗi (nếu có) vào: data/catalog/annotation_errors.json
- Tự động bỏ qua các ảnh đã xử lý thành công (idempotent / resumable)
- Retry với exponential backoff khi gặp lỗi mạng/API
- Không hardcode API key, đọc từ biến môi trường GEMINI_API_KEY
- Dùng trực tiếp chuẩn Gemini REST API v1beta qua urllib/requests (không phụ thuộc SDK cồng kềnh)
"""

import argparse
import base64
import json
import mimetypes
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Tự động nạp .env nếu có (không bắt buộc thư viện ngoài)
def _load_dotenv_if_exists(base_path: Path):
    env_path = base_path / ".env"
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("\"'")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass

# Đảm bảo UTF-8 trên Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "ao_dai"
CATALOG_DIR = DATA_DIR / "catalog"
CATALOG_FILE = CATALOG_DIR / "catalog.json"
ERROR_FILE = CATALOG_DIR / "annotation_errors.json"
METADATA_FILE = DATA_DIR / "metadata" / "source_metadata.json"

DEFAULT_MODEL = "gemini-3.8-flash"
API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

SYSTEM_PROMPT = """Bạn là chuyên gia nghiên cứu văn hóa và trang phục truyền thống Việt Nam.
Nhiệm vụ của bạn là phân tích hình ảnh áo dài được cung cấp và trích xuất thông tin khách quan, chuẩn xác theo định dạng JSON.

QUY TẮC CẦN TUÂN THỦ NGHIÊM NGẶT:
1. "name": Tên hoặc mô tả ngắn gọn, súc tích của mẫu áo dài (Ví dụ: "Áo dài cách tân đỏ hoa mai", "Áo dài cưới gấm thêu", "Áo dài trắng nữ sinh").
2. "description": Mô tả ngoại hình của trang phục (dáng áo, cổ áo, tà áo, quần đi kèm, chất liệu nếu quan sát rõ).
3. "occasion": Danh sách các dịp phù hợp, chọn từ hoặc tương tự các từ: "Tết", "Cưới hỏi", "Lễ hội", "Đi học", "Sự kiện", "Dạo phố", "Chụp ảnh kỷ niệm".
4. "color": Danh sách màu sắc chủ đạo quan sát được (ví dụ: ["đỏ"], ["vàng"], ["trắng"], ["xanh"], ["hồng"], ["nhiều màu"]).
5. "style": Danh sách phong cách, chọn từ: "truyền thống", "cách tân", "trẻ trung", "thanh lịch", "sang trọng".
6. "pattern": Danh sách họa tiết quan sát được trên áo (ví dụ: "hoa mai", "hoa đào", "hoa sen", "thổ cẩm", "hoa văn gấm", "trơn", "chấm bi", "thêu hoa").
7. "cultural_context": Thông tin văn hóa, bối cảnh truyền thống có thể suy ra MỘT CÁCH THẬN TRỌNG từ hình ảnh. Tuyệt đối KHÔNG tự bịa thông tin lịch sử không thể xác định. Nếu không có bối cảnh đặc thù, ghi "Không xác định".
8. "warnings": Danh sách lưu ý văn hóa, bối cảnh sử dụng hoặc cảnh báo nếu hình ảnh không đủ chi tiết (ví dụ: ["Cần chú ý độ dài tà áo khi đi lễ chùa", "Màu đỏ đám cưới thường ưu tiên cô dâu"]). Nếu không có, để danh sách rỗng [].

Chỉ trả về DUY NHẤT một chuỗi JSON hợp lệ với cấu trúc sau, không kèm bất kỳ giải thích nào khác ngoài JSON:
{
  "name": "...",
  "description": "...",
  "occasion": [...],
  "color": [...],
  "style": [...],
  "pattern": [...],
  "cultural_context": "...",
  "warnings": [...]
}
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Gán nhãn ảnh áo dài bằng Gemini Vision API")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Tên model Gemini")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số ảnh xử lý (mặc định: tất cả)")
    parser.add_argument("--delay", type=float, default=1.0, help="Thời gian chờ giữa các request (giây)")
    parser.add_argument("--check-sample", type=int, default=0, help="In ngẫu nhiên N kết quả mẫu từ catalog và thoát")
    return parser.parse_args()


def load_catalog_data():
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    catalog = []
    if CATALOG_FILE.exists():
        try:
            with open(CATALOG_FILE, "r", encoding="utf-8") as f:
                catalog = json.load(f)
        except Exception:
            catalog = []

    errors = []
    if ERROR_FILE.exists():
        try:
            with open(ERROR_FILE, "r", encoding="utf-8") as f:
                errors = json.load(f)
        except Exception:
            errors = []

    return catalog, errors


def save_catalog_data(catalog, errors):
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CATALOG_FILE, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    with open(ERROR_FILE, "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)


def get_image_base64(image_path: Path):
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "image/jpeg"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return mime_type, encoded


class DailyQuotaExceeded(Exception):
    """Hết quota theo ngày của free tier, không thể thử lại hôm nay."""


def call_gemini_vision(api_key: str, model: str, image_path: Path, max_retries: int = 3):
    url = API_URL_TEMPLATE.format(model=model, api_key=api_key)
    mime_type, img_data = get_image_base64(image_path)

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": SYSTEM_PROMPT},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": img_data,
                        }
                    },
                    {"text": "Hãy phân tích hình ảnh áo dài này và trả về JSON theo yêu cầu:"},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "response_mime_type": "application/json",
        },
    }

    req_data = json.dumps(payload).encode("utf-8")

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                res_body = response.read().decode("utf-8")
                res_json = json.loads(res_body)

                usage_meta = res_json.get("usageMetadata", {})
                candidates = res_json.get("candidates", [])
                if not candidates:
                    raise ValueError(f"Không có candidate nào được trả về: {res_json}")

                content_parts = candidates[0].get("content", {}).get("parts", [])
                if not content_parts:
                    raise ValueError(f"Candidate không có content parts: {candidates[0]}")

                text_response = content_parts[0].get("text", "")
                parsed_result = json.loads(text_response)

                return parsed_result, usage_meta

        except urllib.error.HTTPError as he:
            err_msg = he.read().decode("utf-8", errors="ignore")
            if he.code == 429 and "PerDay" in err_msg:
                raise DailyQuotaExceeded(err_msg[:300])
            if he.code in [429, 500, 503] and attempt < max_retries:
                wait_time = attempt * 3
                print(f"    [!] Lỗi HTTP {he.code}. Thử lại lần {attempt}/{max_retries} sau {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"HTTP {he.code}: {err_msg}")
        except Exception as e:
            if attempt < max_retries:
                wait_time = attempt * 2
                print(f"    [!] Lỗi: {e}. Thử lại lần {attempt}/{max_retries} sau {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e

    raise RuntimeError("Vượt quá số lần retry tối đa.")


def normalize_annotation(ann: dict, item_id: str, image_rel_path: str):
    """Đảm bảo các trường đúng chuẩn và hợp lệ."""
    return {
        "id": item_id,
        "image": image_rel_path,
        "name": str(ann.get("name") or "Áo dài truyền thống"),
        "description": str(ann.get("description") or "Không có mô tả chi tiết"),
        "occasion": [str(x) for x in ann.get("occasion", [])] if isinstance(ann.get("occasion"), list) else [],
        "color": [str(x) for x in ann.get("color", [])] if isinstance(ann.get("color"), list) else [],
        "style": [str(x) for x in ann.get("style", [])] if isinstance(ann.get("style"), list) else [],
        "pattern": [str(x) for x in ann.get("pattern", [])] if isinstance(ann.get("pattern"), list) else [],
        "cultural_context": str(ann.get("cultural_context") or "Không xác định"),
        "warnings": [str(x) for x in ann.get("warnings", [])] if isinstance(ann.get("warnings"), list) else [],
    }


def main():
    _load_dotenv_if_exists(BASE_DIR)
    args = parse_args()
    catalog, errors = load_catalog_data()

    if args.check_sample > 0:
        if not catalog:
            print("[!] Catalog hiện đang rỗng.")
            return
        n = min(args.check_sample, len(catalog))
        samples = random.sample(catalog, n)
        print(f"\n===== {n} MẪU ANNOTATION TỪ CATALOG =====")
        for i, s in enumerate(samples, 1):
            print(f"\n--- Mẫu {i}: {s.get('id')} ({s.get('name')}) ---")
            print(json.dumps(s, ensure_ascii=False, indent=2))
        return

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("[!] LỖI: Biến môi trường GEMINI_API_KEY chưa được thiết lập!")
        print("    Vui lòng thiết lập bằng lệnh:")
        print("    $env:GEMINI_API_KEY=\"AIzaSy...\"")
        print("    hoặc chạy lại sau khi đã set biến môi trường.")
        sys.exit(1)

    # Đọc metadata nếu có để giữ ID nhất quán
    metadata_map = {}
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                meta_json = json.load(f)
                for item in meta_json.get("items", []):
                    metadata_map[item.get("local_filename")] = item.get("id")
        except Exception:
            pass

    # Thu thập danh sách ảnh trong raw/ao_dai/
    image_files = sorted([
        f for f in RAW_DIR.glob("*.*")
        if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
    ])

    total_images = len(image_files)
    print(f"[*] Tìm thấy {total_images} ảnh trong '{RAW_DIR}'")

    processed_ids = {item["id"] for item in catalog}
    processed_images = {item.get("image") for item in catalog}
    error_paths = {err.get("file") for err in errors}

    # Lọc danh sách cần xử lý
    to_process = []
    for f in image_files:
        rel_path = str(f.relative_to(BASE_DIR))
        item_id = metadata_map.get(f.name, f.stem)
        if item_id in processed_ids or rel_path in processed_images:
            continue
        to_process.append((item_id, f, rel_path))

        # Xen kẽ các danh mục (ao_dai_cach_tân, ao_dai_cuới, ...) để dừng giữa chừng vẫn cân bằng
    from collections import defaultdict
    groups = defaultdict(list)
    for entry in to_process:
        groups[entry[1].name.split("__")[0]].append(entry)
    queues = list(groups.values())
    to_process = [q[i] for i in range(max(map(len, queues), default=0)) for q in queues if i < len(q)]
    if args.limit:
        to_process = to_process[:args.limit]

    print(f"[*] Đã xử lý trước đó: {len(catalog)} ảnh.")
    print(f"[*] Cần xử lý tiếp: {len(to_process)} ảnh.")

    if not to_process:
        print("[*] Tất cả các ảnh đã được xử lý xong!")
        save_catalog_data(catalog, errors)
        return

    success_count = 0
    fail_count = 0
    total_prompt_tokens = 0
    total_candidate_tokens = 0

    print(f"[*] Bắt đầu gọi Gemini Vision ({args.model})...\n")

    for idx, (item_id, img_path, rel_path) in enumerate(to_process, 1):
        print(f"[{idx:03d}/{len(to_process):03d}] Đang phân tích: {img_path.name}...")
        try:
            ann_raw, usage = call_gemini_vision(api_key, args.model, img_path)
            normalized = normalize_annotation(ann_raw, item_id, rel_path)
            catalog.append(normalized)
            processed_ids.add(item_id)
            processed_images.add(rel_path)

            # Xóa khỏi error nếu trước đó bị lỗi mà nay thành công
            errors = [e for e in errors if e.get("file") != img_path.name]

            success_count += 1
            p_tokens = usage.get("promptTokenCount", 0)
            c_tokens = usage.get("candidatesTokenCount", 0)
            total_prompt_tokens += p_tokens
            total_candidate_tokens += c_tokens

            print(f"    -> Thành công: {normalized['name']} | Màu: {normalized['color']} | Token: {p_tokens + c_tokens}")

            # Lưu định kỳ từng ảnh để không mất dữ liệu nếu bị gián đoạn
            save_catalog_data(catalog, errors)

            if args.delay > 0:
                time.sleep(args.delay)

        except DailyQuotaExceeded:
            print("\n[!] Đã hết quota theo ngày. Dừng lại, chạy tiếp vào ngày mai.")
            save_catalog_data(catalog, errors)
            break
        except Exception as e:
            fail_count += 1
            print(f"    [!] Thất bại: {e}")
            errors.append({
                "id": item_id,
                "file": img_path.name,
                "path": rel_path,
                "error": str(e),
                "timestamp": time.time(),
            })
            save_catalog_data(catalog, errors)

    print("\n" + "=" * 55)
    print("BÁO CÁO HOÀN THÀNH:")
    print(f"- Tổng số ảnh trong thư mục: {total_images}")
    print(f"- Đã xử lý thành công đợt này: {success_count}")
    print(f"- Lỗi đợt này: {fail_count}")
    print(f"- Tổng số ảnh trong catalog: {len(catalog)}")
    print(f"- Tổng số lỗi trong log: {len(errors)}")
    print(f"- Token ước tính: prompt={total_prompt_tokens}, candidates={total_candidate_tokens}")
    print(f"- Catalog lưu tại: {CATALOG_FILE}")
    print(f"- Error log lưu tại: {ERROR_FILE}")
    print("=" * 55)


if __name__ == "__main__":
    main()