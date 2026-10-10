"""Script download ảnh áo dài từ dataset Dangindev/viet-cultural-vqa trên Hugging Face.

Chỉ tải các ảnh thuộc danh mục Áo dài, không tải toàn bộ dataset (~12.9GB).
Hỗ trợ:
- Tự động kiểm tra file đã tồn tại để tránh download lại.
- Tránh trùng lặp (deduplication bằng sha256).
- Kiểm tra tính toàn vẹn của ảnh bằng PIL.Image.
- Tải đa luồng (concurrent workers) giúp tốc độ nhanh và có tiến trình rõ ràng.
- Ghi log tiến trình và lưu metadata vào data/metadata/source_metadata.json.
- Giới hạn số lượng ảnh (mặc định 200 ảnh cho MVP).
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

# Đảm bảo UTF-8 trên Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ID = "Dangindev/viet-cultural-vqa"
REPO_TYPE = "dataset"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# 6 danh mục thư mục con chính xác của áo dài trong dataset
AO_DAI_SUBDIRS = [
    "images/trang_phuc/ao_dai_Việt_Nam",
    "images/trang_phuc/ao_dai_cach_tân",
    "images/trang_phuc/ao_dai_cuới",
    "images/trang_phuc/ao_dai_học_sinh",
    "images/trang_phuc/ao_dai_nam",
    "images/trang_phuc/ao_dai_trẻ_em",
]

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "ao_dai"
META_DIR = DATA_DIR / "metadata"


def parse_args():
    parser = argparse.ArgumentParser(description="Tải ảnh áo dài từ Dangindev/viet-cultural-vqa")
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Số lượng ảnh tối đa cần tải (mặc định: 200 ảnh cho MVP)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Số luồng tải song song (mặc định: 8)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Chỉ liệt kê danh sách file tìm thấy, không tải về đĩa",
    )
    return parser.parse_args()


def get_ao_dai_files(api: HfApi):
    """Lấy danh sách các file ảnh áo dài từ repository."""
    all_files = api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    matched = []
    for f in all_files:
        if any(f.startswith(subdir + "/") for subdir in AO_DAI_SUBDIRS):
            if Path(f).suffix.lower() in VALID_EXTS:
                matched.append(f)
    return sorted(matched)


def download_single_file(item, revision, existing_items, lock, stats):
    idx, hf_path = item
    relative_part = hf_path.replace("images/trang_phuc/", "").replace("/", "__")
    local_filename = relative_part
    local_path = RAW_DIR / local_filename

    # Kiểm tra file đã tồn tại và hợp lệ
    if local_path.exists() and local_path.stat().st_size > 0:
        try:
            with Image.open(local_path) as img:
                img.verify()
            sha256_val = hashlib.sha256(local_path.read_bytes()).hexdigest()
            with lock:
                stats["skipped"] += 1
                stats["seen_hashes"][sha256_val] = local_filename
            record = existing_items.get(hf_path, {
                "id": f"ao_dai_{idx:04d}",
                "hf_path": hf_path,
                "local_filename": local_filename,
                "local_path": str(local_path.relative_to(BASE_DIR)),
                "size_bytes": local_path.stat().st_size,
                "sha256": sha256_val,
                "status": "cached",
            })
            return record
        except Exception:
            pass

    # Download từ Hugging Face
    try:
        cached_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=hf_path,
            repo_type=REPO_TYPE,
            revision=revision,
        )
        shutil.copyfile(cached_path, local_path)

        with Image.open(local_path) as img:
            img.verify()
        with Image.open(local_path) as img:
            width, height = img.width, img.height
            img_format = img.format

        sha256_val = hashlib.sha256(local_path.read_bytes()).hexdigest()
        file_size = local_path.stat().st_size

        with lock:
            if sha256_val in stats["seen_hashes"]:
                status = "duplicate"
                stats["duplicate"] += 1
            else:
                stats["seen_hashes"][sha256_val] = local_filename
                status = "ok"
            stats["downloaded"] += 1
            current_done = stats["downloaded"] + stats["skipped"]
            total_target = stats["total"]
            print(f"[{current_done:03d}/{total_target:03d}] Tải xong: {local_filename} ({file_size / 1024:.1f} KB)")

        return {
            "id": f"ao_dai_{idx:04d}",
            "hf_path": hf_path,
            "local_filename": local_filename,
            "local_path": str(local_path.relative_to(BASE_DIR)),
            "size_bytes": file_size,
            "width": width,
            "height": height,
            "format": img_format,
            "sha256": sha256_val,
            "status": status,
        }
    except Exception as e:
        with lock:
            stats["error"] += 1
            print(f"[!] Lỗi tải {hf_path}: {e}")
        return None


def main():
    args = parse_args()
    api = HfApi()

    print(f"[*] Đang truy vấn repository '{REPO_ID}'...")
    try:
        revision = api.repo_info(REPO_ID, repo_type=REPO_TYPE).sha
    except Exception:
        revision = "main"

    ao_dai_files = get_ao_dai_files(api)
    total_found = len(ao_dai_files)
    print(f"[*] Tìm thấy tổng cộng {total_found} ảnh áo dài trong 6 danh mục chuẩn:")
    folder_counts = Counter("/".join(f.split("/")[:-1]) for f in ao_dai_files)
    for folder, count in folder_counts.items():
        print(f"    - {folder}: {count} ảnh")

    if args.list_only:
        print("\n[*] 10 đường dẫn file mẫu:")
        for f in ao_dai_files[:10]:
            print(f"    {f}")
        return

    # Lựa chọn phân bổ đều các thư mục
    target_limit = min(args.limit, total_found)
    selected_files = []
    by_category = {}
    for f in ao_dai_files:
        cat = "/".join(f.split("/")[:-1])
        by_category.setdefault(cat, []).append(f)

    cat_keys = list(by_category.keys())
    idx = 0
    while len(selected_files) < target_limit:
        cat = cat_keys[idx % len(cat_keys)]
        if by_category[cat]:
            selected_files.append(by_category[cat].pop(0))
        idx += 1
        if all(len(v) == 0 for v in by_category.values()):
            break

    print(f"\n[*] Bắt đầu tải {len(selected_files)} ảnh (workers={args.workers})...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    metadata_file = META_DIR / "source_metadata.json"
    existing_items = {}
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                old_meta = json.load(f)
                for it in old_meta.get("items", []):
                    existing_items[it["hf_path"]] = it
        except Exception:
            pass

    stats = {
        "total": len(selected_files),
        "downloaded": 0,
        "skipped": 0,
        "duplicate": 0,
        "error": 0,
        "seen_hashes": {},
    }
    lock = threading.Lock()

    tasks = [(i, path) for i, path in enumerate(selected_files, 1)]
    records = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                download_single_file,
                t,
                revision,
                existing_items,
                lock,
                stats,
            )
            for t in tasks
        ]
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                records.append(res)

    records.sort(key=lambda x: x["id"])

    # Lưu metadata
    meta_output = {
        "dataset_source": f"https://huggingface.co/datasets/{REPO_ID}",
        "revision": revision,
        "total_ao_dai_found": total_found,
        "total_selected": len(selected_files),
        "total_downloaded": stats["downloaded"],
        "total_skipped_existing": stats["skipped"],
        "total_duplicates": stats["duplicate"],
        "total_errors": stats["error"],
        "categories_used": AO_DAI_SUBDIRS,
        "items": records,
    }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(meta_output, f, ensure_ascii=False, indent=2)

    total_disk_bytes = sum(f.stat().st_size for f in RAW_DIR.glob("*.*") if f.is_file())
    print("\n" + "=" * 50)
    print("BÁO CÁO KẾT QUẢ:")
    print(f"- Tổng số ảnh áo dài trong dataset gốc: {total_found}")
    print(f"- Số ảnh tải mới: {stats['downloaded']}")
    print(f"- Số ảnh đã có sẵn (bỏ qua): {stats['skipped']}")
    print(f"- Số file ảnh hợp lệ trên đĩa: {len(list(RAW_DIR.glob('*.*')))}")
    print(f"- Tổng dung lượng trên đĩa: {total_disk_bytes / (1024 * 1024):.2f} MB")
    print(f"- File metadata đã lưu: {metadata_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()
