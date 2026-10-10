"""Bước 1 dữ liệu: tìm và tải CHỈ ảnh áo dài từ Dangindev/viet-cultural-vqa.
Không dùng load_dataset, không tải toàn bộ repo. Không sửa dataset gốc (chỉ đọc).

  pip install huggingface_hub pillow
  python fetch_ao_dai.py --list        # chỉ liệt kê cấu trúc + file khớp, KHÔNG tải
  python fetch_ao_dai.py --download    # tải đúng các file khớp, kiểm tra, tạo metadata/catalog
"""
import argparse, hashlib, json, random, re, shutil, unicodedata
from collections import Counter
from pathlib import Path

REPO, TYPE = "Dangindev/viet-cultural-vqa", "dataset"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
DATA = Path(__file__).resolve().parents[1] / "data"
RAW, META, CAT = DATA / "raw" / "ao_dai", DATA / "metadata", DATA / "catalog"


def norm(s):
    s = unicodedata.normalize("NFD", s.lower().replace("đ", "d"))
    return re.sub(r"[^a-z0-9]+", "_", "".join(c for c in s if unicodedata.category(c) != "Mn"))


def is_ao_dai(path):
    """Khớp theo TÊN THƯ MỤC/FILE chứa 'ao_dai' hoặc 'aodai' (không giả định vị trí)."""
    n = "_" + norm(path) + "_"
    return Path(path).suffix.lower() in IMG_EXT and ("_ao_dai_" in n or "_aodai_" in n)


def list_files(api):
    return [f for f in api.list_repo_tree(REPO, repo_type=TYPE, recursive=True) if hasattr(f, "size")]


def show_structure(files):
    print(f"Tổng số file trong repo: {len(files)}")
    imgs = [f for f in files if Path(f.path).suffix.lower() in IMG_EXT]
    print(f"Tổng số ảnh: {len(imgs)}  | thư mục cấp 1/2 chứa ảnh:")
    for k, v in sorted(Counter("/".join(f.path.split("/")[:3][:-1]) for f in imgs).items()):
        print(f"  {k:55s} {v}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true"); g.add_argument("--download", action="store_true")
    a = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    revision = api.dataset_info(REPO).sha
    files = list_files(api)
    show_structure(files)
    hits = sorted((f for f in files if is_ao_dai(f.path)), key=lambda f: f.path)
    folders = sorted({str(Path(f.path).parent) for f in hits})
    print(f"\nẢnh áo dài khớp: {len(hits)} | dung lượng khai báo: {sum(f.size for f in hits)/1e6:.1f} MB")
    print("Thư mục khớp:", *folders, sep="\n  ")
    for f in hits[:5]: print("  vd:", f.path)
    if a.list or not hits:
        return print("\n(--list: chưa tải gì. Kiểm tra thư mục khớp ở trên rồi chạy --download)")

    from PIL import Image
    for d in (RAW, META, CAT): d.mkdir(parents=True, exist_ok=True)
    rows, seen = [], {}
    for i, f in enumerate(hits, 1):
        local = RAW / f.path.split("images/", 1)[-1].replace("/", "__")
        row = {"id": f"ao_dai_{i:04d}", "hf_path": f.path, "local_path": str(local.relative_to(DATA.parent)),
               "size_bytes": f.size, "source": REPO, "hf_revision": revision}
        try:
            cached = hf_hub_download(REPO, f.path, repo_type=TYPE, revision=revision)
            shutil.copyfile(cached, local)
            with Image.open(local) as im:
                im.verify()
            with Image.open(local) as im:
                row.update(width=im.width, height=im.height, format=im.format)
            row["sha256"] = hashlib.sha256(local.read_bytes()).hexdigest()
            if row["sha256"] in seen:
                row["status"] = "duplicate"; row["duplicate_of"] = seen[row["sha256"]]
            else:
                seen[row["sha256"]] = row["id"]; row["status"] = "ok"
        except Exception as e:
            row.update(status="corrupt_or_failed", error=repr(e)[:200])
        rows.append(row)
        print(f"[{i}/{len(hits)}] {row['status']:9s} {f.path}")

    ok = [r for r in rows if r["status"] == "ok"]
    (META / "source_metadata.json").write_text(json.dumps(
        {"source": REPO, "revision": revision, "folders_used": folders, "items": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    (CAT / "catalog.json").write_text(json.dumps(
        [{"id": r["id"], "image": r["local_path"], "source": REPO} for r in ok], ensure_ascii=False, indent=1), encoding="utf-8")

    # ảnh mẫu: contact sheet 4x3
    pick = random.Random(0).sample(ok, min(12, len(ok)))
    if pick:
        sheet = Image.new("RGB", (4 * 240, 3 * 240), "white")
        for n, r in enumerate(pick):
            with Image.open(DATA.parent / r["local_path"]) as im:
                im = im.convert("RGB"); im.thumbnail((236, 236))
                sheet.paste(im, ((n % 4) * 240 + 2, (n // 4) * 240 + 2))
        sheet.save(META / "samples.png")
    c = Counter(r["status"] for r in rows)
    print("\n== BÁO CÁO ==", f"\nTìm thấy: {len(hits)} | tải: {len(rows)} | ok: {c['ok']} | trùng: {c['duplicate']} | hỏng/lỗi: {c['corrupt_or_failed']}",
          f"\nDung lượng trên đĩa (raw/ao_dai): {sum(p.stat().st_size for p in RAW.iterdir())/1e6:.1f} MB",
          "\nThư mục HF đã dùng:", *folders, "\nCatalog:", len(ok), "ảnh -> data/catalog/catalog.json | ảnh mẫu: data/metadata/samples.png")


if __name__ == "__main__":
    main()
