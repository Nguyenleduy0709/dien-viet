# Dữ liệu Áo dài - Dự án "Diện Việt"

Thư mục lưu trữ dữ liệu ảnh và metadata dành riêng cho Áo dài Việt Nam, phục vụ giai đoạn MVP.

---

## 1. Nguồn dữ liệu (Dataset Source)
- **Dataset gốc:** [Dangindev/viet-cultural-vqa](https://huggingface.co/datasets/Dangindev/viet-cultural-vqa) trên Hugging Face.
- **Quy mô toàn bộ dataset gốc:** ~28,500 file (~12.9 GB) gồm nhiều lĩnh vực (ẩm thực, kiến trúc, trang phục, lễ hội,...).
- **Phạm vi thu thập:** Chỉ lọc và tải chính xác các danh mục thuộc Áo dài nằm trong `images/trang_phuc/ao_dai_*`, tuyệt đối không tải toàn bộ dataset.

---

## 2. Thống kê dữ liệu đã tải
- **Tổng số ảnh áo dài phát hiện trong dataset gốc:** 342 ảnh (thuộc 6 phân loại, mỗi loại 57 ảnh):
  - `ao_dai_Việt_Nam`: 57 ảnh
  - `ao_dai_cach_tân`: 57 ảnh
  - `ao_dai_cuới`: 57 ảnh
  - `ao_dai_học_sinh`: 57 ảnh
  - `ao_dai_nam`: 57 ảnh
  - `ao_dai_trẻ_em`: 57 ảnh
- **Số lượng ảnh đã tải (MVP):** 200 ảnh (phân bổ đều qua các danh mục).
- **Định dạng file:** `.jpg`, `.png` (đã verify bằng `PIL.Image`).
- **Tổng dung lượng trên đĩa:** ~57.91 MB.
- **Trùng lặp (Duplicate):** 0 (được kiểm tra và lọc tự động bằng SHA-256 hash).

---

## 3. Cấu trúc thư mục
```text
D:\vietphuc5.5\data\
├── raw\
│   └── ao_dai\                           # Chứa 200 file ảnh áo dài đã tải
│       ├── ao_dai_Việt_Nam__000001.jpg
│       ├── ao_dai_cach_tân__000001.jpg
│       ├── ao_dai_cuới__000001.jpg
│       └── ...
├── metadata\
│   └── source_metadata.json              # Metadata chi tiết từng ảnh (ID, hf_path, sha256, kích thước WxH, định dạng)
└── README.md                             # Tài liệu này
```

---

## 4. Cách chạy lại script download

Script tải dữ liệu được đặt tại `D:\vietphuc5.5\scripts\download_ao_dai.py`.

### Liệt kê file và danh mục (không tải):
```powershell
py scripts/download_ao_dai.py --list-only
```

### Tải 200 ảnh áo dài (mặc định cho MVP):
```powershell
py scripts/download_ao_dai.py --limit 200 --workers 8
```

### Tính năng script:
- Tự động bỏ qua các ảnh đã có trên đĩa (`cached`), không tải lại.
- Tự động kiểm tra SHA-256 để phát hiện và ngăn trùng lặp (`duplicate`).
- Tải đa luồng (`--workers`) tăng tốc độ tải mà không tốn nhiều băng thông.
- Tự động cập nhật `data/metadata/source_metadata.json`.

