# Diện Việt: backend

Luồng: `User -> Frontend -> POST /search -> AISearchService -> Sentence Transformer -> Hybrid Ranking -> Cultural KB + Guardrail -> Top K`

## Cài đặt (PowerShell, trong D:\vietphuc5.5)
```
py -m pip install -r requirements.txt
```
Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (local, 384 chiều). Lần đầu chạy sẽ tải model về máy (cần mạng một lần).

## Build embedding (offline, chạy lại thoải mái)
```
py scripts/build_embeddings.py          # bỏ qua nếu catalog không đổi
py scripts/build_embeddings.py --force
```
Tạo `data/embeddings/catalog_embeddings.npy` và `catalog_embedding_ids.json` (id theo thứ tự dòng). Sửa catalog xong nhớ build lại; quên thì server cảnh báo và tự tính trong bộ nhớ.

## Chạy
```
py -m uvicorn backend.main:app --reload --port 8000     # mở http://localhost:8000
```
`GET /health` cho biết đang dùng `search_backend: ai` hay `mock`. Ép dùng mock: `$env:SEARCH_BACKEND="mock"`.

## API
- `POST /search` `{query, filters{occasion,color,style,pattern}, top_k}` -> `results[]` (giữ nguyên các trường cũ; thêm tùy chọn `warning_details`, `cultural_sources`)
- `GET /images/{id}` ảnh thật từ `data/raw/ao_dai` (chặn đường dẫn ngoài thư mục này), `GET /catalog`, `GET /health`, `/docs`

## Xếp hạng
`Final = 0.6 x cosine(query, mẫu) + 0.4 x attribute_match` (trọng số dịp 3, màu 3, phong cách 2.5, họa tiết 2). Filter là lọc cứng: OR trong nhóm, AND giữa nhóm ("đỏ" khớp "đỏ tươi"; có thể gõ `red`, `tet`...).

## Cultural layer (không dùng LLM)
- `data/cultural/knowledge.json`: KB nháp, **chưa có nguồn**, cần bổ sung `source` rồi đặt `verified: true`.
- `data/cultural/rules.json`: luật cảnh báo (không loại kết quả), mỗi cảnh báo có `severity`.

## Test
```
py scripts/test_search.py            # model thật
py scripts/test_search.py --api      # qua API đang chạy
```
