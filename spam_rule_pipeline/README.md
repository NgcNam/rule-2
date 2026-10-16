# Spam Rulebase Mining Pipeline

Chuyển prompt yêu cầu thành mã nguồn: pipeline **6 bước độc lập**, mỗi bước
đọc/ghi **JSON** để bạn kiểm tra từng bước (output bước trước = input bước sau).

Sinh ra bộ **rulebase keyword (OR/AND/NOT)** để chặn spam với ràng buộc:
**precision > 99%, recall > 30%** — ưu tiên *thà bỏ sót còn hơn xoá nhầm*.

## Cài đặt

```bash
pip install openpyxl scikit-learn scipy numpy underthesea flask
# tuỳ chọn (đúng "PhoBERT"): pip install transformers torch
```

## 🐳 Chạy bằng Docker (khuyến nghị — không cần cài Python/thư viện)

Từ thư mục gốc dự án (`rule-2`, nơi có `Dockerfile` + `docker-compose.yml`):

```bash
docker compose up --build        # build & chạy, mở http://localhost:5000
# chạy nền:   docker compose up -d --build
# dừng:       docker compose down
```

- Giao diện web có **2 chế độ import** (bấm nút để chọn):
  - **2 file riêng**: 1 file **SPAM** + 1 file **NON-SPAM**.
  - **1 file (2 sheet)**: 1 file Excel có sẵn 2 sheet `spam` & `nonspam`.
  - Mỗi file/sheet: dòng 1 là tiêu đề, nội dung từ dòng 2 — lấy cột
    `content`/`nội dung` nếu có, ngược lại cột đầu.
  - Bấm **Chạy pipeline** → chạy đủ 6 bước → xem precision/recall + rulebase →
    **Tải report Excel**.
- **Danh mục đã chạy**: trang chính liệt kê các lần chạy trước (report Excel đã
  lưu trong `./output/`), bấm để tải lại.
- Kết quả mỗi lần chạy lưu ở host tại `./output/<tên-dataset>/` (volume gắn vào
  `/data` trong container).
- Image gọn: dùng `underthesea` + gom nhóm lexical. PhoBERT embedding là tuỳ chọn
  (thiếu `transformers/torch` thì step4 tự fallback), nên **không** đưa torch vào
  image mặc định.

## 🌐 Web local (import Excel → xem kết quả → tải report)

```bash
cd spam_rule_pipeline
python webapp.py                       # mở http://127.0.0.1:5000
python webapp.py --host 0.0.0.0 --port 8000   # cho máy khác trong LAN truy cập
```

Trên web: **chọn 2 file Excel** (1 file `spam` + 1 file `nonspam`, mỗi file có
dòng tiêu đề, dữ liệu từ dòng 2) → bấm **Chạy pipeline**
→ theo dõi 6 bước chạy trực tiếp → xem **precision/recall + toàn bộ rulebase
(OR/AND/NOT)** → bấm **Tải report Excel**. Có mục *Tuỳ chọn nâng cao* để chỉnh bộ
tách từ, `min_gram`, `kw_min_precision`, `and_cover`, `sim_threshold`, số rule/keyword,
bật/tắt gom nhóm PhoBERT. Mỗi lần chạy tạo 1 folder (tên = tên file, tự thêm
`_v1/_v2`) tại thư mục gốc dự án, file bên trong mang đuôi = tên folder.

### Tách từ bằng VnCoreNLP RDRSegmenter (bộ tách từ GỐC của PhoBERT — khuyến nghị)

```bash
pip install py_vncorenlp        # kèm pyjnius
# cần có JDK (Java 8+). Kiểm tra: java -version
```

- Model (~28MB) **tự tải** lần đầu vào `../vncorenlp_models` (không cần `wget`).
- `JAVA_HOME` **tự dò** từ `java` trên PATH; hoặc chỉ định `--java-home "C:\path\to\jdk"`.
- Nếu thiếu Java/py_vncorenlp, pipeline tự fallback sang `underthesea`.

## Chạy từng bước (mỗi bước chạy độc lập, output là JSON)

```bash
cd spam_rule_pipeline

# B1 — PhoBERT: tách content thành cụm từ có nghĩa + chia train/test 75/25
python step1_segment.py --input ../longchau2.xlsx --output ../out/step1_segmented.json
#   --segmenter auto|vncorenlp|underthesea|whitespace   (auto ưu tiên VnCoreNLP)
#   --no-phobert-tokenizer : bỏ tải tokenizer PhoBERT (nhanh/offline)
#   --vncorenlp-dir / --java-home : tuỳ chỉnh nơi model & JDK

# B2 — TF-IDF + spam_score cho từng cụm từ
python step2_score.py --input ../out/step1_segmented.json --output ../out/step2_scores.json

# B3 — sinh keyword ứng viên (OR/AND/NOT nguyên liệu) theo spam_score
python step3_candidates.py --step1 ../out/step1_segmented.json \
    --step2 ../out/step2_scores.json --output ../out/step3_candidates.json

# B4 — chọn & ghép rule bảo đảm precision>99%, recall>30%
python step4_select_rules.py --step1 ../out/step1_segmented.json \
    --step3 ../out/step3_candidates.json --output ../out/step4_rules.json

# B5 — ép ràng buộc (<=20 rule, <=20 kw, min-gram) + cấu hình conflict
python step5_finalize.py --input ../out/step4_rules.json --output ../out/step5_final_rules.json

# B6 — backtest + xuất Excel theo khung format.xlsx
python step6_export.py --step1 ../out/step1_segmented.json \
    --step5 ../out/step5_final_rules.json --output ../out/report.xlsx
```

Hoặc chạy 1 phát cả 6 bước:

```bash
python run_all.py --input ../longchau2.xlsx --outdir ../out
```

## Ánh xạ prompt → bước

| Prompt | Bước | Ghi chú |
|---|---|---|
| 1. PhoBERT tách cụm từ có nghĩa | step1 | RDRSegmenter (VnCoreNLP) → fallback underthesea; nạp tokenizer PhoBERT khi có |
| 2. TF-IDF + spam_score | step2 | `spam_score = precision^α · log1p(df_spam) · (0.5+tfidf)` |
| 3. Sinh rulebase OR/AND/NOT | step3 + step4 | ứng viên → greedy set-cover ghép rule |
| 4. precision>99%, recall>30% | step4 | greedy có ràng buộc precision; keyword rất sạch (>=97%) trên train |
| — gom keyword CHUNG Ý NGHĨA | step4 | nhúng PhoBERT → cluster cosine (fallback token nội dung) |
| — lọc keyword VÔ NGHĨA | step3 | bỏ hư từ/mảnh vụn, rác font (𝟎𝟗/ℎ/ɪ), cụm toàn số |
| — khử keyword CHỨA NHAU | step3+4+5 | giữ cụm ngắn/tổng quát, bỏ cụm dài bao nó |
| — sinh AND & NOT an toàn recall | step4 | NOT: cụm chỉ có ở nonspam bẩn; AND: anchor phủ >=95% doc spam của rule |
| 5. ≤20 rule, ≤20 kw, min-gram, conflict | step5 | ép cứng ràng buộc + logic conflict |
| output theo format.xlsx | step6 | 6 sheet: Summary, RuleStats, AllResults, FalsePositives, FalseNegatives, RuleExamples |

## Cơ chế phân loại (conflict)

1. **Có rule SPAM match** → `spam` (định đoạt).
2. **Chỉ rule NON-SPAM match** → `nonspam` (xác nhận an toàn).
3. **Cả hai match** → `uncertain_spam` (trạng thái trung gian, coi như **SPAM**).
4. **Không match gì** → `uncertain` (review tay, coi như nonspam).

> Rule NON-SPAM **không bao giờ override** SPAM → precision chỉ phụ thuộc rule spam.

## Tinh chỉnh (đổi qua tham số dòng lệnh, mặc định ở `common.py`)

- `--min-gram` (mặc định 3): số gram tối thiểu mỗi keyword (không được ít quá).
- `--kw-min-precision` (0.97): độ sạch tối thiểu của keyword trên train. Nới nhẹ
  khỏi 1.0 để rule có chút FP → AND/NOT có việc dọn; greedy vẫn ép precision
  TOÀN CỤC >=99%. Đặt `1.0` để quay lại chế độ keyword tuyệt đối sạch.
- `--target-precision` / `--target-recall` (0.99 / 0.30): ngưỡng ràng buộc.
- `--max-rules` / `--max-kw` (20 / 20): trần số rule & keyword mỗi rule.
- `--alpha` (step2, 3.0): độ ưu tiên precision trong `spam_score`.

### Gom nhóm ngữ nghĩa & AND/NOT (step4)

- `--sim-threshold` (0.62): ngưỡng cosine gom keyword cùng nghĩa (cao = nhóm chặt hơn).
- `--no-embedding`: bỏ PhoBERT, gom theo token nội dung chia sẻ (nhanh, offline).
- `--max-and` (2) / `--max-not` (4): trần số AND / NOT sinh mỗi rule.
- `--and-cover` (0.95): AND phải có mặt ở >= tỉ lệ này doc spam của rule. Cao =
  an toàn recall (thường không sinh AND cho rule đa dạng); hạ xuống (vd 0.8) để
  ép sinh nhiều AND hơn, ĐÁNH ĐỔI recall.
- `--stopword-df-ratio` (step3, 0.03): token xuất hiện > tỉ lệ này doc train bị
  coi là "đời thường" (không phải nội dung phân biệt spam).

## Kết quả backtest hiện tại (tập test 25% giữ riêng)

- (VnCoreNLP) Precision **99.45%** ✅ · Recall **41.04%** ✅ · FP=9 · 20 rule · 170 keyword
- (underthesea) Precision **99.63%** ✅ · Recall **41.11%** ✅ · FP=6
- Cả hai đều tốt hơn bản tham chiếu format.xlsx (recall 36.8% / F1 53.7%). VnCoreNLP
  cho chất lượng TÁCH TỪ cao hơn (đúng bộ segmenter gốc của PhoBERT).

### Lưu ý quan trọng về tính nhất quán khi match

`spam_score`/precision của keyword được đo bằng **n-gram token** (step2/3) nên
lúc backtest (step6) BẮT BUỘC match cùng đơn vị token đó — dùng
`common.match_tokens()` (tokenize `\w+`, giữ '_' của từ ghép, bỏ dấu câu) và so
khớp **dãy con token liên tiếp**. Nếu match bằng "substring văn bản thô" thì một
keyword sạch trên train vẫn có thể dính nonspam → precision tụt giả tạo.
Bộ lọc *content-anchor* (`--stopword-df-ratio`) loại cụm từ toàn từ đời thường
(vd "các bạn đã", "cách tốt nhất") vốn chỉ "sạch" do dữ liệu lệch spam.
