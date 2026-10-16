# -*- coding: utf-8 -*-
"""
STEP 1 — Dùng PhoBERT (pipeline tách từ tiếng Việt) để phân tích content
         thành các cụm từ / token có ý nghĩa, và chia train/test 75/25.

VÌ SAO PhoBERT:
  PhoBERT (vinai/phobert-base) LUÔN được dùng kèm bước "word segmentation"
  (RDRSegmenter của VnCoreNLP) để gộp các âm tiết thành TỪ có nghĩa
  ("kho nhật" -> "kho_nhật"). Bước này tạo ra "các cụm từ có ý nghĩa".
  - Nếu cài được VnCoreNLP/py_vncorenlp  -> dùng đúng segmenter của PhoBERT.
  - Nếu không, fallback sang underthesea (cùng thuật toán RDR) -> tương đương.
  Ngoài ra ta nạp tokenizer PhoBERT (khi có mạng) để đính kèm subword,
  minh chứng dữ liệu đã ở đúng định dạng PhoBERT.

INPUT : file .xlsx có 2 sheet 'spam' và 'nonspam'. Nhãn của mỗi dòng = TÊN SHEET.
        Bỏ qua DÒNG ĐẦU (tiêu đề) của mỗi sheet, dữ liệu bắt đầu từ DÒNG 2.
        Nội dung lấy ở cột 'content' nếu có, nếu không lấy CỘT ĐẦU TIÊN.
        (Vẫn tương thích ngược: nếu không có 2 sheet trên -> đọc kiểu cũ
         1 sheet 2 cột 'content','value'.)
OUTPUT: JSON danh sách document đã tách từ + nhãn + split (train/test).

Chạy:
  python step1_segment.py --input ../longchau2.xlsx --output ../out/step1_segmented.json
"""
import argparse
import re
import sys
import unicodedata

import openpyxl

from common import DEFAULTS, EMOJI_RE, MULTISPACE_RE, URL_RE, norm_match, write_json

# Ký tự bullet / phân tách field thường gặp trong bài social (mỗi cái = 1 ranh giới)
BULLET_RE = re.compile(r"[•·►▶▷➡→⇒▪♦♥★☆※|~\\/]+")
LONGSEP_RE = re.compile(r"[-–—=_.]{3,}")   # chuỗi dài gạch/chấm/gạch dưới -> ranh giới
BOUNDARY = " . "                           # dấu ngắt câu: chặn tách-từ gộp qua field


def clean_for_segment(content):
    """Chuẩn bị văn bản để tách từ, GIỮ NGUYÊN HOA/THƯỜNG.

    - Bỏ URL; biến emoji, xuống dòng, bullet, chuỗi gạch dài thành dấu ngắt câu
      -> bộ tách từ không gộp nhầm các từ ở hai field khác nhau
      (vd 'VĨNH NGUYỄN\\n📌 BÁN' sẽ không thành 'vĩnh_nguyễn_bán').
    - KHÔNG lowercase ở đây: giữ hoa/thường để nhận diện danh từ riêng
      (vd 'Long Châu', 'Vĩnh Nguyên') -> lowercase SAU khi đã tách từ.
    """
    t = unicodedata.normalize("NFC", str(content))
    t = URL_RE.sub(" ", t)
    t = EMOJI_RE.sub(BOUNDARY, t)
    t = t.replace("\n", BOUNDARY).replace("\r", BOUNDARY)
    t = BULLET_RE.sub(BOUNDARY, t)
    t = LONGSEP_RE.sub(BOUNDARY, t)
    t = MULTISPACE_RE.sub(" ", t).strip()
    # đoạn TOÀN CHỮ HOA -> Title Case: bộ tách từ (huấn luyện trên chữ thường/
    # hỗn hợp) nhận diện danh từ riêng tốt hơn (vd 'LONG CHÂU' -> 'Long Châu').
    t = " . ".join(_detitle_segment(s) for s in t.split(" . "))
    return t


def _detitle_segment(seg):
    letters = [c for c in seg if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) >= 0.6:
        return " ".join(w.capitalize() for w in seg.split())
    return seg


def postprocess_tokens(tokens):
    """Sau tách từ: lowercase từng token, bỏ token chỉ gồm dấu câu / ranh giới."""
    out = []
    for tk in tokens:
        tk = tk.strip()
        if not tk or not re.search(r"\w", tk):   # bỏ '.', '-', '(', ':' ...
            continue
        out.append(tk.lower())
    return out


# ---------------------------------------------------------------- segmenter ---
import os as _os

VNCORENLP_BASE = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/"
VNCORENLP_WSEG_FILES = [
    "VnCoreNLP-1.2.jar",
    "models/wordsegmenter/vi-vocab",
    "models/wordsegmenter/wordsegmenter.rdr",
]


def setup_java_home(java_home=None):
    """Bảo đảm có JAVA_HOME cho pyjnius. Nếu chưa set thì tự dò từ 'java' trên PATH."""
    import shutil
    if java_home:
        _os.environ["JAVA_HOME"] = java_home
        return _os.environ["JAVA_HOME"]
    if _os.environ.get("JAVA_HOME"):
        return _os.environ["JAVA_HOME"]
    jp = shutil.which("java")
    if jp:
        # java.exe nằm ở <JAVA_HOME>/bin/java(.exe)
        _os.environ["JAVA_HOME"] = _os.path.dirname(_os.path.dirname(_os.path.realpath(jp)))
    return _os.environ.get("JAVA_HOME")


def ensure_vncorenlp_model(save_dir):
    """Tải model tách từ VnCoreNLP (jar + wordsegmenter) nếu chưa có.
    Dùng urllib (py_vncorenlp.download_model gọi 'wget' -> lỗi trên Windows)."""
    import urllib.request
    for rel in VNCORENLP_WSEG_FILES:
        dpath = _os.path.join(save_dir, rel.replace("/", _os.sep))
        if _os.path.exists(dpath) and _os.path.getsize(dpath) > 0:
            continue
        _os.makedirs(_os.path.dirname(dpath), exist_ok=True)
        print(f"[step1] tải model VnCoreNLP: {rel}", file=sys.stderr)
        urllib.request.urlretrieve(VNCORENLP_BASE + rel, dpath)


def build_segmenter(use_phobert_tokenizer, segmenter="auto",
                    vncorenlp_dir=None, java_home=None):
    """Trả về (seg_fn, backend_name, phobert_tokenizer_or_None).

    segmenter: 'auto' (ưu tiên VnCoreNLP, fallback underthesea) | 'vncorenlp' |
               'underthesea' | 'whitespace'.
    """
    seg_fn = None
    backend = None

    # 1) VnCoreNLP RDRSegmenter — bộ tách từ GỐC của PhoBERT (chính xác nhất)
    if segmenter in ("auto", "vncorenlp"):
        try:
            save_dir = _os.path.abspath(vncorenlp_dir or _os.path.join(
                _os.path.dirname(_os.path.abspath(__file__)), "..", "vncorenlp_models"))
            _os.makedirs(save_dir, exist_ok=True)
            setup_java_home(java_home)
            ensure_vncorenlp_model(save_dir)
            import py_vncorenlp  # type: ignore
            cwd = _os.getcwd()
            model = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=save_dir)
            _os.chdir(cwd)                      # VnCoreNLP tự chdir -> khôi phục lại

            def seg_fn(text):
                cur = _os.getcwd()
                try:
                    segs = model.word_segment(text)
                finally:
                    _os.chdir(cur)
                return " ".join(segs).split() if segs else []
            backend = "vncorenlp_rdrsegmenter"
        except Exception as e:
            print(f"[step1] VnCoreNLP không dùng được ({e}); chuyển fallback",
                  file=sys.stderr)
            seg_fn = None
            if segmenter == "vncorenlp":
                raise

    # 2) underthesea — cùng họ RDR, không cần Java
    if seg_fn is None and segmenter in ("auto", "underthesea"):
        try:
            from underthesea import word_tokenize  # type: ignore

            def seg_fn(text):
                return word_tokenize(text, format="text").split()
            backend = "underthesea_wordseg"
        except Exception:
            seg_fn = None

    # 3) fallback tối giản: tách theo khoảng trắng (chỉ khi 2 cái trên fail)
    if seg_fn is None:
        def seg_fn(text):
            return text.split()
        backend = "whitespace_fallback"

    tok = None
    if use_phobert_tokenizer:
        try:
            from transformers import AutoTokenizer  # type: ignore
            tok = AutoTokenizer.from_pretrained("vinai/phobert-base")
        except Exception as e:
            print(f"[step1] Bỏ qua PhoBERT tokenizer ({e})", file=sys.stderr)
            tok = None
    return seg_fn, backend, tok


# --------------------------------------------------------------------- split ---
def stratified_split(items, train_ratio, seed):
    """Chia train/test theo nhãn (stratified), tất định theo seed — không dùng random."""
    from collections import defaultdict
    by_label = defaultdict(list)
    for it in items:
        by_label[it["label"]].append(it)

    train, test = [], []
    for label, group in by_label.items():
        # xáo trộn tất định: sắp theo hash(seed, id) để tái lập được
        group_sorted = sorted(group, key=lambda x: (hash((seed, x["id"])) & 0xFFFFFFFF))
        n_train = int(round(len(group_sorted) * train_ratio))
        for i, it in enumerate(group_sorted):
            (train if i < n_train else test).append(it)
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sheet", default=None, help="tên sheet, mặc định sheet đầu")
    ap.add_argument("--train-ratio", type=float, default=DEFAULTS["train_ratio"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--no-phobert-tokenizer", action="store_true",
                    help="không nạp tokenizer PhoBERT (chạy offline nhanh hơn)")
    ap.add_argument("--segmenter", choices=["auto", "vncorenlp", "underthesea", "whitespace"],
                    default="auto",
                    help="bộ tách từ: auto (ưu tiên VnCoreNLP) | vncorenlp | underthesea | whitespace")
    ap.add_argument("--vncorenlp-dir", default=None,
                    help="thư mục chứa model VnCoreNLP (mặc định ../vncorenlp_models, tự tải nếu thiếu)")
    ap.add_argument("--java-home", default=None,
                    help="đường dẫn JDK (mặc định tự dò từ 'java' trên PATH)")
    ap.add_argument("--limit", type=int, default=0, help="giới hạn số dòng để test nhanh")
    args = ap.parse_args()

    spam_lbl = DEFAULTS["spam_label"]
    non_lbl = DEFAULTS["nonspam_label"]

    wb = openpyxl.load_workbook(args.input, read_only=True, data_only=True)
    raw = []
    dropped_empty = 0

    # Nhận diện sheet SPAM: sheet có tên đúng bằng 'spam' (không phân biệt hoa/thường).
    # MỌI sheet còn lại -> coi là NON-SPAM (kể cả tên 'nonspam', 'Social', 'ham'…).
    spam_sheets = [ws for ws in wb.worksheets if str(ws.title).strip().lower() == spam_lbl]

    if spam_sheets and len(wb.worksheets) >= 2:
        # ---- ĐỊNH DẠNG MỚI: nhãn = TÊN SHEET (spam vs. các sheet còn lại) ----
        def read_sheet(ws, label):
            n = 0
            rows = ws.iter_rows(values_only=True)
            header = next(rows, None)          # BỎ QUA dòng đầu (tiêu đề)
            # xác định cột nội dung: ưu tiên cột tên 'content'/'nội dung', mặc định cột đầu
            ci = 0
            if header is not None:
                hdr = [str(h).strip().lower() if h is not None else "" for h in header]
                for name in ("content", "nội dung", "noi dung"):
                    if name in hdr:
                        ci = hdr.index(name)
                        break
            for r in rows:                     # dữ liệu bắt đầu từ DÒNG 2
                content = r[ci] if ci < len(r) else None
                if content is None or str(content).strip() == "":
                    n += 1
                    continue
                raw.append((str(content), label))
            return n

        for ws in wb.worksheets:
            label = spam_lbl if str(ws.title).strip().lower() == spam_lbl else non_lbl
            dropped_empty += read_sheet(ws, label)
    else:
        # ---- ĐỊNH DẠNG CŨ (tương thích ngược): 1 sheet, 2 cột content/value ---
        ws = wb[args.sheet] if args.sheet else wb.worksheets[0]
        rows = ws.iter_rows(values_only=True)
        header = next(rows)
        header = [str(h).strip().lower() if h is not None else "" for h in header]
        try:
            ci = header.index("content")
            vi = header.index("value")
        except ValueError:
            ci, vi = 0, 1  # giả định cột 0=content, 1=value
        for r in rows:
            content = r[ci] if ci < len(r) else None
            value = r[vi] if vi < len(r) else None
            if content is None or str(content).strip() == "":
                dropped_empty += 1
                continue
            label = str(value).strip().lower() if value is not None else ""
            if label not in (spam_lbl, non_lbl):
                dropped_empty += 1
                continue
            raw.append((str(content), label))
    wb.close()

    if args.limit:
        raw = raw[: args.limit]

    seg_fn, backend, tok = build_segmenter(
        not args.no_phobert_tokenizer, segmenter=args.segmenter,
        vncorenlp_dir=args.vncorenlp_dir, java_home=args.java_home)
    print(f"[step1] Segmentation backend = {backend}; "
          f"phobert_tokenizer = {'on' if tok else 'off'}", file=sys.stderr)

    # tách từ trên nội dung UNIQUE để tăng tốc (dữ liệu có nhiều bản trùng)
    seg_cache = {}
    docs = []
    for i, (content, label) in enumerate(raw):
        cnorm = norm_match(content)          # bản lowercase để hiển thị/tham chiếu
        seg_input = clean_for_segment(content)  # bản GIỮ hoa/thường để tách từ
        if seg_input in seg_cache:
            tokens = seg_cache[seg_input]
        else:
            try:
                raw_tokens = seg_fn(seg_input)
            except Exception:
                raw_tokens = seg_input.split()
            tokens = postprocess_tokens(raw_tokens)   # lowercase + bỏ dấu câu/ranh giới
            seg_cache[seg_input] = tokens
        doc = {
            "id": i,
            "label": label,
            "content": content,
            "content_norm": cnorm,
            "tokens": tokens,                    # các "từ có nghĩa" (âm tiết nối bằng _)
            "segmented_text": " ".join(tokens),  # định dạng đầu vào chuẩn của PhoBERT
        }
        if tok is not None and i < 50:  # đính kèm subword mẫu (minh hoạ, tránh nặng)
            doc["phobert_subwords"] = tok.tokenize(doc["segmented_text"])
        docs.append(doc)
        if (i + 1) % 2000 == 0:
            print(f"[step1] segmented {i+1}/{len(raw)}", file=sys.stderr)

    train, test = stratified_split(docs, args.train_ratio, args.seed)
    train_ids = {d["id"] for d in train}
    for d in docs:
        d["split"] = "train" if d["id"] in train_ids else "test"

    from collections import Counter
    def dist(ds):
        return dict(Counter(d["label"] for d in ds))

    out = {
        "meta": {
            "step": 1,
            "input": args.input,
            "segmentation_backend": backend,
            "phobert_tokenizer": bool(tok),
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "n_total": len(docs),
            "n_dropped_empty_or_bad_label": dropped_empty,
            "dist_total": dist(docs),
            "dist_train": dist(train),
            "dist_test": dist(test),
            "n_train": len(train),
            "n_test": len(test),
        },
        "documents": docs,
    }
    write_json(args.output, out)
    print(f"[step1] OK -> {args.output} | total={len(docs)} "
          f"train={len(train)} test={len(test)} | dist={dist(docs)}", file=sys.stderr)


if __name__ == "__main__":
    main()
