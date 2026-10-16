# -*- coding: utf-8 -*-
"""
common.py — tiện ích dùng chung cho toàn bộ pipeline sinh rulebase spam.

Mọi bước (step1..step6) đều import từ đây để đảm bảo:
  - Chuẩn hoá văn bản GIỐNG NHAU giữa lúc sinh keyword và lúc match keyword.
  - Đọc/ghi JSON thống nhất (UTF-8, không escape unicode) để bạn dễ kiểm tra.
"""
import json
import os
import re
import sys
import unicodedata

# Console Windows mặc định là cp1252 -> in tiếng Việt sẽ lỗi UnicodeEncodeError.
# Ép stdout/stderr sang UTF-8 (mọi step đều import common nên áp dụng toàn cục).
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ----------------------------------------------------------------------------
# Cấu hình mặc định của pipeline (có thể override qua tham số dòng lệnh mỗi bước)
# ----------------------------------------------------------------------------
DEFAULTS = {
    "seed": 42,
    "train_ratio": 0.75,        # 75% train, 25% backtest (test)
    "spam_label": "spam",
    "nonspam_label": "nonspam",

    # step2 — TF-IDF / spam_score
    "ngram_min": 2,
    "ngram_max": 4,
    "min_df": 2,                # 1 cụm từ phải xuất hiện >=3 doc train mới xét
    "spam_score_alpha": 3.0,    # số mũ ưu tiên precision trong spam_score

    # step3/4 — sinh & chọn rule
    "min_gram": 4,              # mỗi keyword tối thiểu 3 gram (không được ít quá)
    "min_gram_fallback": 2,     # nếu recall chưa đủ, cho phép hạ xuống 2 gram
    "kw_min_precision": 0.97,   # keyword ứng viên phải RẤT sạch trên train (>=97%).
                                # Cho phép chút nhiễu để rule có FP -> AND/NOT có
                                # việc dọn; greedy vẫn ép precision TOÀN CỤC >=99%.
    "kw_min_spam_docs": 3,      # keyword phải phủ >=3 doc spam (tránh nhiễu lẻ)
    "target_precision": 0.99,   # ràng buộc precision toàn cục > 99%
    "target_recall": 0.30,      # ràng buộc recall > 30%
    "max_rules": 20,            # không quá 20 rule
    "max_kw_per_rule": 30,      # mỗi rule không quá 20 keyword

    # non-spam (chỉ để xác nhận, KHÔNG override spam)
    "nonspam_min_gram": 2,          # cụm xác nhận non-spam thường NGẮN (2-3 từ)
    "nonspam_min_gram_fallback": 2, # nếu quá ít ứng viên -> vẫn giữ mức 2
    "nonspam_min_precision": 0.98,  # rất sạch nghiêng nonspam (cụm df_spam=0 -> =1.0)
    "nonspam_min_docs": 4,          # phủ >=4 doc nonspam mới xét (tránh nhiễu lẻ)
    "max_nonspam_rules": 6,
    "max_nonspam_kw": 20,
}

EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF\U00002B00-\U00002BFF️‍]",
    flags=re.UNICODE,
)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MULTISPACE_RE = re.compile(r"\s+")


def norm_match(text):
    """Chuẩn hoá văn bản dùng để MATCH keyword (giữ dấu tiếng Việt).

    lower -> NFC -> bỏ url/emoji -> gộp mọi khoảng trắng thành 1 space.
    Keyword và nội dung đều đi qua hàm này nên phép so khớp là nhất quán.
    """
    if text is None:
        return ""
    t = unicodedata.normalize("NFC", str(text)).lower()
    t = URL_RE.sub(" ", t)
    t = EMOJI_RE.sub(" ", t)
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t


# ----------------------------------------------------------------------------
# MATCHING NHẤT QUÁN
# Keyword và document đều được biểu diễn bằng CHUỖI TOKEN (đơn vị "cụm từ có
# nghĩa" của PhoBERT/bộ tách từ; âm tiết trong 1 từ nối bằng '_').
# Một keyword match nếu dãy token của nó là DÃY CON LIÊN TIẾP trong token của doc.
# Đây CHÍNH LÀ cách n-gram được tính ở step2/step3 -> precision đo được là thật.
# ----------------------------------------------------------------------------
WORD_RE = re.compile(r"(?u)\w+")


def match_tokens(segmented_text):
    """Tokenize GIỐNG HỆT CountVectorizer(token_pattern=r'\\b\\w+\\b') dùng ở step2/3:
    giữ '_' (nằm trong \\w), bỏ dấu câu. Đây là list token dùng để MATCH."""
    return WORD_RE.findall(segmented_text or "")


def kw_tokens(kw):
    """Tách keyword (match-form, có thể chứa '_') thành list token."""
    if kw is None:
        return []
    return str(kw).strip().split()


def kw_display(kw):
    """Dạng hiển thị cho người đọc: '_' -> space, gộp khoảng trắng thừa."""
    return MULTISPACE_RE.sub(" ", str(kw).replace("_", " ")).strip()


def phrase_in_tokens(kwt, doc_tokens):
    """kwt (list token của keyword) có là dãy con liên tiếp của doc_tokens không."""
    n, m = len(kwt), len(doc_tokens)
    if n == 0 or n > m:
        return False
    first = kwt[0]
    for i in range(m - n + 1):
        if doc_tokens[i] == first and doc_tokens[i:i + n] == kwt:
            return True
    return False


def gram_count(kw):
    """Số 'gram' (số từ) của 1 keyword."""
    return len(kw_tokens(kw))


# ----------------------------------------------------------------------------
# STOPWORD tiếng Việt (từ chức năng / hư từ) — dùng để loại keyword VÔ NGHĨA.
# Một "token" sau tách từ có thể là từ ghép nối bằng '_' (vd 'các_bạn').
# Ta coi 1 token là stopword nếu MỌI âm tiết của nó đều là hư từ.
# ----------------------------------------------------------------------------
STOP_SYLLABLES = {
    # liên từ / giới từ / trợ từ
    "và", "là", "của", "có", "các", "những", "được", "cho", "với", "này",
    "đó", "kia", "ấy", "khi", "đã", "sẽ", "đang", "thì", "mà", "nếu", "vì",
    "nên", "cũng", "rất", "quá", "lắm", "để", "từ", "về", "trong", "ngoài",
    "trên", "dưới", "ra", "vào", "lên", "xuống", "bị", "bởi", "tại", "theo",
    "như", "hay", "hoặc", "còn", "chỉ", "đến", "tới", "sau", "trước", "bằng",
    "vẫn", "lại", "nữa", "ở", "hơn", "thế", "nào", "gì", "ai", "sao", "đâu",
    "một", "hai", "ba", "mấy", "bao", "nhiêu", "vài", "mọi", "từng", "mỗi",
    # đại từ / xưng hô / cảm thán
    "bạn", "mình", "tôi", "tớ", "em", "anh", "chị", "cô", "chú", "bác", "họ",
    "nó", "ta", "chúng", "ạ", "nhé", "nha", "nhá", "nghen", "ơi", "à", "ừ",
    "vâng", "dạ", "thôi", "đây", "đi", "rồi", "luôn", "ngay", "chứ", "mới",
    "cứ", "hết", "cả", "được", "làm", "thấy", "biết", "muốn", "cần", "nhau",
}


# Chữ cái HỢP LỆ: latin cơ bản a-z + toàn bộ nguyên âm có dấu tiếng Việt + đ.
# Chữ nằm ngoài tập này (vd 'ℎ' U+210E, 'ɪ ᴄ ʏ' small-caps, ký tự toán học 𝐡)
# => 'rác font' -> keyword chứa nó bị loại (vô nghĩa).
_LATIN = set("abcdefghijklmnopqrstuvwxyz")
_VIET = set("áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ")
_GOOD_LETTERS = _LATIN | _VIET


def _has_garbled_char(tok):
    """True nếu token chứa chữ cái ngoài bảng latin/tiếng Việt chuẩn, hoặc chữ SỐ
    'font lạ' (vd 𝟎𝟗 bold, ① khoanh tròn) — dấu hiệu văn bản rác/vô nghĩa."""
    for ch in tok:
        if ch.isalpha() and ch.lower() not in _GOOD_LETTERS:
            return True
        if ch.isdigit() and ch not in "0123456789":
            return True
    return False


def _is_word_token(tok):
    """Token là 'từ thực' khi có >=1 chữ cái hợp lệ và độ dài >=2 (không phải
    số thuần/ký tự lẻ)."""
    syls = "".join(_syllables(tok))
    if len(syls) < 2:
        return False
    return any(ch.lower() in _GOOD_LETTERS for ch in syls)


def _syllables(tok):
    return str(tok).split("_")


def is_stopword_token(tok):
    """Token là hư từ nếu MỌI âm tiết của nó nằm trong STOP_SYLLABLES."""
    syls = _syllables(tok)
    return bool(syls) and all(s in STOP_SYLLABLES for s in syls)


def is_meaningful_kw(kw, common_tokens=None, min_content=1):
    """Keyword 'có nghĩa' khi:
      - có >= min_content token KHÔNG phải hư từ VÀ không phải 'token phổ biến'
        (content-anchor: mang thông tin phân biệt spam), VÀ
      - token ĐẦU và token CUỐI không phải hư từ (tránh mảnh vụn kiểu
        'khám bệnh tại', 'để đặt hàng').
    common_tokens: tập token đời thường (df cao) — coi như không phải nội dung.
    """
    toks = kw_tokens(kw)
    if not toks:
        return False
    common_tokens = common_tokens or set()
    # loại 'rác font' (chữ toán học/small-caps/letterlike, số font lạ) -> vô nghĩa
    if any(_has_garbled_char(t) for t in toks):
        return False
    if is_stopword_token(toks[0]) or is_stopword_token(toks[-1]):
        return False
    # phải có >=min_content 'từ thực' mang nội dung (không phải số/ký tự thuần,
    # không hư từ, không token đời thường) -> loại 'khối số' kiểu SĐT/giá.
    content = [t for t in toks
               if _is_word_token(t)
               and not is_stopword_token(t)
               and t not in common_tokens]
    return len(content) >= min_content


def contains_phrase(outer_kw, inner_kw):
    """True nếu inner_kw là DÃY CON LIÊN TIẾP của outer_kw (outer 'chứa' inner)."""
    return phrase_in_tokens(kw_tokens(inner_kw), kw_tokens(outer_kw))


def prune_containment(keywords, keep="shorter"):
    """Khử các keyword 'chứa nhau'.

    keep='shorter': giữ cụm NGẮN/tổng quát, bỏ cụm DÀI chứa nó (cụm ngắn đã
    match mọi doc mà cụm dài match và nhiều hơn -> cụm dài dư thừa cho OR).
    Thứ tự ưu tiên đầu vào được tôn trọng khi độ dài bằng nhau (giữ cái đứng trước).
    Trả về danh sách keyword đã lọc (giữ nguyên format gốc).
    """
    # sắp: ngắn trước (để cụm tổng quát được giữ trước), giữ thứ tự gốc khi bằng
    order = sorted(range(len(keywords)), key=lambda i: (gram_count(keywords[i]), i))
    kept = []
    for i in order:
        kw = keywords[i]
        redundant = False
        for k in kept:
            # nếu 1 keyword đã giữ là dãy con của kw -> kw dư thừa
            if gram_count(k) <= gram_count(kw) and contains_phrase(kw, k):
                redundant = True
                break
        if not redundant:
            kept.append(kw)
    # trả về theo THỨ TỰ ƯU TIÊN GỐC (không theo thứ tự sắp tạm)
    kept_set = set(kept)
    return [k for k in keywords if k in kept_set]


# ----------------------------------------------------------------------------
# EMBEDDING NGỮ NGHĨA (PhoBERT) — để gom keyword "chung ý nghĩa" ở step4.
# Lazy-load; nếu thiếu transformers/torch -> trả None (step4 fallback lexical).
# ----------------------------------------------------------------------------
_EMBEDDER_CACHE = {}


def get_phobert_embedder(model_name="vinai/phobert-base"):
    """Trả về hàm embed(list_text)->np.ndarray (mean-pooled), hoặc None nếu lỗi.

    Đầu vào NÊN là văn bản đã tách từ (âm tiết nối bằng '_') — đúng định dạng
    PhoBERT. Keyword match-form (giữ '_') dùng trực tiếp được.
    """
    if "fn" in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE["fn"]
    fn = None
    try:
        import numpy as _np
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        def embed(texts, batch_size=64):
            vecs = []
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch = [t if t else "." for t in texts[i:i + batch_size]]
                    enc = tok(batch, padding=True, truncation=True,
                              max_length=64, return_tensors="pt")
                    out = model(**enc).last_hidden_state         # (B,T,H)
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    summed = (out * mask).sum(1)
                    cnt = mask.sum(1).clamp(min=1e-9)
                    mean = (summed / cnt).cpu().numpy()
                    vecs.append(mean)
            arr = _np.vstack(vecs).astype(_np.float32)
            norm = _np.linalg.norm(arr, axis=1, keepdims=True)
            return arr / _np.maximum(norm, 1e-9)                 # đã chuẩn hoá cosine

        fn = embed
    except Exception as e:
        print(f"[common] Không nạp được PhoBERT embedder ({e}); dùng lexical",
              file=sys.stderr)
        fn = None
    _EMBEDDER_CACHE["fn"] = fn
    return fn


def make_output_dir(input_path, parent=None):
    """Tạo 1 folder chứa toàn bộ output, tên = tên file input (không phần mở rộng).

    Nếu folder đã tồn tại -> thêm đuôi _v1, _v2, _v3, ... cho tới khi chưa tồn tại.
    parent: nơi đặt folder (mặc định = thư mục chứa file input).
    Trả về đường dẫn tuyệt đối của folder vừa tạo.
    """
    input_path = os.path.abspath(input_path)
    base = os.path.splitext(os.path.basename(input_path))[0]
    parent = os.path.abspath(parent) if parent else os.path.dirname(input_path)

    candidate = os.path.join(parent, base)
    n = 1
    while os.path.exists(candidate):
        candidate = os.path.join(parent, f"{base}_v{n}")
        n += 1
    os.makedirs(candidate)
    return candidate


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


# ----------------------------------------------------------------------------
# Đánh giá 1 rule / bộ rule trên tập document đã chuẩn hoá
# ----------------------------------------------------------------------------
def rule_matches(rule, doc_tokens):
    """rule = {'or':[...], 'and':[...], 'not':[...]} — trả về True nếu match.

    match = (OR rỗng hoặc có >=1 kw OR) AND (mọi kw AND đều có) AND (không kw NOT nào có).
    doc_tokens: list token của document (đơn vị cụm từ có nghĩa).
    """
    ors = rule.get("or") or []
    ands = rule.get("and") or []
    nots = rule.get("not") or []

    if ors:
        if not any(phrase_in_tokens(kw_tokens(k), doc_tokens) for k in ors):
            return False
    if ands:
        if not all(phrase_in_tokens(kw_tokens(k), doc_tokens) for k in ands):
            return False
    for k in nots:
        if phrase_in_tokens(kw_tokens(k), doc_tokens):
            return False
    return True


def classify(spam_rules, nonspam_rules, doc_tokens):
    """Trả về (predicted_label, detail, matched_spam_ids, matched_nonspam_ids).

    Logic conflict (khớp format):
      P1: có rule spam match         -> 'spam'          (definitive)
      P2: chỉ non-spam match         -> 'nonspam'       (confirmed)
      Conflict: cả hai match         -> 'uncertain_spam'(coi như SPAM)
      Không match gì                 -> 'uncertain'     (coi như nonspam, review tay)
    """
    matched_spam = [r["id"] for r in spam_rules if rule_matches(r, doc_tokens)]
    matched_non = [r["id"] for r in nonspam_rules if rule_matches(r, doc_tokens)]

    if matched_spam and matched_non:
        return "spam", "uncertain_spam", matched_spam, matched_non
    if matched_spam:
        return "spam", "spam", matched_spam, matched_non
    if matched_non:
        return "nonspam", "confirmed_nonspam", matched_spam, matched_non
    return "nonspam", "uncertain", matched_spam, matched_non


def prf(tp, fp, fn):
    """precision, recall, f1 an toàn với chia 0."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f
