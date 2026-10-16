# -*- coding: utf-8 -*-
"""
STEP 3 — Sinh KEYWORD ỨNG VIÊN (bộ nguyên liệu để ghép rule ở step4).

Từ bảng spam_score (step2), lọc ra các cụm từ đủ điều kiện làm keyword:
  - precision(train) >= kw_min_precision  (rất sạch -> rule spam "chắc chắn")
  - df_spam         >= kw_min_spam_docs   (phủ đủ nhiều -> không phải nhiễu lẻ)
  - gram            >= min_gram           (số gram không được ít quá)
Đồng thời tính TẬP DOC (train) mà mỗi keyword match — phục vụ set-cover ở step4.

Cũng sinh keyword ỨNG VIÊN NON-SPAM (precision nghiêng nonspam) — chỉ để
xác nhận nội dung an toàn, KHÔNG bao giờ override spam.

INPUT : step1_segmented.json, step2_scores.json
OUTPUT: step3_candidates.json

Chạy:
  python step3_candidates.py --step1 ../out/step1_segmented.json \
      --step2 ../out/step2_scores.json --output ../out/step3_candidates.json
"""
import argparse

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from common import (DEFAULTS, is_meaningful_kw, prune_containment, read_json,
                    write_json)


def membership(terms, texts):
    """Trả về dict term -> np.array chỉ số doc (theo thứ tự texts) mà term match."""
    if not terms:
        return {}
    vocab = {t: i for i, t in enumerate(terms)}
    cv = CountVectorizer(vocabulary=vocab, ngram_range=(1, 8),
                         lowercase=False, binary=True, token_pattern=r"(?u)\b\w+\b")
    X = cv.fit_transform(texts).tocsc()
    out = {}
    for t, j in vocab.items():
        out[t] = X.getcol(j).nonzero()[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1", required=True)
    ap.add_argument("--step2", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-gram", type=int, default=DEFAULTS["min_gram"])
    ap.add_argument("--nonspam-min-gram", type=int, default=DEFAULTS["nonspam_min_gram"],
                    help="số gram tối thiểu cho cụm xác nhận NON-SPAM (mặc định thấp hơn spam)")
    ap.add_argument("--kw-min-precision", type=float, default=DEFAULTS["kw_min_precision"])
    ap.add_argument("--kw-min-spam-docs", type=int, default=DEFAULTS["kw_min_spam_docs"])
    ap.add_argument("--max-candidates", type=int, default=3000)
    ap.add_argument("--stopword-df-ratio", type=float, default=0.03,
                    help="token xuất hiện trong > tỉ lệ này của doc train bị coi là 'phổ biến'; "
                         "mỗi keyword phải có >=1 token KHÔNG phổ biến (content-anchor)")
    args = ap.parse_args()

    s1 = read_json(args.step1)
    s2 = read_json(args.step2)
    docs = s1["documents"]
    train = [d for d in docs if d["split"] == "train"]
    spam_lbl = DEFAULTS["spam_label"]

    texts = [d["segmented_text"] for d in train]
    is_spam = np.array([d["label"] == spam_lbl for d in train])
    train_spam_idx = set(np.where(is_spam)[0].tolist())

    # nhóm "token phổ biến" (đời thường) -> cụm từ chỉ gồm các token này thường
    # là từ nối vô nghĩa về spam (vd 'các bạn đã', 'cách tốt nhất') -> loại.
    n_train_docs = s2["meta"]["n_train_docs"]
    unigram_df = s2.get("unigram_df", {})
    common_tokens = {tok for tok, df in unigram_df.items()
                     if df / max(n_train_docs, 1) > args.stopword_df_ratio}
    

    # ---- lọc ứng viên SPAM ----
    # 'có nghĩa' = có content-anchor (token không hư từ & không phổ biến) VÀ
    # không mở/đóng bằng hư từ (tránh mảnh vụn kiểu 'khám bệnh tại').
    def qualify(t, min_gram):
        return (t["gram"] >= min_gram
                and t["df_spam"] >= args.kw_min_spam_docs
                and t["precision"] >= args.kw_min_precision
                and is_meaningful_kw(t["term"], common_tokens))

    cand_terms = [t for t in s2["terms"] if qualify(t, args.min_gram)]
    # nếu quá ít, hạ min_gram xuống fallback để đủ nguyên liệu recall
    used_min_gram = args.min_gram
    if len(cand_terms) < 30:
        used_min_gram = DEFAULTS["min_gram_fallback"]
        cand_terms = [t for t in s2["terms"] if qualify(t, used_min_gram)]

    # ưu tiên phủ nhiều spam trước (df_spam), rồi spam_score
    cand_terms.sort(key=lambda t: (t["df_spam"], t["spam_score"]), reverse=True)
    cand_terms = cand_terms[: args.max_candidates]

    # KHỬ CHỨA NHAU: bỏ cụm dài đã bao 1 cụm ngắn (cụm ngắn tổng quát hơn,
    # match nhiều doc hơn) -> nguyên liệu OR gọn, không dư thừa.
    by_term = {t["term"]: t for t in cand_terms}
    kept_terms = prune_containment([t["term"] for t in cand_terms], keep="shorter")
    cand_terms = [by_term[k] for k in kept_terms]

    spam_mem = membership([t["term"] for t in cand_terms], texts)
    spam_candidates = []
    for t in cand_terms:
        idx = spam_mem.get(t["term"], np.array([], dtype=int))
        cov_spam = sorted(int(i) for i in idx if i in train_spam_idx)
        cov_non = sorted(int(i) for i in idx if i not in train_spam_idx)
        spam_candidates.append({
            "keyword": t["term"],                     # dạng MATCH (giữ '_')
            "keyword_display": t["term"].replace("_", " "),
            "term_raw": t["term"],
            "gram": t["gram"],
            "precision": t["precision"],
            "spam_score": t["spam_score"],
            "df_spam": t["df_spam"],
            "df_non": t["df_non"],
            "train_spam_docs": cov_spam,   # chỉ số doc train (spam) match keyword
            "train_non_docs": cov_non,     # chỉ số doc train (nonspam) match keyword
        })

    # ---- lọc ứng viên NON-SPAM (chỉ xác nhận) ----
    # Dùng min_gram RIÊNG cho non-spam (thường ngắn hơn spam) + fallback nếu quá ít.
    n_non_train = int((~is_spam).sum())

    def qualify_non(t, min_gram):
        n_all = t["df_spam"] + t["df_non"]
        prec_non = t["df_non"] / n_all if n_all else 0.0
        ok = (t["gram"] >= min_gram
              and t["df_non"] >= DEFAULTS["nonspam_min_docs"]
              and prec_non >= DEFAULTS["nonspam_min_precision"]
              and is_meaningful_kw(t["term"], common_tokens))
        return ok, prec_non

    used_non_min_gram = args.nonspam_min_gram
    non_cands_raw = [(t, pn) for t in s2["terms"]
                     for ok, pn in [qualify_non(t, used_non_min_gram)] if ok]
    if len(non_cands_raw) < 30:      # quá ít -> hạ về mức fallback để đủ nguyên liệu
        used_non_min_gram = DEFAULTS["nonspam_min_gram_fallback"]
        non_cands_raw = [(t, pn) for t in s2["terms"]
                         for ok, pn in [qualify_non(t, used_non_min_gram)] if ok]
    non_cands_raw.sort(key=lambda x: (x[0]["df_non"], x[1]), reverse=True)
    non_cands_raw = non_cands_raw[:1000]
    non_mem = membership([t["term"] for t, _ in non_cands_raw], texts)
    nonspam_candidates = []
    for t, prec_non in non_cands_raw:
        idx = non_mem.get(t["term"], np.array([], dtype=int))
        cov_non = sorted(int(i) for i in idx if i not in train_spam_idx)
        cov_spam = sorted(int(i) for i in idx if i in train_spam_idx)
        nonspam_candidates.append({
            "keyword": t["term"],
            "keyword_display": t["term"].replace("_", " "),
            "term_raw": t["term"],
            "gram": t["gram"],
            "precision_nonspam": round(prec_non, 6),
            "df_non": t["df_non"],
            "df_spam": t["df_spam"],
            "train_non_docs": cov_non,
            "train_spam_docs": cov_spam,
        })

    out = {
        "meta": {
            "step": 3,
            "n_train": len(train),
            "n_train_spam": len(train_spam_idx),
            "n_train_nonspam": n_non_train,
            "min_gram_used": used_min_gram,
            "nonspam_min_gram_used": used_non_min_gram,
            "kw_min_precision": args.kw_min_precision,
            "kw_min_spam_docs": args.kw_min_spam_docs,
            "n_spam_candidates": len(spam_candidates),
            "n_nonspam_candidates": len(nonspam_candidates),
            "stopword_df_ratio": args.stopword_df_ratio,
        },
        "common_tokens": sorted(common_tokens),   # token đời thường (cho step4)
        "spam_candidates": spam_candidates,
        "nonspam_candidates": nonspam_candidates,
    }
    write_json(args.output, out)
    print(f"[step3] OK -> {args.output} | spam_cand={len(spam_candidates)} "
          f"nonspam_cand={len(nonspam_candidates)} min_gram={used_min_gram} "
          f"nonspam_min_gram={used_non_min_gram}")


if __name__ == "__main__":
    main()
