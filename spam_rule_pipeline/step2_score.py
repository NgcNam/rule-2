# -*- coding: utf-8 -*-
"""
STEP 2 — Dùng TF-IDF để đánh giá dữ liệu và tính spam_score cho từng cụm từ.

Ý tưởng:
  - Xây từ vựng n-gram (1..4) TRÊN TẬP TRAIN từ token đã tách ở step1.
  - Với mỗi cụm từ t, thống kê trên train:
        n_spam(t)   = số doc SPAM chứa t
        n_non(t)    = số doc NONSPAM chứa t
        precision(t)= n_spam / (n_spam + n_non)       -> độ "sạch"
        support(t)  = n_spam / N_spam_train           -> độ phủ (đóng góp recall)
        tfidf_spam  = TF-IDF trung bình của t trong các doc spam
  - spam_score(t) = precision(t)^alpha * log1p(n_spam) * (0.5 + tfidf_spam)
        (alpha lớn => cực kỳ ưu tiên precision, đúng tinh thần "thà bỏ sót").

INPUT : step1_segmented.json
OUTPUT: step2_scores.json (danh sách cụm từ + thống kê, sort theo spam_score)

Chạy:
  python step2_score.py --input ../out/step1_segmented.json --output ../out/step2_scores.json
"""
import argparse
import math

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from common import DEFAULTS, read_json, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--ngram-min", type=int, default=DEFAULTS["ngram_min"])
    ap.add_argument("--ngram-max", type=int, default=DEFAULTS["ngram_max"])
    ap.add_argument("--min-df", type=int, default=DEFAULTS["min_df"])
    ap.add_argument("--alpha", type=float, default=DEFAULTS["spam_score_alpha"])
    ap.add_argument("--max-terms", type=int, default=20000,
                    help="số cụm từ SPAM tối đa xuất ra (đã sort theo spam_score)")
    ap.add_argument("--max-nonspam-terms", type=int, default=8000,
                    help="số cụm từ XÁC NHẬN NON-SPAM tối đa xuất ra (sort theo df_non)")
    args = ap.parse_args()

    data = read_json(args.input)
    docs = data["documents"]
    train = [d for d in docs if d["split"] == "train"]

    spam_lbl = DEFAULTS["spam_label"]
    texts = [d["segmented_text"] for d in train]
    y = np.array([1 if d["label"] == spam_lbl else 0 for d in train], dtype=np.int64)
    n_spam = int(y.sum())
    n_non = int((1 - y).sum())

    # đếm nhị phân (doc-frequency) để tính precision/support theo doc
    cv = CountVectorizer(
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
        lowercase=False,        # step1 đã lower
        binary=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    Xbin = cv.fit_transform(texts)          # (n_docs, n_terms) 0/1
    vocab = cv.get_feature_names_out()

    # df theo lớp: nhân ma trận với vector nhãn (densify kết quả sparse)
    def dense1d(m):
        return np.asarray(m.todense()).ravel() if hasattr(m, "todense") else np.asarray(m).ravel()

    yb = csr_matrix(y.reshape(1, -1))       # (1, n_docs)
    df_spam = dense1d(yb.dot(Xbin))          # (n_terms,)
    df_all = dense1d(Xbin.sum(axis=0))
    df_non = df_all - df_spam

    # TF-IDF trung bình trong doc spam (dùng đánh giá độ quan trọng của cụm từ)
    cnt = CountVectorizer(
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df, lowercase=False, token_pattern=r"(?u)\b\w+\b",
        vocabulary=cv.vocabulary_,
    )
    Xc = cnt.fit_transform(texts)
    tfidf = TfidfTransformer().fit_transform(Xc)        # (n_docs, n_terms)
    spam_rows = tfidf[y == 1]
    # trung bình theo doc spam có chứa cụm từ đó
    sum_tfidf_spam = dense1d(spam_rows.sum(axis=0))
    tfidf_spam_mean = np.divide(
        sum_tfidf_spam, np.maximum(df_spam, 1),
        out=np.zeros_like(sum_tfidf_spam), where=df_spam > 0,
    )

    # df của TỪNG token đơn (unigram) trên toàn train -> phát hiện "từ phổ biến"
    cv1 = CountVectorizer(ngram_range=(1, 1), min_df=args.min_df, lowercase=False,
                          binary=True, token_pattern=r"(?u)\b\w+\b")
    X1 = cv1.fit_transform(texts)
    uni_df = dense1d(X1.sum(axis=0))
    unigram_df = {tok: int(uni_df[j]) for tok, j in cv1.vocabulary_.items()}

    precision = np.divide(df_spam, np.maximum(df_all, 1))
    precision_non = np.divide(df_non, np.maximum(df_all, 1))
    support = df_spam / max(n_spam, 1)
    alpha = args.alpha
    spam_score = (precision ** alpha) * np.log1p(df_spam) * (0.5 + tfidf_spam_mean)

    def build_term(j):
        term = vocab[j]
        return {
            "term": term,
            "gram": len(term.split()),
            "df_spam": int(df_spam[j]),
            "df_non": int(df_non[j]),
            "precision": round(float(precision[j]), 6),
            "precision_nonspam": round(float(precision_non[j]), 6),
            "support": round(float(support[j]), 6),
            "tfidf_spam_mean": round(float(tfidf_spam_mean[j]), 6),
            "spam_score": round(float(spam_score[j]), 6),
        }

    # (1) cụm từ LIÊN QUAN SPAM: xuất hiện ở >=1 doc spam -> sort theo spam_score
    spam_idx = [j for j in range(len(vocab)) if df_spam[j] > 0]
    spam_idx.sort(key=lambda j: spam_score[j], reverse=True)
    if args.max_terms:
        spam_idx = spam_idx[: args.max_terms]

    # (2) cụm từ XÁC NHẬN NON-SPAM: KHÔNG có trong doc spam (df_spam=0) nhưng phổ
    #     biến ở nonspam. TRƯỚC ĐÂY bị loại hết bởi 'df_spam<=0 -> continue' khiến
    #     step3 không có nguyên liệu -> nonspam_cand=0. Giữ lại để sinh rule non-spam.
    nonspam_min_docs = DEFAULTS["nonspam_min_docs"]
    non_idx = [j for j in range(len(vocab))
               if df_spam[j] == 0 and df_non[j] >= nonspam_min_docs]
    non_idx.sort(key=lambda j: df_non[j], reverse=True)
    if args.max_nonspam_terms:
        non_idx = non_idx[: args.max_nonspam_terms]

    terms = [build_term(j) for j in spam_idx] + [build_term(j) for j in non_idx]
    n_spam_terms, n_nonspam_terms = len(spam_idx), len(non_idx)

    out = {
        "meta": {
            "step": 2,
            "from": args.input,
            "n_train_docs": len(train),
            "n_train_spam": n_spam,
            "n_train_nonspam": n_non,
            "ngram_range": [args.ngram_min, args.ngram_max],
            "min_df": args.min_df,
            "alpha": alpha,
            "n_terms_kept": len(terms),
            "n_spam_terms": n_spam_terms,
            "n_nonspam_terms": n_nonspam_terms,
            "spam_score_formula": "precision^alpha * log1p(df_spam) * (0.5 + tfidf_spam_mean)",
        },
        "terms": terms,
        "unigram_df": unigram_df,          # token -> số doc train chứa token (lọc từ phổ biến)
    }
    write_json(args.output, out)
    print(f"[step2] OK -> {args.output} | terms={len(terms)} "
          f"(spam={n_spam_terms} nonspam={n_nonspam_terms}, vocab={len(vocab)}) "
          f"train_spam={n_spam} train_non={n_non}")


if __name__ == "__main__":
    main()
