# -*- coding: utf-8 -*-
"""
STEP 4 — Chọn keyword & ghép thành RULEBASE bảo đảm precision > 99%, recall > 30%.

Thuật toán:
  1) GREEDY MAX-COVERAGE có ràng buộc precision:
       lặp chọn keyword đem lại NHIỀU doc spam MỚI nhất, với điều kiện thêm nó
       vào KHÔNG làm precision train tụt xuống dưới target (0.99).
       => keyword "sạch" (0 doc nonspam) luôn được ưu tiên -> precision ~1.0.
       "Thà bỏ sót còn hơn xoá nhầm": không bao giờ hi sinh precision lấy recall.
  2) KHỬ CHỨA NHAU toàn cục: bỏ cụm dài đã bao 1 cụm ngắn (giữ cụm tổng quát).
  3) GOM NHÓM THEO NGỮ NGHĨA: nhúng keyword bằng PhoBERT rồi cluster theo cosine
     -> mỗi rule chứa các keyword CHUNG Ý NGHĨA (không phải chỉ đồng-xuất-hiện).
     Thiếu PhoBERT -> fallback gom theo token nội dung chia sẻ (lexical).
  4) Sinh AND & NOT theo hướng AN TOÀN RECALL (kiểm nghiệm trên train):
       - NOT: cụm từ CHỈ có trong doc nonspam mà rule dính (0 doc spam) -> loại FP,
              không bao giờ cắt doc spam.
       - AND: token/cụm nội dung xuất hiện ở >=95% doc spam của rule NHƯNG vắng ở
              >=1 doc nonspam bẩn -> siết precision mà gần như không mất recall.
  5) Chọn rule NON-SPAM (chỉ xác nhận, không override spam).

INPUT : step1_segmented.json, step3_candidates.json
OUTPUT: step4_rules.json

Chạy:
  python step4_select_rules.py --step1 ../out/step1_segmented.json \
      --step3 ../out/step3_candidates.json --output ../out/step4_rules.json
"""
import argparse
from collections import Counter

from common import (DEFAULTS, get_phobert_embedder, is_meaningful_kw,
                    kw_tokens, match_tokens, prune_containment, read_json,
                    write_json)


def greedy_cover(cands, n_spam_train, target_prec, target_recall, max_total_kw):
    """Chọn keyword theo max-coverage có ràng buộc precision. Trả về list keyword đã chọn."""
    for c in cands:
        c["_spam"] = set(c["train_spam_docs"])
        c["_non"] = set(c["train_non_docs"])

    covered_spam, covered_non = set(), set()
    selected = []
    pool = [c for c in cands if c["_spam"]]

    while len(selected) < max_total_kw:
        best, best_gain = None, 0
        for c in pool:
            if c.get("_used"):
                continue
            gain = len(c["_spam"] - covered_spam)
            if gain <= best_gain:
                continue
            new_tp = len(covered_spam | c["_spam"])
            new_fp = len(covered_non | c["_non"])
            prec = new_tp / (new_tp + new_fp) if (new_tp + new_fp) else 1.0
            if prec < target_prec:
                continue
            best, best_gain = c, gain
        if best is None:
            break
        best["_used"] = True
        covered_spam |= best["_spam"]
        covered_non |= best["_non"]
        selected.append(best)

    recall = len(covered_spam) / max(n_spam_train, 1)
    precision = (len(covered_spam) / (len(covered_spam) + len(covered_non))
                 if (covered_spam or covered_non) else 1.0)
    return selected, covered_spam, covered_non, precision, recall


def dedupe_containment(selected):
    """Bỏ keyword dài đã 'chứa' 1 keyword ngắn khác trong tập đã chọn (giữ cụm ngắn).

    Cụm ngắn match superset -> gộp tập _spam/_non của cụm dài vào cụm ngắn để
    không mất độ phủ khi loại (thực chất cụm ngắn vốn đã phủ >= cụm dài)."""
    by_kw = {c["keyword"]: c for c in selected}
    kept_kw = prune_containment([c["keyword"] for c in selected], keep="shorter")
    return [by_kw[k] for k in kept_kw]


# ---------------------------------------------------------------- semantic ---
def _content_tokens(kw, common_tokens):
    """Các âm tiết mang nội dung của keyword (bỏ hư từ & token phổ biến)."""
    from common import is_stopword_token
    out = []
    for t in kw_tokens(kw):
        if is_stopword_token(t) or t in common_tokens:
            continue
        out.extend(t.split("_"))
    return set(out)


def cluster_semantic(selected, common_tokens, embedder, sim_threshold=0.62):
    """Gán mỗi keyword vào 1 cụm NGỮ NGHĨA. Trả về list nhãn cluster (int)."""
    kws = [c["keyword_display"] for c in selected]
    emb_input = [c["keyword"] for c in selected]   # match-form '_' = input PhoBERT
    n = len(selected)
    if n == 0:
        return []

    if embedder is not None:
        try:
            import numpy as np
            from sklearn.cluster import AgglomerativeClustering
            X = embedder(emb_input)                        # (n,H) đã chuẩn hoá
            if n == 1:
                return [0]
            model = AgglomerativeClustering(
                n_clusters=None, metric="cosine", linkage="average",
                distance_threshold=1.0 - sim_threshold)
            labels = model.fit_predict(X)
            return labels.tolist()
        except Exception as e:
            print(f"[step4] cluster embedding lỗi ({e}); fallback lexical")

    # ---- fallback LEXICAL: gom theo token nội dung chia sẻ (Jaccard) ----
    feats = [_content_tokens(k, common_tokens) for k in emb_input]
    labels = [-1] * n
    centroids = []          # list[set] token nội dung của từng cụm
    order = sorted(range(n), key=lambda i: selected[i]["df_spam"], reverse=True)
    for i in order:
        best, best_sim = -1, 0.0
        for ci, cen in enumerate(centroids):
            inter = len(feats[i] & cen)
            if not inter:
                continue
            sim = inter / len(feats[i] | cen)
            if sim > best_sim:
                best, best_sim = ci, sim
        if best_sim >= 0.34 and best >= 0:
            labels[i] = best
            centroids[best] |= feats[i]
        else:
            labels[i] = len(centroids)
            centroids.append(set(feats[i]))
    _ = kws
    return labels


def pack_clusters(selected, labels, max_rules, max_kw):
    """Từ nhãn cluster -> danh sách 'chunk' (mỗi chunk = 1 rule), tôn trọng
    <=max_rules rule và mỗi rule <=max_kw keyword. Tách cụm quá lớn, dồn cụm nhỏ."""
    groups = {}
    for c, lb in zip(selected, labels):
        groups.setdefault(lb, []).append(c)

    chunks = []
    for lb, kws in groups.items():
        kws = sorted(kws, key=lambda c: c["df_spam"], reverse=True)
        for i in range(0, len(kws), max_kw):
            chunks.append(kws[i:i + max_kw])

    chunks.sort(key=lambda ch: sum(c["df_spam"] for c in ch), reverse=True)
    # nếu quá nhiều chunk: dồn chunk nhỏ nhất vào chunk còn chỗ (ưu tiên cùng nghĩa
    # bằng cách nhét vào chunk nhỏ nhất còn chỗ -> giữ chunk lớn tinh khiết)
    while len(chunks) > max_rules:
        small = chunks.pop()                       # chunk ít phủ nhất
        target = None
        for ch in sorted(chunks, key=len):
            if len(ch) + len(small) <= max_kw:
                target = ch
                break
        if target is None:
            for ch in chunks:
                room = max_kw - len(ch)
                if room > 0:
                    ch.extend(small[:room])
                    small = small[room:]
                if not small:
                    break
        else:
            target.extend(small)
        chunks.sort(key=lambda ch: sum(c["df_spam"] for c in ch), reverse=True)
    return chunks


# ------------------------------------------------------------ AND / NOT ---
def _phrases_in_docs(doc_ids, all_tokens, grams=(2, 3)):
    """Đếm số doc (trong doc_ids) chứa mỗi cụm n-gram. Trả Counter(phrase->#doc)."""
    cnt = Counter()
    for di in doc_ids:
        toks = all_tokens.get(di, [])
        seen = set()
        for n in grams:
            for j in range(len(toks) - n + 1):
                ph = " ".join(toks[j:j + n])
                if ph not in seen:
                    seen.add(ph)
                    cnt[ph] += 1
    return cnt


def _unigrams_in_docs(doc_ids, all_tokens):
    cnt = Counter()
    for di in doc_ids:
        for t in set(all_tokens.get(di, [])):
            cnt[t] += 1
    return cnt


def gen_not(rule_kws, all_tokens, common_tokens, max_not=4):
    """NOT an toàn recall: cụm CHỈ có trong doc nonspam bẩn, VẮNG ở mọi doc spam
    của rule -> thêm NOT loại FP mà không bao giờ cắt doc spam."""
    spam_docs, dirty = set(), set()
    for c in rule_kws:
        spam_docs |= c["_spam"]
        dirty |= c["_non"]
    if not dirty:
        return []
    # tập cụm 2-3 gram xuất hiện trong BẤT KỲ doc spam nào của rule -> cấm làm NOT
    spam_phrases = set(_phrases_in_docs(spam_docs, all_tokens).keys())
    cand = _phrases_in_docs(dirty, all_tokens)
    scored = []
    for ph, dfreq in cand.items():
        if ph in spam_phrases:
            continue                       # dính doc spam -> sẽ cắt recall, loại
        if not is_meaningful_kw(ph, common_tokens):
            continue
        # NOT phải MỞ ĐẦU bằng token nội dung (không phổ biến) -> loại trừ 'sạch',
        # tránh mảnh vụn kiểu 'giá tin' (bắt đầu bằng token đời thường 'giá').
        first = ph.split()[0]
        if first in common_tokens:
            continue
        scored.append((ph, dfreq))
    scored.sort(key=lambda x: x[1], reverse=True)

    # greedy phủ các doc nonspam bẩn
    nots, remaining = [], set(dirty)
    ph_docs = {}
    for ph, _ in scored:
        ph_docs[ph] = {di for di in dirty
                       if _contains(all_tokens.get(di, []), ph.split())}
    for ph, _ in scored:
        if len(nots) >= max_not:
            break
        gain = ph_docs[ph] & remaining
        if not gain:
            continue
        nots.append(ph)
        remaining -= gain
    return prune_containment(nots, keep="shorter")


def _and_candidates(spam_docs, dirty, all_tokens, common_tokens, cover):
    """Cụm nội dung có ở >=cover doc spam NHƯNG vắng ở >=1 doc nonspam bẩn."""
    n_spam = len(spam_docs)
    uni_spam = _unigrams_in_docs(spam_docs, all_tokens)
    bi_spam = _phrases_in_docs(spam_docs, all_tokens, grams=(2,))
    cands = []
    for ph, sc in list(uni_spam.items()) + list(bi_spam.items()):
        if sc / n_spam < cover:
            continue
        if not is_meaningful_kw(ph, common_tokens):
            continue
        toks = ph.split()
        dirty_has = sum(1 for di in dirty if _contains(all_tokens.get(di, []), toks))
        removed = len(dirty) - dirty_has          # doc nonspam bị AND loại
        if removed <= 0:
            continue
        cands.append((ph, removed, sc))
    # ưu tiên loại nhiều FP, rồi phủ spam cao, rồi cụm dài (đặc trưng hơn)
    cands.sort(key=lambda x: (x[1], x[2], len(x[0].split())), reverse=True)
    return cands


def gen_and(rule_kws, all_tokens, common_tokens, max_and=2, cover=0.90, cover_floor=0.75):
    """AND an toàn recall: token/cụm nội dung có ở >=cover doc spam của rule NHƯNG
    vắng ở >=1 doc nonspam bẩn -> siết precision, mất <=(1-cover) recall mỗi AND.

    Nếu ở mức `cover` không tìm được AND nào (dữ liệu sạch/đa dạng), HẠ DẦN cover
    (bước 0.05, tới sàn `cover_floor`) để vẫn sinh được AND dọn FP khi rule có bẩn.
    Mỗi mức vẫn ép cụm AND tổng hợp giữ >= cover_used doc spam -> an toàn recall.
    """
    spam_docs, dirty = set(), set()
    for c in rule_kws:
        spam_docs |= c["_spam"]
        dirty |= c["_non"]
    if not dirty or not spam_docs:
        return []
    n_spam = len(spam_docs)

    cover_used = cover
    while cover_used >= cover_floor - 1e-9:
        cands = _and_candidates(spam_docs, dirty, all_tokens, common_tokens, cover_used)
        ands, matched = [], set(spam_docs)
        for ph, _removed, _sc in cands:
            toks = ph.split()
            new_matched = {di for di in matched if _contains(all_tokens.get(di, []), toks)}
            if len(new_matched) / n_spam < cover_used:   # tổng hợp vẫn giữ >=cover_used
                continue
            ands.append(ph)
            matched = new_matched
            if len(ands) >= max_and:
                break
        if ands:
            return prune_containment(ands, keep="shorter")   # bỏ AND chứa nhau
        cover_used = round(cover_used - 0.05, 2)
    return []


def _contains(doc_tokens, phrase_tokens):
    n, m = len(phrase_tokens), len(doc_tokens)
    if n == 0 or n > m:
        return False
    first = phrase_tokens[0]
    for i in range(m - n + 1):
        if doc_tokens[i] == first and doc_tokens[i:i + n] == phrase_tokens:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1", required=True)
    ap.add_argument("--step3", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--target-precision", type=float, default=DEFAULTS["target_precision"])
    ap.add_argument("--target-recall", type=float, default=DEFAULTS["target_recall"])
    ap.add_argument("--max-rules", type=int, default=DEFAULTS["max_rules"])
    ap.add_argument("--max-kw", type=int, default=DEFAULTS["max_kw_per_rule"])
    ap.add_argument("--max-and", type=int, default=2, help="số AND tối đa mỗi rule")
    ap.add_argument("--max-not", type=int, default=4, help="số NOT tối đa mỗi rule")
    ap.add_argument("--and-cover", type=float, default=0.85,
                    help="AND phải có mặt ở >= tỉ lệ này doc spam của rule "
                         "(cao=an toàn recall; hạ xuống sinh nhiều AND hơn nhưng mất recall). "
                         "Nếu không có AND nào, tự hạ dần tới sàn --and-cover-floor.")
    ap.add_argument("--and-cover-floor", type=float, default=0.75,
                    help="sàn thấp nhất khi tự hạ and-cover để vẫn sinh được AND")
    ap.add_argument("--sim-threshold", type=float, default=0.62,
                    help="ngưỡng cosine gom nhóm ngữ nghĩa (cao hơn -> nhóm chặt hơn)")
    ap.add_argument("--no-embedding", action="store_true",
                    help="bỏ PhoBERT, gom nhóm theo token nội dung (lexical)")
    args = ap.parse_args()

    s1 = read_json(args.step1)
    s3 = read_json(args.step3)
    n_spam_train = s3["meta"]["n_train_spam"]
    common_tokens = set(s3.get("common_tokens", []))

    max_total_kw = args.max_rules * args.max_kw
    selected, cov_spam, cov_non, prec, recall = greedy_cover(
        s3["spam_candidates"], n_spam_train,
        args.target_precision, args.target_recall, max_total_kw,
    )
    # KHỬ CHỨA NHAU toàn cục (giữ cụm ngắn/tổng quát)
    selected = dedupe_containment(selected)

    # tokens của TẤT CẢ doc train (cho AND/NOT). Chỉ số = thứ tự train (giống step3).
    train = [d for d in s1["documents"] if d["split"] == "train"]
    all_tokens = {i: match_tokens(train[i]["segmented_text"]) for i in range(len(train))}

    # gom nhóm ngữ nghĩa
    embedder = None if args.no_embedding else get_phobert_embedder()
    labels = cluster_semantic(selected, common_tokens, embedder, args.sim_threshold)
    chunks = pack_clusters(selected, labels, args.max_rules, args.max_kw)

    spam_rules = []
    for ci, kws in enumerate(chunks, 1):
        kws_sorted = sorted(kws, key=lambda c: c["df_spam"], reverse=True)
        or_kw = prune_containment([c["keyword"] for c in kws_sorted], keep="shorter")
        ands = gen_and(kws_sorted, all_tokens, common_tokens, args.max_and,
                       cover=args.and_cover, cover_floor=args.and_cover_floor)
        nots = gen_not(kws_sorted, all_tokens, common_tokens, args.max_not)
        top = kws_sorted[0].get("keyword_display", kws_sorted[0]["keyword"].replace("_", " "))
        spam_rules.append({
            "id": f"SPAM_{ci:03d}",
            "category": top,
            "description": f"Cụm từ spam CHUNG Ý NGHĨA quanh: “{top}”",
            "or": or_kw,
            "and": ands,
            "not": nots,
        })

    # ---- rule NON-SPAM (chỉ xác nhận) ----
    non_selected = []
    covered_non_docs = set()
    for c in sorted(s3["nonspam_candidates"], key=lambda x: x["df_non"], reverse=True):
        if c["train_spam_docs"]:      # có dính spam -> bỏ (phải cực sạch)
            continue
        new = set(c["train_non_docs"]) - covered_non_docs
        if len(new) < DEFAULTS["nonspam_min_docs"]:
            continue
        non_selected.append(c)
        covered_non_docs |= set(c["train_non_docs"])
        if len(non_selected) >= DEFAULTS["max_nonspam_rules"] * DEFAULTS["max_nonspam_kw"]:
            break


    nonspam_rules = []
    for ci, i in enumerate(range(0, len(non_selected), DEFAULTS["max_nonspam_kw"]), 1):
        if ci > DEFAULTS["max_nonspam_rules"]:
            break
        grp = non_selected[i:i + DEFAULTS["max_nonspam_kw"]]
        gdisp = grp[0].get("keyword_display", grp[0]["keyword"].replace("_", " "))
        or_kw = prune_containment([c["keyword"] for c in grp], keep="shorter")
        nonspam_rules.append({
            "id": f"NONSPAM_{ci:03d}",
            "category": gdisp,
            "description": f"Nội dung an toàn quanh: “{gdisp}”",
            "or": or_kw,
            "and": [],
            "not": [],
        })

    n_and = sum(len(r["and"]) for r in spam_rules)
    n_not = sum(len(r["not"]) for r in spam_rules)
    out = {
        "meta": {
            "step": 4,
            "n_train_spam": n_spam_train,
            "n_selected_keywords": len(selected),
            "clustering": "phobert_embedding" if embedder else "lexical_token",
            "train_precision_estimate": round(prec, 6),
            "train_recall_estimate": round(recall, 6),
            "target_precision": args.target_precision,
            "target_recall": args.target_recall,
            "n_spam_rules": len(spam_rules),
            "n_nonspam_rules": len(nonspam_rules),
            "n_and_total": n_and,
            "n_not_total": n_not,
            "max_rules": args.max_rules,
            "max_kw_per_rule": args.max_kw,
        },
        "spam_rules": spam_rules,
        "nonspam_rules": nonspam_rules,
    }
    write_json(args.output, out)
    print(f"[step4] OK -> {args.output} | kw={len(selected)} "
          f"spam_rules={len(spam_rules)} nonspam_rules={len(nonspam_rules)} "
          f"AND={n_and} NOT={n_not} clustering={out['meta']['clustering']} "
          f"train_prec={prec:.4f} train_recall={recall:.4f}")


if __name__ == "__main__":
    main()
