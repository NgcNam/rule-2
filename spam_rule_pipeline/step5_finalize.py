# -*- coding: utf-8 -*-
"""
STEP 5 — Kiểm tra & chốt RÀNG BUỘC + cấu hình cơ chế conflict.

Ép cứng các ràng buộc (loại bỏ vi phạm thay vì để lọt):
  - <= 20 rule spam, mỗi rule <= 20 keyword.
  - Mỗi keyword phải >= min_gram gram (số gram không được ít quá).
  - Rule spam là rule "chắc chắn": chỉ giữ keyword đủ sạch (đã lọc từ step3/4).
  - Rule non-spam KHÔNG override spam — chỉ để xác nhận.
Đính kèm cấu hình logic phân loại + trạng thái trung gian:
  spam-only -> spam | nonspam-only -> nonspam | cả hai -> uncertain_spam (SPAM)
  | không match -> uncertain (review tay, coi như nonspam).

INPUT : step4_rules.json
OUTPUT: step5_final_rules.json

Chạy:
  python step5_finalize.py --input ../out/step4_rules.json --output ../out/step5_final_rules.json
"""
import argparse

from common import (DEFAULTS, gram_count, prune_containment, read_json,
                    write_json)


def clean_rule(rule, min_gram, max_kw):
    """Loại keyword < min_gram, khử trùng lặp & KHỬ CHỨA NHAU, cắt còn <= max_kw."""
    def filt(lst, mg):
        seen, out = set(), []
        for k in lst or []:
            if gram_count(k) < mg:
                continue
            key = k.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(k)
        # khử chứa nhau ngay trong rule (giữ cụm ngắn/tổng quát)
        return prune_containment(out, keep="shorter")

    r = dict(rule)
    r["or"] = filt(rule.get("or"), min_gram)[:max_kw]
    # AND siết precision: cho phép >=1 gram (token nội dung đơn vẫn hợp lệ)
    r["and"] = filt(rule.get("and"), 1)
    # NOT loại trừ: cho phép >=2 gram, không tính vào giới hạn recall
    r["not"] = filt(rule.get("not"), 2)
    # đảm bảo tổng keyword (or+and+not) <= max_kw
    budget = max_kw - len(r["or"]) - len(r["and"])
    if budget < 0:
        r["and"] = r["and"][:max(0, max_kw - len(r["or"]))]
        budget = max_kw - len(r["or"]) - len(r["and"])
    r["not"] = r["not"][:max(0, budget)]
    r["n_kw"] = len(r["or"]) + len(r["and"]) + len(r["not"])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-gram", type=int, default=DEFAULTS["min_gram"])
    ap.add_argument("--nonspam-min-gram", type=int, default=DEFAULTS["nonspam_min_gram"],
                    help="min_gram RIÊNG cho rule non-spam (cụm xác nhận thường ngắn hơn)")
    ap.add_argument("--max-rules", type=int, default=DEFAULTS["max_rules"])
    ap.add_argument("--max-kw", type=int, default=DEFAULTS["max_kw_per_rule"])
    args = ap.parse_args()

    data = read_json(args.input)

    # min_gram thực tế: nếu step3 phải hạ xuống 2 gram thì tôn trọng để không rỗng rule
    eff_min_gram = args.min_gram

    def finalize(rules, min_gram):
        out = []
        for r in rules:
            cr = clean_rule(r, min_gram, args.max_kw)
            if not cr["or"] and not cr["and"]:
                continue  # rule rỗng -> bỏ
            out.append(cr)
        return out[:args.max_rules]

    spam_rules = finalize(data["spam_rules"], eff_min_gram)
    # nếu min_gram=3 làm rỗng hết -> hạ xuống fallback rồi thử lại
    if not spam_rules and eff_min_gram > DEFAULTS["min_gram_fallback"]:
        eff_min_gram = DEFAULTS["min_gram_fallback"]
        spam_rules = finalize(data["spam_rules"], eff_min_gram)

    # rule non-spam dùng min_gram RIÊNG (thấp hơn) -> không bị lọc rỗng oan
    nonspam_rules = finalize(
        data["nonspam_rules"], args.nonspam_min_gram)[:DEFAULTS["max_nonspam_rules"]]

    # đánh số lại cho gọn
    for i, r in enumerate(spam_rules, 1):
        r["id"] = f"SPAM_{i:03d}"
    for i, r in enumerate(nonspam_rules, 1):
        r["id"] = f"NONSPAM_{i:03d}"

    # kiểm tra ràng buộc
    violations = []
    if len(spam_rules) > args.max_rules:
        violations.append(f"#spam_rules={len(spam_rules)} > {args.max_rules}")
    for r, mg in ([(r, eff_min_gram) for r in spam_rules]
                  + [(r, args.nonspam_min_gram) for r in nonspam_rules]):
        if r["n_kw"] > args.max_kw:
            violations.append(f"{r['id']} có {r['n_kw']} kw > {args.max_kw}")
        for k in r["or"]:
            if gram_count(k) < mg:
                violations.append(f"{r['id']} keyword '{k}' < {mg} gram")

    out = {
        "meta": {
            "step": 5,
            "min_gram_effective": eff_min_gram,
            "max_rules": args.max_rules,
            "max_kw_per_rule": args.max_kw,
            "n_spam_rules": len(spam_rules),
            "n_nonspam_rules": len(nonspam_rules),
            "constraints_ok": len(violations) == 0,
            "violations": violations,
        },
        "classification_logic": {
            "priority_1": "Có rule SPAM match -> SPAM (định đoạt)",
            "priority_2": "Chỉ rule NONSPAM match -> NONSPAM (xác nhận)",
            "conflict": "Cả hai match -> uncertain_spam (coi như SPAM)",
            "no_match": "Không match -> uncertain (review tay, coi như NONSPAM)",
            "note": "Rule NONSPAM KHÔNG override SPAM.",
        },
        "spam_rules": spam_rules,
        "nonspam_rules": nonspam_rules,
    }
    write_json(args.output, out)
    print(f"[step5] OK -> {args.output} | spam_rules={len(spam_rules)} "
          f"nonspam_rules={len(nonspam_rules)} constraints_ok={len(violations)==0} "
          f"min_gram={eff_min_gram}")
    if violations:
        print("[step5] VI PHẠM:", violations)


if __name__ == "__main__":
    main()
