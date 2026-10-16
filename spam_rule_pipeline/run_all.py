# -*- coding: utf-8 -*-
"""
run_all.py — chạy tuần tự cả 6 bước và GOM MỌI OUTPUT VÀO 1 FOLDER.

Tên folder = tên file input (bỏ phần mở rộng). Nếu folder đã tồn tại thì tự
thêm đuôi _v1, _v2, _v3, ... Mỗi bước vẫn có thể chạy độc lập bằng step*.py.

Chạy:
  python run_all.py --input ../longchau2.xlsx
  python run_all.py --input ../longchau2.xlsx --no-phobert-tokenizer
  python run_all.py --input ../longchau2.xlsx --outbase ../results   # đặt folder ở nơi khác
"""
import argparse
import os
import subprocess
import sys

from common import make_output_dir

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print("\n>>>", " ".join(cmd))
    r = subprocess.run([sys.executable] + cmd, cwd=HERE)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outbase", default=None,
                    help="thư mục cha để đặt folder output (mặc định = thư mục chứa file input)")
    ap.add_argument("--no-phobert-tokenizer", action="store_true")
    ap.add_argument("--segmenter", choices=["auto", "vncorenlp", "underthesea", "whitespace"],
                    default="auto", help="bộ tách từ B1 (auto ưu tiên VnCoreNLP)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    inp = os.path.abspath(args.input)
    # tạo folder output: tên trùng file input, tự thêm _v1/_v2... nếu đã có
    outdir = make_output_dir(inp, parent=args.outbase)
    tag = os.path.basename(outdir)   # đuôi file = tên folder (vd 'longchau2', 'longchau2_v1')
    print(f"[run_all] Output folder: {outdir}")

    s1 = os.path.join(outdir, f"step1_segmented_{tag}.json")
    s2 = os.path.join(outdir, f"step2_scores_{tag}.json")
    s3 = os.path.join(outdir, f"step3_candidates_{tag}.json")
    s4 = os.path.join(outdir, f"step4_rules_{tag}.json")
    s5 = os.path.join(outdir, f"step5_final_rules_{tag}.json")
    xlsx = os.path.join(outdir, f"report_{tag}.xlsx")

    step1 = ["step1_segment.py", "--input", inp, "--output", s1,
             "--segmenter", args.segmenter]
    if args.no_phobert_tokenizer:
        step1.append("--no-phobert-tokenizer")
    if args.limit:
        step1 += ["--limit", str(args.limit)]
    run(step1)
    run(["step2_score.py", "--input", s1, "--output", s2])
    run(["step3_candidates.py", "--step1", s1, "--step2", s2, "--output", s3])
    run(["step4_select_rules.py", "--step1", s1, "--step3", s3, "--output", s4])
    run(["step5_finalize.py", "--input", s4, "--output", s5])
    run(["step6_export.py", "--step1", s1, "--step5", s5, "--output", xlsx])
    print(f"\n[run_all] DONE. Tất cả output nằm trong: {outdir}")
    print(f"[run_all] Báo cáo Excel: {xlsx}")


if __name__ == "__main__":
    main()
