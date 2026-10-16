# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Rulebase Spam pipeline — web (Flask) + 6-step pipeline
# Import 2 file Excel (spam / nonspam) -> chạy pipeline -> xem & tải report.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# underthesea / scipy / sklearn cần toolchain để build 1 số wheel
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # kết quả mỗi lần chạy ghi vào /data (mount volume để lưu ra host)
    RULEBASE_OUTPUT_DIR=/data

WORKDIR /app/spam_rule_pipeline

# cài phụ thuộc trước (tận dụng cache layer)
COPY spam_rule_pipeline/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# nạp sẵn model underthesea để lần chạy đầu không phải tải
RUN python -c "import underthesea; underthesea.word_tokenize('xin chào bạn')" || true

# copy mã nguồn pipeline
COPY spam_rule_pipeline/ /app/spam_rule_pipeline/

RUN mkdir -p /data
VOLUME ["/data"]
EXPOSE 5000

# Production: gunicorn.
# QUAN TRỌNG: --workers 1 (bắt buộc) vì trạng thái job lưu trong BỘ NHỚ dùng
# chung giữa các request; nhiều worker sẽ khiến polling trúng worker khác -> 404.
# --threads để phục vụ song song nhiều request polling + upload.
CMD ["gunicorn", "--workers", "1", "--threads", "8", "--timeout", "120", \
     "--bind", "0.0.0.0:5000", "--access-logfile", "-", "webapp:app"]
