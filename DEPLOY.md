# Triển khai lên server nội bộ (LAN) — truy cập qua IP:port

App chạy bằng **Docker + gunicorn** (production). Người dùng trong mạng LAN mở
`http://<IP-server>:5000`.

> Lưu ý kiến trúc: app giữ trạng thái job **trong bộ nhớ** nên container chạy
> **1 worker gunicorn + nhiều thread**. Không tăng số worker (sẽ hỏng polling
> tiến độ). Với quy mô 1 nhóm nội bộ, 1 worker là đủ.

---

## 1. Yêu cầu trên server

- Linux (Ubuntu/Debian/CentOS... đều được) có **Docker** + **Docker Compose v2**.
  ```bash
  docker --version
  docker compose version
  ```
  Chưa có Docker → cài nhanh:
  ```bash
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker $USER   # đăng xuất/đăng nhập lại để dùng docker không cần sudo
  ```

## 2. Đưa mã nguồn lên server

Chỉ cần các file sau (không cần copy dữ liệu .xlsx/kết quả cũ):

```
rule-2/
├─ Dockerfile
├─ docker-compose.yml
├─ .dockerignore
└─ spam_rule_pipeline/      (toàn bộ mã nguồn pipeline + requirements.txt)
```

Cách copy (chọn 1):

```bash
# a) Dùng git (khuyến nghị nếu có repo)
git clone <repo-url> rule-2 && cd rule-2

# b) Hoặc đẩy thẳng từ máy bạn qua scp/rsync
rsync -av --exclude '.venv*' --exclude 'output' --exclude '*.xlsx' \
      ./rule-2/ user@<IP-server>:/opt/rulebase/
```

## 3. Chạy

```bash
cd /opt/rulebase          # hoặc thư mục rule-2 bạn vừa copy lên
docker compose up -d --build
```

- Lần đầu build ~10–15 phút (tải thư viện). Các lần sau dùng cache, nhanh.
- Kiểm tra:
  ```bash
  docker compose ps           # thấy state "running (healthy)"
  curl -I http://localhost:5000   # HTTP 200
  ```

## 4. Mở firewall cho cổng 5000

```bash
# Ubuntu (ufw)
sudo ufw allow 5000/tcp

# CentOS/RHEL (firewalld)
sudo firewall-cmd --permanent --add-port=5000/tcp && sudo firewall-cmd --reload
```

Tìm IP server: `hostname -I` (lấy IP LAN, vd `192.168.1.50`).

→ Người dùng mở: **`http://192.168.1.50:5000`**

### Muốn truy cập bằng cổng 80 (bỏ `:5000`)
Sửa `docker-compose.yml`, đổi `"5000:5000"` thành `"80:5000"`, rồi
`docker compose up -d`. Mở firewall cổng 80. Người dùng mở `http://192.168.1.50`.

## 5. Dữ liệu & kết quả

- Kết quả mỗi lần chạy lưu ở **`./output/<dataset>/`** ngay trên server (volume gắn
  vào `/data` trong container). Sao lưu định kỳ thư mục này nếu cần.
- "Danh mục đã chạy" trên web đọc từ chính thư mục này.

## 6. Vận hành

```bash
docker compose logs -f          # xem log realtime
docker compose restart          # khởi động lại
docker compose down             # tắt (giữ nguyên ./output)
docker compose up -d            # bật lại
```

Container đã đặt `restart: unless-stopped` → tự bật lại khi server reboot hoặc
app crash.

## 7. Cập nhật phiên bản mới

```bash
cd /opt/rulebase
git pull                        # hoặc rsync lại mã nguồn mới
docker compose up -d --build    # build lại & thay container, ./output giữ nguyên
```

---

## Ghi chú bảo mật (LAN nội bộ)

- App **không có đăng nhập** — ai vào được mạng LAN + biết IP:port là dùng được.
  Nếu cần giới hạn, có thể: giới hạn firewall theo dải IP, hoặc đặt sau một
  reverse proxy có Basic Auth. Nói nếu bạn muốn thêm.
- Giới hạn kích thước file upload hiện là 200MB (`MAX_CONTENT_LENGTH` trong
  `webapp.py`).
