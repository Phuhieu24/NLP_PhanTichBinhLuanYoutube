# 🎥 Đồ Án: Phân Tích Chủ Đề Bình Luận YouTube (Topic Modeling)

## 📌 Giới thiệu
Dự án Xử lý ngôn ngữ tự nhiên (NLP) ứng dụng mô hình học máy để tự động thu thập, phân loại và gom cụm chủ đề các bình luận trên nền tảng YouTube.

Hệ thống sử dụng **PhoBERT** kết hợp thuật toán **BERTopic** để phân nhóm dữ liệu tiếng Việt. Ngoài ra, dự án còn tích hợp **Ollama (Local LLM)** để tự động giải nghĩa và tóm tắt chủ đề thành các câu văn hoàn chỉnh. Toàn bộ hệ thống được đóng gói thành một giao diện web trực quan bằng **Streamlit**.

---

## 🚀 Các tính năng chính
1. **Crawl dữ liệu tự động:** Lấy hàng ngàn bình luận từ video YouTube bất kỳ qua YouTube Data API v3.
2. **Tiền xử lý tiếng Việt:** Làm sạch văn bản, tách từ (word tokenization) sử dụng thư viện `pyvi`.
3. **Gom cụm chủ đề (Clustering):** Sử dụng `vietnamese-sbert` (PhoBERT) và `BERTopic` để phân tích ngữ nghĩa. Thuật toán `UMAP` đã được khóa seed để đảm bảo tính nhất quán của kết quả.
4. **Phân tích Cảm xúc Đa lớp (Multiclass Sentiment Analysis):** Tự động huấn luyện mô hình Máy Véc-tơ hỗ trợ (LinearSVC kết hợp TF-IDF N-grams) để phân loại bình luận thành 3 sắc thái: Tích cực / Trung tính / Tiêu cực. Hỗ trợ chuẩn hóa Teencode.
5. **Tóm tắt bằng AI (GenAI):** Tích hợp Ollama (mô hình `qwen2`) để đọc hiểu và tóm tắt chủ đề.
6. **Trực quan hóa:** Các biểu đồ Barchart, Bản đồ không gian 2D (Intertopic Distance Map), và Biểu đồ tròn Cảm xúc (Pie Chart) tương tác trực tiếp trên Web.

---

## ⚙️ Yêu cầu hệ thống
- **Hệ điều hành:** Windows / macOS / Linux.
- **Python:** Phiên bản 3.9 trở lên.
- **Ollama:** Cài đặt sẵn trên máy tính để chạy tính năng tóm tắt AI offline (Tải tại [ollama.com](https://ollama.com/)).

---

## 🛠️ Hướng dẫn Cài đặt

### Bước 1: Thiết lập môi trường Python
1. Mở Terminal / PowerShell tại thư mục dự án.
2. Tạo môi trường ảo (khuyên dùng để tránh xung đột thư viện):
   ```bash
   python -m venv venv
   ```
3. Kích hoạt môi trường ảo:
   - Trên Windows: `.\venv\Scripts\activate`
   - Trên macOS/Linux: `source venv/bin/activate`
4. Cài đặt các thư viện phụ thuộc:
   ```bash
   pip install -r requirements.txt
   ```

### Bước 2: Chuẩn bị mô hình Ollama
Để sử dụng tính năng tóm tắt chủ đề bằng GenAI, bạn cần tải mô hình ngôn ngữ lớn (LLM) hỗ trợ tiếng Việt. Mở Terminal (không cần ở trong môi trường ảo) và chạy lệnh:
```bash
ollama pull qwen2
```
*(Lưu ý: Quá trình này sẽ tải về khoảng ~4.5GB dữ liệu mô hình)*

### Bước 3: Cấu hình API Key
1. Tại thư mục gốc của dự án, hãy tạo một file có tên là `.env`.
2. Đăng nhập Google Cloud Console và lấy mã khóa của **YouTube Data API v3**.
3. Điền cấu hình sau vào file `.env`:
   ```env
   YOUTUBE_API_KEY=điền_api_key_của_bạn_vào_đây
   ```

---

## 🎮 Hướng dẫn Sử dụng

### 1. Huấn luyện Mô hình Cảm xúc (Chỉ làm 1 lần đầu tiên)
Trước khi chạy Web, bạn cần tự tay Train mô hình Sentiment Analysis để dự án sinh ra các file AI (`.pkl`):
```bash
python src/train_sentiment.py
```
*Lưu ý: Quá trình này sẽ đọc file `data/dataset_chuan.csv` (20.000 mẫu) và tự động huấn luyện, đánh giá mô hình phân loại Đa lớp (Tích cực/Trung tính/Tiêu cực).*

### 2. Khởi chạy Giao diện Web
Để khởi chạy trang Web phân tích, hãy chắc chắn bạn đã kích hoạt môi trường ảo (chữ `(venv)` xuất hiện ở đầu dòng Terminal) và chạy lệnh:
```bash
streamlit run src/app.py
```

Trình duyệt sẽ tự động mở lên địa chỉ: **http://localhost:8501**. 
1. Dán Link một video YouTube bất kỳ vào thanh cài đặt bên trái.
2. Điều chỉnh giới hạn bình luận (Nên để 1000 - 2000 để tốc độ tải nhanh).
3. Đánh dấu tích vào ô **"Bật AI tóm tắt chủ đề (Ollama)"** nếu muốn xem kết quả được dịch thành câu hoàn chỉnh.
4. Bấm **Bắt đầu Phân tích**, chờ vài phút để mô hình AI xử lý và tận hưởng kết quả!

---

## 📁 Cấu trúc Thư mục
- `src/app.py`: Mã nguồn chính của giao diện Streamlit Web App.
- `src/crawler.py`: Module kết nối API YouTube và cào dữ liệu.
- `src/preprocess.py`: Module làm sạch và tách từ (Tokenization).
- `src/topic_model.py`: Chứa mã nguồn dự phòng chạy BERTopic trên terminal.
- `src/train_sentiment.py`: Mã nguồn tự huấn luyện mô hình Phân tích Cảm xúc bằng Scikit-Learn.
- `data/`: Nơi lưu trữ các file dữ liệu `.csv` sinh ra trong quá trình chạy và tập dữ liệu huấn luyện.
- `models/`: Chứa các file mô hình Machine Learning (`.pkl`) đã được huấn luyện.
- `results/`: Nơi lưu trữ các file biểu đồ tĩnh `.html`.
- `.streamlit/config.toml`: File cấu hình giao diện Dark Mode cho web.
- `requirements.txt`: Danh sách các thư viện cần cài đặt.
- `BAO_CAO.md`: Tài liệu giải thích lý thuyết các thuật toán.
