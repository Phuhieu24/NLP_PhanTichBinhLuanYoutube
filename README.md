# Phân tích bình luận YouTube tiếng Việt: chủ đề và cảm xúc

Đồ án môn CS221 (Xử lý ngôn ngữ tự nhiên, UIT). Ứng dụng nhận bình luận của một video YouTube, gom cụm chủ đề bằng BERTopic trên vector câu của `keepitreal/vietnamese-sbert` (mô hình Sentence-BERT tiếng Việt, vector 768 chiều; theo `config.json` của mô hình thì được tinh chỉnh từ PhoBERT-base), phân loại cảm xúc ba lớp bằng LinearSVC trên đặc trưng TF-IDF, và tùy chọn tóm tắt từng chủ đề bằng một mô hình ngôn ngữ chạy nội bộ qua Ollama. Báo cáo đầy đủ, số liệu và phân tích nằm trong `BAO_CAO.md`.

## Yêu cầu

- Python 3.11 (đã kiểm thử với 3.11.15). Python 3.12 dự kiến chạy được nhưng chưa thử; Python 3.13 chưa được kiểm thử (wheel của `hdbscan`, `umap-learn` và `torch` thường ra chậm hơn).
- Đủ đĩa trống cho `torch` và mô hình nhúng câu (ước lượng vài GB; mô hình tải về một lần từ Hugging Face Hub, cần mạng ở lần chạy đầu).
- Khóa YouTube Data API v3 chỉ cần khi phân tích video trực tiếp. Hai nguồn dữ liệu còn lại (tệp CSV, dữ liệu mẫu) chạy được không cần khóa.
- Ollama là tùy chọn, chỉ cần cho tính năng tóm tắt chủ đề.

## Cài đặt

```bash
python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\activate
pip install -r requirements.txt   # hoặc requirements-dev.txt nếu muốn chạy kiểm thử
cp .env.example .env              # rồi điền YOUTUBE_API_KEY=... (bỏ qua nếu không phân tích video trực tiếp)
```

Các phiên bản thư viện đã kiểm thử được ghi ở đầu `requirements.txt`. Tệp `.env` nằm trong `.gitignore`.

## Chạy

### Ứng dụng web

```bash
streamlit run src/app.py
```

Trình duyệt mở `http://localhost:8501`. Ở thanh bên, chọn một trong ba nguồn dữ liệu rồi bấm "Bắt đầu phân tích":

| Nguồn | Cần API key | Ghi chú |
|---|---|---|
| Link YouTube | có | Hỗ trợ `youtube.com/watch?v=`, `youtu.be/`, `/shorts/`, `/live/` hoặc ID 11 ký tự. Có thể lấy cả bình luận trả lời và chọn thứ tự (phổ biến hoặc mới nhất). |
| Tệp CSV | không | Tệp UTF-8 có một cột chứa nội dung bình luận; chọn cột trong thanh bên. |
| Dữ liệu mẫu | không | Đọc N dòng đầu của `data/dataset_chuan.csv`. Tệp này là tập huấn luyện của mô hình cảm xúc, nên tỉ lệ cảm xúc hiển thị lạc quan hơn thực tế; chỉ phần chủ đề là minh họa công bằng. |

Phần "Cài đặt nâng cao (BERTopic)" cho phép chỉnh kích thước cụm tối thiểu (mặc định 15), `min_samples` (mặc định 1), cách chọn cụm `eom` hoặc `leaf`, số chủ đề, số từ khóa mỗi chủ đề, và bật hoặc tắt việc loại từ dừng. Nên đọc `BAO_CAO.md` mục 5.2 trước khi đổi: trên 1.500 dòng đầu của dữ liệu mẫu, tham số mặc định của BERTopic (kích thước cụm 10) đẩy 49,2% bình luận vào nhiễu, còn `min_samples` 5 gom 91,0% bình luận vào một chủ đề duy nhất (`results/topic_ablation.txt`). Việc loại từ dừng chỉ đổi từ khóa hiển thị, không đổi số chủ đề hay tỉ lệ nhiễu.

Kết quả được giữ trong phiên làm việc: đổi tab, lọc bảng hay tải CSV không chạy lại bước thu thập và nhúng câu. Nút "Xóa kết quả" xóa phiên hiện tại.

### Huấn luyện lại mô hình cảm xúc (tùy chọn)

Mô hình và số liệu đã được commit trong `models/` và `results/`, nên bước này không bắt buộc. Chạy lại để tái lập các tệp kết quả huấn luyện trong `results/` (khoảng 12 giây trên máy thử nghiệm, không cần mạng):

```bash
python src/train_sentiment.py
```

Script so sánh bốn mô hình nền bằng cross-validation trên tập huấn luyện, dò tham số C cho LinearSVC, đánh giá một lần trên tập kiểm tra, kiểm tra độ ổn định theo năm hạt giống, rồi ghi `models/sentiment_model.pkl`, `models/tfidf_vectorizer.pkl`, `models/model_card.json` và các tệp trong `results/`. Tùy chọn: `--quick` bỏ qua so sánh mô hình, dò C và kiểm tra độ ổn định theo hạt giống; `--seed`, `--data`, `--out-models`, `--out-results`, `--n-jobs`.

### Gom cụm chủ đề từ dòng lệnh

```bash
python src/topic_model.py --input data/dataset_chuan.csv --text-col text --limit 1500 --min-topic-size 15 --min-samples 1 --out-dir results
```

CLI và ứng dụng web gọi cùng một module `src/topic_pipeline.py` nên kết quả hai bên khớp nhau. Lệnh trên cho 1.493 bình luận hợp lệ, 26 chủ đề và 33,3% nhiễu trên máy thử nghiệm (`results/topic_cli_demo.txt`); UMAP có hạt giống cố định nên hai lần chạy trên cùng máy cho cùng kết quả, giữa các máy có thể lệch nhỏ. Ghi `comments_with_topics.csv`, `topics_summary.csv` và ba biểu đồ HTML vào `--out-dir` (các tệp này nằm trong `.gitignore`). Mặc định `--input` là `data/comments.csv`, tức tệp thô do `src/crawler.py` tạo ra. `python src/topic_model.py --help` liệt kê đủ tham số.

### Tái lập các bảng thực nghiệm của báo cáo

```bash
python experiments/dataset_stats.py    # thống kê tập dữ liệu (report/BAO_CAO.md mục 2.2) -> results/dataset_stats.json
python experiments/ab_preprocess.py    # so sánh hai phiên bản tiền xử lý (mục 3.5) -> results/ab_preprocess.txt, .csv
python experiments/topic_ablation.py   # khảo sát tham số HDBSCAN (mục 5.2) -> results/topic_ablation.txt, .csv
```

`experiments/preprocess_baseline.py` là bản sao đóng băng của mã tiền xử lý ban đầu, chỉ dùng cho phép so sánh ở mục 3.5. Các tệp kết quả đã được commit trong `results/`; chạy lại chỉ khi muốn kiểm chứng.

### Kiểm thử

```bash
pip install -r requirements-dev.txt
pytest
```

146 kiểm thử, khoảng 16 giây trên máy thử nghiệm. Hai kiểm thử đầu-cuối (`test_sample_source_runs_the_pipeline_end_to_end` và kiểm thử nguồn YouTube giả lập trong `tests/test_app_smoke.py`) nạp `keepitreal/vietnamese-sbert` từ cache Hugging Face trên máy; hãy chạy ứng dụng hoặc lệnh gom cụm dòng lệnh một lần trước để mô hình được tải về, nếu không hai kiểm thử này thất bại khi máy không có mạng. Các kiểm thử còn lại không cần mạng. Các kiểm thử phủ tiền xử lý, bộ thu thập (giả lập API), pipeline chủ đề (kể cả kiểm tra `BERTopic` được dựng với `language` là `None`), script huấn luyện, lớp gọi Ollama và khởi động ứng dụng Streamlit.

### Tóm tắt chủ đề bằng Ollama (tùy chọn)

```bash
ollama pull qwen2
ollama serve   # nếu Ollama chưa chạy nền
```

Trong ứng dụng, mở "Tóm tắt bằng LLM (Ollama)" ở thanh bên và bật "Bật tóm tắt chủ đề". Có thể đổi tên mô hình và địa chỉ (mặc định `http://localhost:11434/v1`). Ứng dụng kiểm tra kết nối trước khi gọi; nếu Ollama không chạy, bước tóm tắt bị bỏ qua và một cảnh báo hiện ra, các bước khác vẫn hoàn tất.

## Cấu trúc dự án

| Đường dẫn | Nội dung |
|---|---|
| `src/app.py` | Ứng dụng Streamlit: thanh bên, sáu bước xử lý, năm tab kết quả. |
| `src/crawler.py` | Thu thập bình luận qua YouTube Data API v3, tiết kiệm quota, lỗi có phân loại (`CrawlError`). |
| `src/preprocess.py` | Làm sạch, chuẩn hóa teencode, tách từ bằng pyvi (tách từ trước, hạ chữ thường sau), đọc danh sách từ dừng. |
| `src/topic_pipeline.py` | Toàn bộ logic BERTopic (UMAP, HDBSCAN, c-TF-IDF), bảng kết quả và biểu đồ; cả app và CLI đều gọi. |
| `src/topic_model.py` | Công cụ dòng lệnh chạy pipeline chủ đề trên một tệp CSV. |
| `src/train_sentiment.py` | Huấn luyện và đánh giá mô hình cảm xúc, ghi số liệu và thẻ mô hình. |
| `src/llm_summary.py` | Lớp `OllamaSummarizer`: gửi từ khóa và bình luận tiêu biểu tới Ollama, nhận một câu tóm tắt. |
| `src/resources/vietnamese_stopwords.txt` | 236 từ dừng tiếng Việt ở dạng đã tách từ, chỉ dùng cho từ khóa chủ đề. |
| `data/dataset_chuan.csv` | 20.000 bình luận có nhãn dùng huấn luyện; nguồn gốc mô tả trong `data/README.md`. |
| `data/README.md` | Nguồn gốc dữ liệu, định nghĩa nhãn, các tệp CLI ghi vào đây. |
| `models/` | `sentiment_model.pkl`, `tfidf_vectorizer.pkl`, `model_card.json` (tạo bằng scikit-learn 1.9.1). |
| `experiments/` | Script tái lập các bảng thực nghiệm của báo cáo: `dataset_stats.py`, `ab_preprocess.py` (với `preprocess_baseline.py`), `topic_ablation.py`. |
| `results/` | Kết quả huấn luyện: `metrics.json`, `model_comparison.csv`, `classification_report.txt`, `top_features.txt`, hai ảnh ma trận nhầm lẫn. Bằng chứng cho các bảng thực nghiệm: `dataset_stats.json`, `ab_preprocess.txt` và `.csv`, `topic_ablation.txt` và `.csv`, `topic_cli_demo.txt`. |
| `tests/` | Kiểm thử pytest cho từng module trong `src/`. |
| `docs/screenshots/` | Ảnh chụp ứng dụng trên dữ liệu mẫu, dùng trong báo cáo và slide. |
| `.streamlit/config.toml` | Giao diện tối và cấu hình máy chủ Streamlit. |
| `requirements.txt`, `requirements-dev.txt` | Thư viện chạy ứng dụng; bản dev thêm pytest. |
| `report/` | Báo cáo đồ án (`BAO_CAO.md`, dựng ra `BAO_CAO.docx` bằng `scripts/build_report.py`) và slide thuyết trình (`slide.md`, định dạng Marp). |
| `scripts/build_report.py` | Dựng file Word từ báo cáo Markdown. |
| `docs/diagrams/` | Sơ đồ SVG dùng trong báo cáo và slide. |

## Xử lý sự cố

| Hiện tượng | Nguyên nhân và cách xử lý |
|---|---|
| Thông báo hết quota (`quotaExceeded`) | YouTube Data API v3 cấp 10.000 đơn vị mỗi ngày; mỗi trang bình luận tốn 1 đơn vị. Chờ quota được cấp lại vào ngày hôm sau, giảm số bình luận tối đa, hoặc dùng nguồn tệp CSV. |
| Video đã tắt bình luận (`commentsDisabled`) | Không thu thập được; chọn video khác. |
| Khóa API không hợp lệ | Kiểm tra `.env` có dòng `YOUTUBE_API_KEY=...`, khóa đã bật YouTube Data API v3 trong Google Cloud Console, và khởi động lại Streamlit sau khi sửa `.env`. |
| Cảnh báo "Ollama chưa chạy hoặc không truy cập được" | Chạy `ollama serve`, kiểm tra `ollama list` có `qwen2`, và địa chỉ trong thanh bên đúng là `http://localhost:11434/v1`. Không ảnh hưởng các bước khác. |
| Log in cảnh báo về `use_container_width` bị deprecated | Thông báo của Streamlit phiên bản mới, không ảnh hưởng kết quả. |
| App không tự nạp lại khi sửa mã | `fileWatcherType = "none"` trong `.streamlit/config.toml`: bộ theo dõi tệp quét mọi module đã nạp, chạm vào các module ảnh của `transformers` và in traceback `No module named 'torchvision'` dù app không dùng tới. Đổi thành `"auto"` khi cần phát triển. |
| Không nạp được `models/*.pkl` | Pickle được tạo bằng scikit-learn 1.9.1; nếu phiên bản cài khác nhiều, chạy `python src/train_sentiment.py` để tạo lại. |
| Lần chạy đầu rất lâu | Đang tải `keepitreal/vietnamese-sbert` từ Hugging Face Hub; các lần sau dùng cache. |

## Giới hạn đã biết

- Theo kinh nghiệm cộng đồng, YouTube ngừng phân trang ở khoảng 1.000 chuỗi bình luận gốc mỗi video (chưa kiểm chứng trong đồ án); nếu đúng, số bình luận thu được có trần dù thanh trượt cho tới 5.000.
- Quota 10.000 đơn vị mỗi ngày giới hạn số video phân tích được trong ngày.
- Emoji bị xóa ở bước làm sạch; bình luận không dấu không được khôi phục dấu.
- Mô hình cảm xúc huấn luyện trên bình luận của một chương trình duy nhất; hiệu năng trên video thuộc chủ đề khác chưa được đo. Chi tiết trong `report/BAO_CAO.md` mục 9.
