# Dữ liệu

Thư mục này chứa tập dữ liệu huấn luyện `dataset_chuan.csv` (được commit) và các tệp sinh ra khi chạy công cụ dòng lệnh (không commit).

## Nguồn gốc của `dataset_chuan.csv`

Các mục dưới đây không suy ra được từ mã nguồn hay từ chính tệp dữ liệu. Nhóm điền trước khi nộp; mục nào không biết thì ghi "không rõ" thay vì bỏ trống.

| Mục | Nội dung |
|---|---|
| Video hoặc kênh nguồn | TODO (danh sách link video, số video, kênh) |
| Thời điểm thu thập | TODO (khoảng ngày) |
| Công cụ thu thập | TODO (`src/crawler.py` ở phiên bản nào, hay công cụ khác) |
| Quy tắc lấy mẫu từ tập thô | TODO (xem ghi chú về số tròn bên dưới) |
| Cách gán nhãn | TODO (gán tay, dùng LLM, hay heuristic; nếu nhiều bước thì mô tả từng bước) |
| Người gán nhãn | TODO (ai, bao nhiêu người, mỗi bình luận có mấy người gán) |
| Hướng dẫn gán nhãn | TODO (đường dẫn tới tài liệu quy ước ba lớp) |
| Độ đồng thuận giữa người gán | TODO (Cohen's kappa hoặc tỉ lệ trùng khớp, trên bao nhiêu mẫu; ghi "chưa đo" nếu chưa đo) |
| Giấy phép và điều khoản sử dụng | TODO (điều khoản dịch vụ YouTube API, phạm vi sử dụng cho mục đích học tập) |
| Xử lý dữ liệu cá nhân | Tệp chỉ có hai cột `text` và `label`; không có tên tác giả, mã bình luận hay thời điểm đăng. TODO xác nhận tệp thô (nếu còn giữ) có được lưu ngoài repo không. |

## Thống kê

Kiểm tra ngày 18-09-2026 trên tệp đã commit.

| Chỉ số | Giá trị |
|---|---|
| Số dòng | 20.000 |
| Cột | `text` (chuỗi), `label` (số nguyên 0, 1, 2) |
| Nhãn 0, tiêu cực | 7.000 (35,0%) |
| Nhãn 1, trung tính | 3.409 (17,0%) |
| Nhãn 2, tích cực | 9.591 (48,0%) |
| Dòng trùng nội dung | 0 |
| Dòng thiếu giá trị | 0 |
| Độ dài bình luận | 5 đến 500 ký tự, trung vị 52 |
| Bình luận có 3 ký tự chữ lặp liên tiếp trở lên | 21% |
| Bình luận có emoji hoặc dấu câu | 71% |
| Chủ đề | Toàn bộ bình luận nói về chương trình "Anh Trai Say Hi" |

Hai ghi chú:

- Số 7.000 tròn cho nhãn 0 và tổng 20.000 tròn cho thấy tệp là kết quả của một bước lấy mẫu từ tập thô lớn hơn. Quy tắc lấy mẫu (ngẫu nhiên, theo lớp, theo video) cần ghi vào bảng nguồn gốc ở trên.
- 3 dòng trở thành rỗng sau tiền xử lý (chỉ gồm emoji, dấu câu hoặc URL) và bị loại trước khi huấn luyện; 19.997 dòng còn lại được chia thành 15.997 dòng huấn luyện và 4.000 dòng kiểm tra (nhãn 2 còn 9.588). Xem `results/metrics.json`, khóa `dataset` và `split`.

## Định nghĩa nhãn

| Nhãn | Tên | Cách đọc từ dữ liệu |
|---|---|---|
| 0 | Tiêu cực | Chê, thất vọng, bức xúc với tiết mục, thí sinh, giám khảo hoặc khâu tổ chức. |
| 1 | Trung tính | Nhận xét không nghiêng về khen hay chê, câu hỏi, câu kể, hoặc vừa khen vừa chê. |
| 2 | Tích cực | Khen, yêu thích, cảm động. |

Đây là cách nhóm đọc lại từ dữ liệu. Hướng dẫn gán nhãn gốc là mục TODO trong bảng nguồn gốc.

## Tệp sinh ra khi chạy (không commit)

| Tệp | Do lệnh nào tạo | Nội dung |
|---|---|---|
| `comments.csv` | `python src/crawler.py --url <link video>` | Bình luận thô theo hợp đồng cột `comment_id`, `parent_id`, `is_reply`, `author`, `published_at`, `like_count`, `reply_count`, `text`. |
| `comments_clean.csv` | `python src/preprocess.py` | `comments.csv` thêm hai cột `clean_text` và `tokenized_text`, bỏ dòng dưới 2 token. |
| `comments_with_topics.csv` | `python src/topic_model.py --out-dir data` | Bình luận kèm cột `Topic` và `Topic_Name` (chỉ khi `--out-dir` trỏ vào đây; mặc định là `results/`). |

Cả ba tệp nằm trong `.gitignore` vì chứa tên tác giả và nội dung bình luận thu thập trực tiếp từ YouTube. Không commit chúng.

## Lưu ý khi dùng nguồn "Dữ liệu mẫu" trong ứng dụng

Nguồn "Dữ liệu mẫu" của ứng dụng Streamlit đọc N dòng đầu của chính `dataset_chuan.csv`, tức là tập huấn luyện của mô hình cảm xúc. Tỉ lệ cảm xúc hiển thị ở chế độ này lạc quan hơn kết quả trên bình luận mới, vì mô hình đã thấy các dòng đó khi huấn luyện. Chỉ phần gom cụm chủ đề (không dùng nhãn) là minh họa công bằng. Muốn đánh giá mô hình cảm xúc, dùng các số liệu trên tập kiểm tra trong `results/metrics.json`, không dùng tỉ lệ hiển thị trên ứng dụng.
