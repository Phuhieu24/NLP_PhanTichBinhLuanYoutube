# Dữ liệu

Thư mục này chứa tập dữ liệu huấn luyện `dataset_chuan.csv` (được commit) và các tệp sinh ra khi chạy công cụ dòng lệnh (không commit).

## Nguồn gốc của `dataset_chuan.csv`

`dataset_chuan.csv` là hai cột `text` và `label` của tệp `atsh_sentiment_20k.csv` trong gói dữ liệu ATSH-NLP-20k, đã đối chiếu từng dòng: 20.000 dòng, cùng thứ tự, cùng nhãn. Mọi thông tin dưới đây lấy từ README của tác giả gói dữ liệu, sao chép nguyên văn tại `data/README_ATSH_NLP_20k_goc.md`.

| Mục | Nội dung theo README của tác giả |
|---|---|
| Nguồn | Trích từ dự án nghiên cứu ATSH-ABSA của Phạm Xuân Vĩnh Hà (UIT, ĐHQG TP.HCM). Lấy từ bản gán nhãn tự động (silver), không lấy từ bản gán nhãn thủ công (gold); không có bình luận nào trùng với bộ gold. |
| Nội dung | Bình luận YouTube tiếng Việt về chương trình Anh Trai Say Hi, mùa 1, tập 1 đến 14. |
| Cách gán nhãn | Nhãn do mô hình ngôn ngữ lớn gán tự động theo một bộ hướng dẫn gán nhãn, theo từng đối tượng (chương trình hoặc nghệ sĩ) và từng khía cạnh (chuyên môn, ngoại hình/phong cách, tính cách, độ nổi tiếng). Chưa được người kiểm tra từng dòng. Tác giả ghi nhãn trung tính là nhãn nhiễu nhất và khuyên kiểm tra thủ công 100 đến 200 dòng của tập test rồi ghi tỉ lệ nhãn đúng vào báo cáo. |
| Gộp nhãn tổng thể | Có ít nhất một nhãn tích cực: tích cực. Không có nhãn tích cực nhưng có nhãn tiêu cực: tiêu cực. Chỉ có nhãn trung tính: trung tính. Bình luận vừa có nhãn tích cực vừa có nhãn tiêu cực (hỗn hợp) đã bị loại. |
| Lấy mẫu lại | Dữ liệu gốc khoảng 88% tích cực. Bộ này lấy toàn bộ bình luận trung tính hợp lệ, 7.000 bình luận tiêu cực, phần còn lại lấy từ bình luận tích cực; chọn mẫu với seed 42. Tỉ lệ nhãn vì vậy không phản ánh tỉ lệ thật trên YouTube. |
| Bộ lọc | Không phải spam, có liên quan tới chương trình, có thể hiện cảm xúc, dài 5 đến 500 ký tự; đã loại bình luận trùng nhau và bình luận có chứa đường link. |
| Dữ liệu cá nhân | Không có tên tài khoản hay thông tin của người bình luận. `dataset_chuan.csv` chỉ giữ hai cột `text` và `label`. |
| Điều kiện sử dụng | "Chỉ dùng cho học tập trong khuôn khổ môn học. Không công bố lại, không đưa lên GitHub, Kaggle, Hugging Face hay bất kỳ nơi công khai nào, và không dùng cho bài báo khi chưa có sự đồng ý của tác giả." |
| Câu trích dẫn bắt buộc | "Dữ liệu được cung cấp bởi dự án ATSH-ABSA (Phạm Xuân Vĩnh Hà, UIT), chỉ dùng cho mục đích học tập." |
| Các cột và file khác trong gói gốc | Cột `id`, `tap` (TAP1 đến TAP14), `sentiment` (nhãn dạng chữ), `doi_tuong` (`nghe_si`, `chuong_trinh`, `ca_hai`), `nghe_si` (tối đa 3 tên), `split`; bộ chia sẵn train/val/test 15.999 / 2.000 / 2.001 (80/10/10, phân tầng theo nhãn); `atsh_kol_aspect.csv` gồm 11.766 cặp bình luận và nghệ sĩ kèm nhãn cảm xúc theo 4 khía cạnh. Nhóm không dùng các cột và tệp này; phép chia 80/20 với hạt giống 42 trong đánh giá là của nhóm, không phải bộ chia sẵn. |
| Tệp README gốc | `data/README_ATSH_NLP_20k_goc.md` |

Theo điều kiện sử dụng trên, repo này phải giữ ở chế độ riêng tư và `dataset_chuan.csv` không được đẩy lên bất kỳ nơi công khai nào.

## Thống kê

Kiểm tra ngày 18-09-2026 trên tệp đã commit; số liệu do `experiments/dataset_stats.py` tính và ghi vào `results/dataset_stats.json`.

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
| Bình luận có 3 chữ cái giống nhau liên tiếp trở lên | 7,75% (20,7% nếu tính cả dấu câu và emoji lặp) |
| Bình luận có emoji | 34,5% |
| Bình luận có dấu câu | 51,5% |
| Bình luận có emoji hoặc dấu câu | 71,2% |
| Chủ đề | Toàn bộ bình luận nói về chương trình "Anh Trai Say Hi" |

Hai ghi chú:

- Số 7.000 tròn cho nhãn 0 và tổng 20.000 tròn là kết quả của bước lấy mẫu lại do tác giả gói dữ liệu thực hiện (bảng nguồn gốc ở trên): giữ toàn bộ bình luận trung tính hợp lệ, lấy đúng 7.000 bình luận tiêu cực, bù phần còn lại bằng bình luận tích cực cho đủ 20.000. Tỉ lệ ba lớp vì vậy do tác giả chọn để giảm mất cân bằng, không phải tỉ lệ thật trên YouTube (dữ liệu gốc khoảng 88% tích cực).
- 3 dòng trở thành rỗng sau tiền xử lý (chỉ gồm emoji, dấu câu hoặc URL) và bị loại trước khi huấn luyện; 19.997 dòng còn lại được chia thành 15.997 dòng huấn luyện và 4.000 dòng kiểm tra (nhãn 2 còn 9.588). Xem `results/metrics.json`, khóa `dataset` và `split`.

## Định nghĩa nhãn

| Nhãn | Tên | Cách đọc từ dữ liệu |
|---|---|---|
| 0 | Tiêu cực | Chê, thất vọng, bức xúc với tiết mục, thí sinh, giám khảo hoặc khâu tổ chức. |
| 1 | Trung tính | Nhận xét không nghiêng về khen hay chê, câu hỏi, câu kể, hoặc vừa khen vừa chê. |
| 2 | Tích cực | Khen, yêu thích, cảm động. |

Cột "Cách đọc từ dữ liệu" là cách nhóm đọc lại từ chính tệp. Quy tắc gốc của tác giả (`data/README_ATSH_NLP_20k_goc.md`): nhãn được gán theo từng đối tượng và từng khía cạnh, rồi gộp thành một nhãn tổng thể. Có ít nhất một nhãn tích cực thì tích cực; không có tích cực nhưng có tiêu cực thì tiêu cực; chỉ có trung tính thì trung tính. Bình luận vừa có nhãn tích cực vừa có nhãn tiêu cực đã bị loại. Tác giả cũng ghi nhãn trung tính là nhãn nhiễu nhất, một số câu chê nhẹ hoặc khen nhẹ vẫn được gán trung tính; điều này khớp với việc nhóm gặp câu vừa khen vừa chê trong lớp 1.

## Tệp sinh ra khi chạy (không commit)

| Tệp | Do lệnh nào tạo | Nội dung |
|---|---|---|
| `comments.csv` | `python src/crawler.py --url <link video>` | Bình luận thô theo hợp đồng cột `comment_id`, `parent_id`, `is_reply`, `author`, `published_at`, `like_count`, `reply_count`, `text`. |
| `comments_clean.csv` | `python src/preprocess.py` | `comments.csv` thêm hai cột `clean_text` và `tokenized_text`, bỏ dòng dưới 2 token. |
| `comments_with_topics.csv` | `python src/topic_model.py --out-dir data` | Bình luận kèm cột `Topic` và `Topic_Name` (chỉ khi `--out-dir` trỏ vào đây; mặc định là `results/`). |
| `topics_summary.csv` và ba tệp `.html` | cùng lệnh trên | Bảng từ khóa theo chủ đề và ba biểu đồ BERTopic; cũng chỉ xuất hiện ở đây khi `--out-dir data`. |

Các tệp này nằm trong `.gitignore`: ba tệp CSV đầu vì chứa tên tác giả và nội dung bình luận thu thập trực tiếp từ YouTube, các tệp còn lại vì là sản phẩm sinh ra khi chạy. Không commit chúng.

## Lưu ý khi dùng nguồn "Dữ liệu mẫu" trong ứng dụng

Nguồn "Dữ liệu mẫu" của ứng dụng Streamlit đọc N dòng đầu của chính `dataset_chuan.csv`, tức là tập huấn luyện của mô hình cảm xúc. Tỉ lệ cảm xúc hiển thị ở chế độ này lạc quan hơn kết quả trên bình luận mới, vì mô hình đã thấy các dòng đó khi huấn luyện. Chỉ phần gom cụm chủ đề (không dùng nhãn) là minh họa công bằng. Muốn đánh giá mô hình cảm xúc, dùng các số liệu trên tập kiểm tra trong `results/metrics.json`, không dùng tỉ lệ hiển thị trên ứng dụng.
