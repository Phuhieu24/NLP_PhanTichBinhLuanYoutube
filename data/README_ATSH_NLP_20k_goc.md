# ATSH-NLP-20k — Bộ dữ liệu phân tích cảm xúc bình luận tiếng Việt

Bộ dữ liệu gồm **20.000 bình luận YouTube tiếng Việt** về chương trình *Anh Trai Say Hi* (mùa 1, tập 1–14). Mỗi bình luận có nhãn cảm xúc tổng thể. Có thể dùng cho đề tài môn Xử lý ngôn ngữ tự nhiên, ví dụ phân loại cảm xúc, tiền xử lý teencode/không dấu, so sánh TF-IDF + ML với PhoBERT.

## Nguồn và điều kiện sử dụng

- **Nguồn:** trích từ dự án nghiên cứu ATSH-ABSA của Phạm Xuân Vĩnh Hà (UIT, ĐHQG TP.HCM). Dữ liệu được lấy từ bản gán nhãn tự động (*silver*), không lấy từ bản gán nhãn thủ công (*gold*). Không có bình luận nào trùng với bộ gold.
- **Chỉ dùng cho học tập** trong khuôn khổ môn học. **Không** công bố lại, không đưa lên GitHub, Kaggle, Hugging Face hay bất kỳ nơi công khai nào, và không dùng cho bài báo khi chưa có sự đồng ý của tác giả.
- Khi dùng trong báo cáo, cần ghi: *"Dữ liệu được cung cấp bởi dự án ATSH-ABSA (Phạm Xuân Vĩnh Hà, UIT), chỉ dùng cho mục đích học tập."*
- Dữ liệu không có tên tài khoản hay thông tin của người bình luận. Các bình luận có chứa đường link đã bị loại bỏ.

## Các file

| File | Nội dung |
|---|---|
| `atsh_sentiment_20k.xlsx` | Bản Excel gồm 4 sheet: `du_lieu` (20.000 dòng), `kol_aspect` (nhãn theo từng khía cạnh của nghệ sĩ), `mo_ta_cot`, `thong_ke` |
| `atsh_sentiment_20k.csv` | Toàn bộ 20.000 dòng, mã hoá UTF-8 |
| `atsh_sentiment_train/val/test.csv` | Bộ dữ liệu đã chia sẵn theo tỉ lệ 80/10/10, chia phân tầng theo nhãn |
| `atsh_kol_aspect.csv` | 11.766 dòng, mỗi dòng ứng với một cặp bình luận – nghệ sĩ, kèm nhãn cảm xúc theo 4 khía cạnh. Dùng cho bài toán mở rộng. |

## Các cột chính

| Cột | Ý nghĩa |
|---|---|
| `id` | Mã dòng, từ ATSH00001 đến ATSH20000 |
| `tap` | Tập của chương trình, từ TAP1 đến TAP14 |
| `text` | Nội dung bình luận |
| `label` | Nhãn dạng số: 0 = tiêu cực, 1 = trung tính, 2 = tích cực |
| `sentiment` | Nhãn dạng chữ, tương ứng với cột `label` |
| `doi_tuong` | Đối tượng mà bình luận nhắc tới: `nghe_si`, `chuong_trinh` hoặc `ca_hai` |
| `nghe_si` | Tên các nghệ sĩ được nhắc tới, tối đa 3 người. Cột này để trống nếu bình luận chỉ nói về chương trình. |
| `split` | Bình luận thuộc tập train, val hay test |

## Thống kê

| Tập | Tích cực | Tiêu cực | Trung tính | Tổng |
|---|---:|---:|---:|---:|
| train | 7.672 | 5.600 | 2.727 | 15.999 |
| val | 959 | 700 | 341 | 2.000 |
| test | 960 | 700 | 341 | 2.001 |
| **Tổng** | **9.591** | **7.000** | **3.409** | **20.000** |

Trong dữ liệu gốc, khoảng 88% bình luận là tích cực. Bộ này đã được **lấy mẫu lại để giảm mất cân bằng**, cụ thể:

- Lấy toàn bộ bình luận trung tính hợp lệ.
- Lấy 7.000 bình luận tiêu cực.
- Phần còn lại lấy từ các bình luận tích cực.

Vì vậy, tỉ lệ nhãn trong bộ này **không** phản ánh tỉ lệ thật trên YouTube. Nên nêu rõ điểm này trong báo cáo.

## Cách tạo nhãn tổng thể

- Nhãn gốc được gán theo từng đối tượng (chương trình hoặc nghệ sĩ) và từng khía cạnh: chuyên môn, ngoại hình/phong cách, tính cách, độ nổi tiếng.
- Nhãn tổng thể được gộp như sau:
  - Nếu có ít nhất một nhãn tích cực → **tích cực**.
  - Nếu không có nhãn tích cực nhưng có nhãn tiêu cực → **tiêu cực**.
  - Nếu chỉ có nhãn trung tính → **trung tính**.
- Bình luận **vừa có nhãn tích cực vừa có nhãn tiêu cực** (trường hợp hỗn hợp) đã bị loại để 3 lớp tách bạch hơn.
- Dữ liệu chỉ gồm các bình luận: không phải spam, có liên quan tới chương trình và có thể hiện cảm xúc. Độ dài mỗi bình luận từ 5 đến 500 ký tự, và các bình luận trùng nhau đã được loại bỏ.
- Việc chọn mẫu dùng seed 42 nên có thể tạo lại đúng bộ này.

## Lưu ý về chất lượng nhãn

- Nhãn do mô hình ngôn ngữ lớn gán tự động theo một bộ hướng dẫn gán nhãn, **chưa được người kiểm tra từng dòng**. Vì vậy sẽ có một phần nhãn bị sai.
- Nhãn **trung tính** là nhãn nhiễu nhất: một số câu có sắc thái chê nhẹ hoặc khen nhẹ vẫn bị gán là trung tính.
- Nên kiểm tra thủ công khoảng 100–200 dòng của tập test và ghi tỉ lệ nhãn đúng vào báo cáo. Đây là phần đánh giá dữ liệu mà giảng viên thường đánh giá cao.
- Văn bản còn giữ nguyên emoji, teencode, câu không dấu và mốc thời gian (ví dụ "4:01:27"). Nhóm có thể dùng những đặc điểm này làm nội dung cho phần tiền xử lý.
