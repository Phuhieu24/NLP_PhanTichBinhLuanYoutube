# experiments/ - các thí nghiệm sinh ra bằng chứng cho báo cáo

Các script trong thư mục này sinh ra các tệp trong `results/` mà `BAO_CAO.md` trích dẫn.
Chúng chạy bằng đúng mã nguồn trong `src/`, không phụ thuộc mạng (ngoài lần đầu tải mô
hình nhúng câu), nên người chấm chạy lại được và ra đúng những con số trong báo cáo.

Chạy từ thư mục gốc dự án:

```bash
python experiments/dataset_stats.py      # ~30 giây  -> results/dataset_stats.json
python experiments/ab_preprocess.py      # ~1-2 phút -> results/ab_preprocess.csv, results/ab_preprocess.txt
python experiments/topic_ablation.py     # ~1 phút   -> results/topic_ablation.csv, results/topic_ablation.txt
```

| Script | Làm gì | Ghi ra |
|---|---|---|
| `dataset_stats.py` | Thống kê mô tả `data/dataset_chuan.csv`: số dòng, phân bố nhãn, trùng lặp, độ dài, tỷ lệ bình luận có emoji / dấu câu / ký tự lặp, số dòng chưa chuẩn hóa NFC, số dòng rỗng sau tiền xử lý. Ghi kèm định nghĩa (biểu thức chính quy, cách nhận diện emoji) để kiểm chứng lại được. | `results/dataset_stats.json` |
| `ab_preprocess.py` | So sánh A/B bản tiền xử lý gốc ở commit `afb1c3e` với bản đang xuất xưởng, giữ nguyên TF-IDF và LinearSVC, chạy trên các hạt giống 0-4 và 42, với C = 0.3 và C = 1.0. | `results/ab_preprocess.csv` (từng lần chạy), `results/ab_preprocess.txt` (bảng tổng hợp) |
| `topic_ablation.py` | Năm cấu hình HDBSCAN trên 1.500 dòng đầu tiên, nhúng câu một lần rồi dùng lại, đi qua đúng `src/topic_pipeline.py`; kèm một lần chạy tắt từ dừng để kiểm tra xem danh sách từ dừng có làm đổi cột `Topic` hay không. | `results/topic_ablation.csv`, `results/topic_ablation.txt` |
| `export_misclassified.py` | Dựng lại đúng tập test của `src/train_sentiment.py` (seed 42), dự đoán bằng mô hình đã lưu, đối chiếu accuracy với `results/metrics.json` rồi xuất 935 dòng bị phân loại sai kèm `margin` (điểm lớp dự đoán trừ điểm lớp đúng), thêm một mẫu phân tầng 150 dòng có cột trống `hien_tuong` để gán nhãn hiện tượng lỗi bằng tay. | `results/misclassified_test.csv`, `results/error_sample_150.csv` |

`preprocess_baseline.py` là bản sao đóng băng phần tiền xử lý ở commit `afb1c3e`, chỉ
phục vụ so sánh A/B. Không import tệp này ở bất kỳ đâu trong `src/`.

Bản demo dòng lệnh của pipeline chủ đề nằm sẵn ở `results/topic_cli_demo.txt` (đầu ra
của `src/topic_model.py` với `--limit 1500 --min-topic-size 15 --min-samples 1`).

Lưu ý khi chạy lại: kết quả gom cụm chỉ tất định trên cùng một máy và cùng phiên bản
thư viện; UMAP với `random_state=42` cho kết quả lặp lại được trên cùng máy, nhưng số
chủ đề có thể lệch vài đơn vị khi đổi kiến trúc CPU hoặc phiên bản `umap-learn`,
`hdbscan`. Phiên bản đã dùng được ghi ngay trong `results/topic_ablation.txt`.
