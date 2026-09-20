# Sửa danh sách thành viên (7 -> 5) và thêm thông tin giảng viên

Ngày bắt đầu: 2026-09-20

## Bối cảnh cần cho phiên mới
- Repo: `/Users/narblack/Code/_course-projects/CS221-F31_XLNNTN_course-proj`, nhánh `review-2026-09-18`.
- Chủ repo đã tự sửa `report/slide.md`: bỏ Nguyễn Ngọc Bích (25730012) và Nguyễn Anh Tài (25730063),
  thêm dòng `Giảng viên: TS. Đặng Văn Thìn`. Còn 5 thành viên, tất cả lớp LT.K2026.1.TTNT.
- Cần đồng bộ sang: `scripts/build_slides_pptx.py` (nguồn của `report/CS221_slide.pptx`),
  `report/BAO_CAO.md` (nguồn của `report/BAO_CAO.docx`), và phần ghi chú người nói (bài nói).
- `review-2026-09-19/` là gói bàn giao, GIỮ NGUYÊN, không sửa dù trong đó có bản slide.md/BAO_CAO.md cũ.

## Quyết định của chủ repo (2026-09-20)
- Học vị giảng viên: dùng **TS.** ở mọi chỗ, kể cả các dòng tài liệu tham khảo đang ghi `NCS.ThS`.
- Thông tin giảng viên phải hiện rõ trên slide bìa (yêu cầu "rất quan trọng").

## Danh sách thành viên đúng (5 người)
| Họ tên | MSSV | Lớp |
|---|---|---|
| Lê Phú Hiếu | 26410038 | LT.K2026.1.TTNT |
| Nguyễn Thanh Duy | 26410030 | LT.K2026.1.TTNT |
| Nguyễn Thanh Phong | 26410090 | LT.K2026.1.TTNT |
| Nguyễn Thị Mai Thi | 26410117 | LT.K2026.1.TTNT |
| Hồ Viết Trịnh | 26410140 | LT.K2026.1.TTNT |

## Checklist
- [x] Ghi checklist này
- [x] `report/slide.md`: đổi `NCS.ThS` -> `TS.` ở phụ lục B1, cập nhật ghi chú bìa (chào thầy theo tên)
- [x] `scripts/build_slides_pptx.py`: members còn 5, thêm dòng giảng viên lên bìa, đổi `NCS.ThS` -> `TS.`, cập nhật notes bìa
- [x] Dựng lại `report/CS221_slide.pptx` (theme slate, mặc định)
- [x] Kiểm tra bìa pptx: 5 thành viên, có dòng giảng viên, 18 slide, không tràn khung
- [x] `report/BAO_CAO.md`: bỏ 2 thành viên, đổi 6 chỗ `NCS.ThS` -> `TS.`
- [x] Dựng lại `report/BAO_CAO.docx`, kiểm tra header
- [x] Grep toàn repo (trừ `review-2026-09-19/` và `.venv/`) xác nhận không còn 25730012 / 25730063 / NCS.ThS
- [x] Commit

## Kết quả (2026-09-20, hoàn tất)
- `report/slide.md`: bìa có `Giảng viên hướng dẫn: **TS. Đặng Văn Thìn**`, nhãn `Nhóm thực hiện`, 5 thành viên;
  phụ lục B1 đổi sang `TS.`; ghi chú bìa chào thầy theo tên và nói rõ nhóm năm thành viên.
- `scripts/build_slides_pptx.py`: bìa thêm dòng giảng viên (15pt, màu `ink`, in đậm, y=4,76) và nhãn
  `NHÓM THỰC HIỆN` (y=5,34); thành viên chia 3 trái / 2 phải tại y=5,70. Ghi chú bìa đồng bộ với slide.md.
- `report/CS221_slide.pptx`: dựng lại theme slate, 18 slide, 852 KB. Bìa đã kiểm: 5 thành viên, có dòng giảng viên.
- `report/BAO_CAO.md` + `report/BAO_CAO.docx`: bỏ 2 thành viên, 6 chỗ `NCS.ThS` -> `TS.`, đổi nhãn thành
  `Giảng viên hướng dẫn`. Docx dựng lại: 31 tiêu đề, 118 đoạn, 12 bảng, 8 hình, 0 hình thiếu, 1.077 KB.
- Grep toàn repo (trừ `.venv/`, `.git/`, `review-2026-09-19/`): không còn 25730012, 25730063, `NCS.ThS`.
  Kiểm cả nội dung .docx và .pptx bằng python-docx / python-pptx: sạch.
- Nhịp bài nói: 2.217 chữ ghi chú, ~14,3 phút ở 155 chữ/phút (không đổi đáng kể, phần thêm vào bìa là +8 chữ).
- KHÔNG đụng vào `review-2026-09-19/` (gói bàn giao còn bản cũ 7 thành viên, đúng như thỏa thuận giữ nguyên).
