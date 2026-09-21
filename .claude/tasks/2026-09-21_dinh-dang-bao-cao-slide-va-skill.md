# Định dạng báo cáo/slide theo review của chủ repo, bỏ tên tác giả dữ liệu, skill dùng chung

Ngày bắt đầu: 2026-09-21

## Bối cảnh cần cho phiên mới
- Repo: `/Users/narblack/Code/_course-projects/CS221-F31_XLNNTN_course-proj`, nhánh `main`.
- Chủ repo đã sửa tay `report/BAO_CAO.docx` (bản working tree, chưa commit). Bản sao phân tích: scratchpad.
  Học được từ bản sửa tay (so với bản dựng 048999e):
  - Bỏ trang bìa và đoạn "Quy ước số". Trang 1 = mục lục, tiêu đề "Nội dung báo cáo đồ án" (style TOC Heading),
    field `TOC \o "1-3" \h \z \u`, rồi ngắt trang. Bìa in rời theo mẫu UIT (repo CourseProj-Report_w_Pandoc).
  - `pgNumType start=2`; header/footer ở mọi trang kể cả trang mục lục.
  - Header: "Xử Lý Ngôn Ngữ Tự Nhiên" (viết hoa chữ đầu mỗi từ), màu 7F7F7F, gạch dưới 0,5pt xám.
  - Footer: tên đề tài bên trái, số trang bên phải, màu 7F7F7F, gạch trên 0,5pt xám.
  - Heading bỏ font theme, dùng Times New Roman. H1 20pt đậm 365F91, H2 16pt đậm 4F81BD, H3 14pt đậm 4F81BD.
  - Thân bài 12pt, bảng 12pt, chú thích hình 11pt nghiêng. Bảng rộng 16 cm (hết bề ngang chữ).
  - Hình ma trận nhầm lẫn thu từ 15,5 cm xuống 13,5 cm (cao 11,2 cm).
  - Chữ trong bản sửa tay khớp 100% với BAO_CAO.md (trừ bìa + đoạn quy ước số đã bỏ).
- Chuẩn gốc: `~/Code/_NarBlack/CourseProj-Report_w_Pandoc` (styles/header-footer.tex: header trái = tên môn,
  footer trái = đề tài, footer phải = số trang, TOC "Nội dung báo cáo đồ án").

## Quyết định của chủ repo (2026-09-21)
- Nguồn dữ liệu: CHỈ bỏ tên cá nhân "Phạm Xuân Vĩnh Hà". Vẫn ghi gói ATSH-NLP-20k từ dự án ATSH-ABSA (UIT),
  nhãn silver do LLM gán. KHÔNG viết là chủ repo tự thu thập/gán nhãn.
- Cấp tiêu đề: mọi mục lớn (Tóm tắt, 1 đến 10, Phụ lục A) là H1; mục con 2.1... là H2.
- Slide: GIỮ Calibri (ý đồ thiết kế riêng). Skill ghi Times New Roman là mặc định cho bộ sau.
- Connector cho skill: "Claude trong PowerPoint/Word" (add-in Office). Skill tự dò công cụ; không có thì báo.
- Được phép mở Word/PowerPoint bằng AppleScript CHO ĐỢT NÀY để cập nhật mục lục và xuất PDF kiểm tra.
- Được sửa bài nói ngoài repo `/Volumes/USR_18T_01/UIT_eLearning/CS221_Slide-speaking.md` và xuất lại
  `report/CS221_Slide-speaking.pdf`.
- Được thêm mục ngắn vào CLAUDE.md máy (`~/nar-core-space/.claude/machine/CLAUDE.md`), chỉ commit phần mình thêm
  (file đang có hunk Co-Authored-By chưa commit của chủ repo, KHÔNG commit hunk đó).
- Skill đặt ở `~/nar-core-space/.claude/skills/`, symlink vào `~/.claude/skills/` như bộ agents.
- `review-2026-09-19/` giữ nguyên (thỏa thuận cũ). Lịch sử git vẫn còn tên, không viết lại lịch sử.
- Commit: KHÔNG thêm trailer Co-Authored-By.

## Checklist
### A. Bỏ tên tác giả dữ liệu (repo CS221)
- [x] data/README_ATSH_NLP_20k_goc.md: bỏ tên, giữ "dự án ATSH-ABSA (UIT, ĐHQG TP.HCM)"
- [x] data/README.md: bỏ tên ở bảng nguồn và câu trích dẫn; bỏ chữ "nguyên văn" (bản sao đã lược tên)
- [x] report/BAO_CAO.md: dòng trích dẫn, bảng nguồn, tài liệu tham khảo 15, chữ "nguyên văn"
- [x] report/slide.md: dòng nguồn, ghi chú, tài liệu tham khảo 8
- [x] scripts/build_slides_pptx.py: dòng nguồn, ghi chú, tài liệu tham khảo
- [x] Bài nói ngoài repo dòng 40, xuất lại report/CS221_Slide-speaking.pdf
- [x] Grep toàn repo (trừ .venv, .git, review-2026-09-19) sạch tên
- [x] Commit A

- Ghi chú A: README gốc thêm dòng "đã lược tên cá nhân"; bỏ chữ "nguyên văn". Bài nói còn ghi "Nhóm em có bảy bạn" (dòng 18), chưa sửa vì ngoài phạm vi, báo chủ repo. PDF bài nói trên ổ ngoài chưa xuất lại.
- Commit 5950ba0: lưu bản Word chủ repo sửa tay.

### B. Bộ dựng báo cáo Word
- [x] BAO_CAO.md: front matter (môn, đề tài, số trang bắt đầu...), bỏ bìa và đoạn quy ước số, hạ cấp tiêu đề (## -> #, ### -> ##)
- [x] build_report.py: một font Times New Roman (styles, docDefaults, theme, header/footer, mã)
- [x] build_report.py: trang mục lục đầu, header/footer, pgNumType start=2
- [x] build_report.py: heading 20/16/14pt, thân 12pt, bảng 12pt rộng hết khổ, hình vừa khung 16 x 11,5 cm
- [x] Dựng lại BAO_CAO.docx, cập nhật mục lục bằng Word, xuất PDF, soi từng trang
- [x] Script kiểm tra docx (font, header/footer, mục lục, kích thước hình/bảng) chạy sạch
- Ghi chú B: Commit A = cfacff8. Word sandbox: file phải đặt trong ~/Library/Containers/com.microsoft.Word/Data/tmp (office_refresh.py tự làm), nếu không Word hiện hộp "Grant File Access" và AppleScript treo. Bảng ngắn (<=15 dòng) giữ trọn một trang; hàng không bị cắt. PDF có ArialMT chỉ ở khoảng trắng sau số thứ tự danh sách (Word tự vẽ), không phải chữ hiển thị.
- [x] Commit B (e689b7b)

### C. Slide
- [x] Sơ đồ slide 3 và 8: bỏ tiêu đề nướng trong ảnh, đặt tiêu đề slide chuẩn, ảnh lấp đầy vùng nội dung
- [x] Ảnh/bảng các slide khác lấp tối đa vùng nội dung, không lấn vùng tiêu đề và số trang
- [x] Đổi --out mặc định sang tên file hiện tại `report/CS221_Xử-Lý-NNTN_slide.pptx`
- [x] Dựng lại pptx, xuất PDF bằng PowerPoint, soi từng slide, chạy script kiểm tra vùng
- [x] slide.md (Marp) đồng bộ nội dung đã đổi
- Ghi chú C: sơ đồ dùng bản _body.png (build_dark_assets.py --crop-only ghép khối + chú giải, bỏ tiêu đề và dải trống); fit_picture() đặt ảnh vừa vùng nội dung y 1,98 đến 6,84 in. check_pptx.py (skill slide-deck) 0 lỗi, 0 cảnh báo với --font Calibri --font "Calibri Light" --skip 1.
- [ ] Commit C

### D. Skill dùng chung (nar-core-space)
- [ ] Skill `word-report`: SKILL.md + references/format-contract.md + scripts/build_docx.py + scripts/check_docx.py
- [ ] Skill `slide-deck`: SKILL.md + references/layout-contract.md + scripts/pptx_zones.py + scripts/check_pptx.py
- [ ] Mục connector "Claude trong PowerPoint/Word" trong cả hai skill
- [ ] USAGE.md thêm hai mục; knowledge/log.md dòng schema-change
- [ ] Symlink hai skill vào ~/.claude/skills/
- [ ] CLAUDE.md máy: mục ngắn trong Coursework (commit riêng hunk của mình)
- [ ] Commit D (nar-core-space)

### E. Kết thúc
- [ ] Memory dự án: quyết định nguồn dữ liệu + con trỏ skill
- [ ] Báo cáo chủ repo
