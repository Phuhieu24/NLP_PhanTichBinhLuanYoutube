# Bìa CS221 trong repo Pandoc và skill tạo bìa dùng lại

Ngày bắt đầu: 2026-09-21

## Bối cảnh cần cho phiên mới
- Repo Pandoc: `~/Code/_NarBlack/CourseProj-Report_w_Pandoc`. Chủ repo đang có thay đổi chưa commit ở
  `build.sh`, `configs/AI-*.tex`, `configs/Math4CS.tex`, `styles/cover-body*.tex` và 4 file untracked
  (CS106, Design-and-Analysis). KHÔNG sửa, KHÔNG add các file đó.
- Hai mẫu bìa có sẵn: `styles/cover-body.tex` (không mã) và `styles/cover-body_w_group-and-topic.tex`
  (in cả mã nhóm lẫn mã đề tài). CS221 cần mẫu chỉ in mã nhóm.
- Bìa rời xuất bằng pandoc /dev/null + config + preamble + cover body (bước 1 của build.sh), không chạy build.sh.

## Quyết định của chủ repo (2026-09-21)
- Mã nhóm: "Nhóm 19". KHÔNG in mã đề tài (đề tài do sinh viên tự chọn). Mã môn CS221 không in lên bìa.
- Ngày nộp: 22/09/2026.
- Chỉ xuất bìa rời, không ghép với PDF báo cáo.
- Được commit trong repo Pandoc (chỉ file của mình); không push. CLAUDE.md của repo đó đã cũ ở điểm commit.
- Skill đặt ở nar-core-space, symlink vào ~/.claude/skills như word-report và slide-deck.

## Checklist
- [x] Mẫu bìa mới `styles/cover-body_flex.tex`: giảng viên, mã nhóm, mã đề tài, ngày nộp đều tùy chọn
- [x] `configs/CS221-XLNNTN.tex`
- [x] Xuất `output/course-reports/CS221-XLNNTN_course-report_cover.pdf`, soi hình, kiểm font
- [x] CLAUDE.md repo Pandoc: commit được phép, push thì không; thêm mẫu mới vào bản đồ file
- [x] REGISTRY.md repo Pandoc: thêm mục CS221
- [x] Commit repo Pandoc (chỉ file của mình)
- [x] Skill `uit-cover` (SKILL.md + scripts/make_cover.py) trong nar-core-space, chạy thử từ front matter
- [x] word-report: trỏ sang uit-cover; mẫu Markdown thêm khóa bìa
- [x] USAGE.md, log.md, symlink, commit nar-core-space
- [x] CS221: front matter BAO_CAO.md thêm khóa bìa, commit
- [x] Memory, báo cáo chủ repo

## Kết quả (2026-09-21, hoàn tất)
- Repo Pandoc: f7629a8 (styles/cover-body_flex.tex, configs/CS221-XLNNTN.tex,
  output/course-reports/CS221-XLNNTN_course-report_cover.pdf, CLAUDE.md, REGISTRY.md). Không push.
- nar-core-space: skill uit-cover (make_cover.py đọc front matter), word-report trỏ sang, symlink
  ~/.claude/skills/uit-cover.
- CS221: front matter BAO_CAO.md có khóa bìa (course_en, program, major, group_code, submission_date).
  Dựng lại bìa: `python3 ~/.claude/skills/uit-cover/scripts/make_cover.py --from-md report/BAO_CAO.md --name CS221-XLNNTN --force`
