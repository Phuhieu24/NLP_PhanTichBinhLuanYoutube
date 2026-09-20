# -*- coding: utf-8 -*-
"""Dựng bài thuyết trình PowerPoint (.pptx) từ nội dung của report/slide.md.

Chạy từ thư mục gốc của repo:

    .venv/bin/python scripts/build_slides_pptx.py

Script tự vẽ toàn bộ slide bằng python-pptx nên kiểm soát được màu, phông chữ, bảng và
ghi chú người nói. Phông Calibri Light cho tiêu đề và Calibri cho phần thân (hai phông này
có sẵn trong bộ Office trên cả macOS lẫn Windows và hiển thị đủ dấu tiếng Việt).

Ba bảng màu, chọn bằng --theme (xem THEMES bên dưới):

    slate   nền xám than #1E293B, chữ #F8FAFC, nhấn mòng két #2DD4BF   (mặc định)
    dark    nền xanh đen #0B1220, đậm nhất, cho phòng chiếu tối
    light   nền ngà #FAFAF9, chữ gần đen, nhấn mòng két đậm #0D9488

Mỗi theme kéo theo bộ ảnh có nền khớp (xem THEME_IMAGES); ảnh nền tối sinh bằng
scripts/build_dark_assets.py.

Nội dung trong file này được chép từ report/slide.md; khi sửa slide.md thì sửa cả ở đây
rồi chạy lại script.
"""
import sys, os, argparse
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

THEMES = {
    # Nền xám than: vẫn tối để chiếu, nhưng dịu hơn "dark" rõ rệt.
    "slate": dict(
        bg="1E293B", ink="F8FAFC", body="CBD5E1", muted="94A3B8", faint="8296B0",
        teal="2DD4BF", teal_dk="5EEAD4", teal_50="334155", teal_100="155E59",
        row_alt="2B3A50", white="243247", line="475569",
        amber="FBBF24", violet="A78BFA", rose="FB7185",
        neg="F87171", neu="9CA3AF", pos="4ADE80",
    ),
    # Nền xanh đen: bản tối đậm nhất.
    "dark": dict(
        bg="0B1220", ink="F2F6FB", body="C5D2E4", muted="93A6C0", faint="55688A",
        teal="2DD4BF", teal_dk="5EEAD4", teal_50="12303A", teal_100="164E4A",
        row_alt="121E30", white="0E1929", line="253750",
        amber="FBBF24", violet="A78BFA", rose="FB7185",
        neg="F87171", neu="9CA3AF", pos="4ADE80",
    ),
    # Nền ngà, chữ gần đen: để in hoặc chiếu trong phòng sáng.
    "light": dict(
        bg="FAFAF9", ink="0F172A", body="334155", muted="64748B", faint="94A3B8",
        teal="0D9488", teal_dk="0F766E", teal_50="F0FDFA", teal_100="CCFBF1",
        row_alt="F8FAFC", white="FFFFFF", line="E2E8F0",
        amber="D97706", violet="7C3AED", rose="E11D48",
        neg="EF4444", neu="9CA3AF", pos="22C55E",
    ),
}

_ap = argparse.ArgumentParser(description="Dựng bài thuyết trình PowerPoint của đồ án.")
_ap.add_argument("--theme", choices=sorted(THEMES), default="slate",
                 help="bảng màu: slate (mặc định), dark, light")
_ap.add_argument("--out", default="report/CS221_slide.pptx",
                 help="tệp .pptx xuất ra, tính từ thư mục gốc repo")
ARGS = _ap.parse_args()

C = THEMES[ARGS.theme]
def rgb(k): return RGBColor.from_string(C[k] if k in C else k)

TITLE_FONT = "Calibri Light"
BODY_FONT = "Calibri"

SW, SH = 13.333, 7.5
ML, MR = 0.80, 0.80
CW = SW - ML - MR
KICKER_TOP = 0.36
TITLE_TOP = 0.66
BAR_TOP = 1.60
CONTENT_TOP = 1.98

def set_bg(slide, color="bg"):
    f = slide.background.fill
    f.solid(); f.fore_color.rgb = rgb(color)

def textbox(slide, x, y, w, h):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tb, tf

def para(tf, first=False):
    return tf.paragraphs[0] if first else tf.add_paragraph()

def run(p, text, size, color, font=BODY_FONT, bold=False, italic=False):
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.name = font; r.font.bold = bold
    r.font.italic = italic; r.font.color.rgb = rgb(color)
    return r

def bullet(p, char="•", color="teal", indent=0.24):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(indent * 914400)))
    pPr.set("indent", str(-int(indent * 914400)))
    for tag, attrs in ((qn("a:buClr"), None), (qn("a:buFont"), {"typeface": "Arial"}),
                       (qn("a:buChar"), {"char": char})):
        el = pPr.makeelement(tag, attrs or {})
        if tag == qn("a:buClr"):
            el.append(pPr.makeelement(qn("a:srgbClr"), {"val": C[color]}))
        pPr.append(el)

def kicker(slide, text):
    if not text: return
    _, tf = textbox(slide, ML, KICKER_TOP, CW, 0.26)
    p = tf.paragraphs[0]
    run(p, text.upper(), 10.5, "teal", bold=True)
    p.runs[0].font._rPr.set("spc", "120")

def title(slide, text, size=29):
    _, tf = textbox(slide, ML, TITLE_TOP, CW, 0.95)
    p = tf.paragraphs[0]
    run(p, text, size, "ink", font=TITLE_FONT, bold=True)
    p.line_spacing = 0.95

def accent_bar(slide, y=BAR_TOP, w=0.85, color="teal"):
    from pptx.enum.shapes import MSO_SHAPE
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(ML), Inches(y), Inches(w), Inches(0.052))
    s.fill.solid(); s.fill.fore_color.rgb = rgb(color); s.line.fill.background()
    s.shadow.inherit = False
    return s

def page_number(slide, n):
    _, tf = textbox(slide, SW - MR - 1.0, SH - 0.62, 1.0, 0.3)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT
    run(p, str(n), 11, "faint")

def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text

def bullets(slide, items, x=ML, y=CONTENT_TOP, w=CW, size=15.5, gap=7, color="body", lead=1.18):
    _, tf = textbox(slide, x, y, w, SH - y - 0.7)
    for i, it in enumerate(items):
        p = para(tf, first=(i == 0))
        p.space_after = Pt(gap); p.line_spacing = lead
        bullet(p)
        if isinstance(it, str): it = [(it, {})]
        for seg, kw in it:
            run(p, seg, size, kw.get("color", color), font=kw.get("font", BODY_FONT),
                bold=kw.get("bold", False), italic=kw.get("italic", False))
    return tf

def caption(slide, text, y, size=11.5, align=PP_ALIGN.CENTER, color="muted"):
    _, tf = textbox(slide, ML, y, CW, 0.5)
    p = tf.paragraphs[0]; p.alignment = align; p.line_spacing = 1.15
    run(p, text, size, color)

def plain_table(table):
    tblPr = table._tbl.tblPr
    for el in tblPr.findall(qn("a:tableStyleId")):
        tblPr.remove(el)
    sid = tblPr.makeelement(qn("a:tableStyleId"), {})
    sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"
    tblPr.append(sid)
    table.first_row = False
    table.horz_banding = False

def fill_cell(cell, color):
    f = cell.fill; f.solid(); f.fore_color.rgb = rgb(color)

def cell_text(cell, text, size, color, bold=False, align=PP_ALIGN.LEFT, font=BODY_FONT):
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.margin_left = Inches(0.11); cell.margin_right = Inches(0.11)
    cell.margin_top = Inches(0.04); cell.margin_bottom = Inches(0.04)
    tf = cell.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = align
    for r in list(p.runs): r._r.getparent().remove(r._r)
    run(p, text, size, color, font=font, bold=bold)

def table(slide, rows, x, y, w, col_w, size=12.5, header_size=None, row_h=0.34,
          align_right_from=None, highlight_row=None):
    """rows[0] is the header. col_w: list of inches summing to w."""
    n, m = len(rows), len(rows[0])
    gf = slide.shapes.add_table(n, m, Inches(x), Inches(y), Inches(w), Inches(row_h * n))
    t = gf.table
    plain_table(t)
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i in range(n):
        t.rows[i].height = Inches(row_h if i else row_h + 0.04)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            ar = align_right_from is not None and j >= align_right_from
            al = PP_ALIGN.RIGHT if (ar and i > 0) else (PP_ALIGN.RIGHT if (ar and i == 0) else PP_ALIGN.LEFT)
            if i == 0:
                fill_cell(cell, "teal_50")
                cell_text(cell, val, header_size or size, "teal_dk", bold=True, align=al)
            else:
                hi = highlight_row is not None and i == highlight_row
                fill_cell(cell, "teal_50" if hi else ("white" if i % 2 else "row_alt"))
                cell_text(cell, val, size, "ink" if hi else "body", bold=hi, align=al)
    return t

def picture(slide, path, x=None, y=CONTENT_TOP, w=None, h=None):
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    pic = slide.shapes.add_picture(path, Inches(x if x is not None else ML), Inches(y), **kw)
    if x is None:
        pic.left = Emu(int((Inches(SW) - pic.width)))
    return pic




REPO = Path(__file__).resolve().parents[1]
# Hai sơ đồ phủ kín khung hình nên nền của chúng thay luôn nền slide; riêng ma trận nhầm
# lẫn đặt lọt trong slide nên nền ảnh phải khớp nền theme, nếu không sẽ thấy vệt chữ nhật.
THEME_IMAGES = {
    "slate": dict(pipeline="docs/diagrams/pipeline_slate.png",
                  evaluation="docs/diagrams/evaluation_slate.png",
                  cm="docs/diagrams/confusion_slate.png"),
    "dark": dict(pipeline="docs/diagrams/pipeline_dark.png",
                 evaluation="docs/diagrams/evaluation_dark.png",
                 cm="docs/diagrams/confusion_dark.png"),
    "light": dict(pipeline="docs/diagrams/pipeline.png",
                  evaluation="docs/diagrams/evaluation.png",
                  cm="results/confusion_matrix_normalized.png"),
}
IMG = {k: str(REPO / v) for k, v in THEME_IMAGES[ARGS.theme].items()}
IMG["app"] = str(REPO / "docs/screenshots/03_tab_tong_quan.png")
for k, v in IMG.items():
    assert os.path.exists(v), v

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(SW), Inches(SH)
BLANK = prs.slide_layouts[6]
n = 0

def new(kick=None, ttl=None, tsize=29, number=True):
    global n
    s = prs.slides.add_slide(BLANK)
    set_bg(s)
    n += 1
    if kick: kicker(s, kick)
    if ttl:
        title(s, ttl, tsize)
        accent_bar(s)
    if number and n > 1: page_number(s, n)
    return s

def full_bleed(path):
    """Sơ đồ phủ kín khung hình: hình đã có tiêu đề riêng nên slide không cần tiêu đề."""
    global n
    s = prs.slides.add_slide(BLANK)
    set_bg(s)
    n += 1
    s.shapes.add_picture(path, 0, 0, width=Inches(SW), height=Inches(SH))
    page_number(s, n)
    return s

# ---------------------------------------------------------------- 1. Bìa
s = new(number=False)
from pptx.enum.shapes import MSO_SHAPE
band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.42), Inches(SH))
band.fill.solid(); band.fill.fore_color.rgb = rgb("teal_100"); band.line.fill.background()
band.shadow.inherit = False
_, tf = textbox(s, 1.35, 1.95, 10.6, 1.9)
p = tf.paragraphs[0]; p.line_spacing = 1.02
run(p, "Phân tích chủ đề và cảm xúc\nbình luận YouTube tiếng Việt", 40, "ink", font=TITLE_FONT, bold=True)
b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.35), Inches(3.98), Inches(1.15), Inches(0.06))
b.fill.solid(); b.fill.fore_color.rgb = rgb("teal"); b.line.fill.background(); b.shadow.inherit = False
_, tf = textbox(s, 1.35, 4.28, 10.6, 0.4)
run(tf.paragraphs[0], "Đồ án môn Xử lý ngôn ngữ tự nhiên", 17, "teal_dk")
# Giảng viên đứng riêng một dòng, chữ sáng hơn danh sách thành viên để đọc được từ cuối phòng.
_, tf = textbox(s, 1.35, 4.76, 10.6, 0.4)
p = tf.paragraphs[0]
run(p, "Giảng viên hướng dẫn:  ", 15, "muted")
run(p, "TS. Đặng Văn Thìn", 15, "ink", bold=True)
_, tf = textbox(s, 1.35, 5.34, 10.6, 0.3)
p = tf.paragraphs[0]
r = run(p, "NHÓM THỰC HIỆN", 10, "teal", bold=True)
r.font._rPr.set("spc", "120")
members = ["Lê Phú Hiếu · 26410038 · LT.K2026.1.TTNT", "Nguyễn Thanh Duy · 26410030 · LT.K2026.1.TTNT",
           "Nguyễn Thanh Phong · 26410090 · LT.K2026.1.TTNT", "Nguyễn Thị Mai Thi · 26410117 · LT.K2026.1.TTNT",
           "Hồ Viết Trịnh · 26410140 · LT.K2026.1.TTNT"]
_, tf = textbox(s, 1.35, 5.70, 5.4, 1.2)
for i, m in enumerate(members[:3]):
    p = para(tf, first=(i == 0)); p.space_after = Pt(4); run(p, m, 11.5, "muted")
_, tf = textbox(s, 6.95, 5.70, 5.4, 1.2)
for i, m in enumerate(members[3:]):
    p = para(tf, first=(i == 0)); p.space_after = Pt(4); run(p, m, 11.5, "muted")
notes(s, "Em kính chào thầy Đặng Văn Thìn và các bạn. Nhóm em gồm năm thành viên, hôm nay xin trình bày đồ án phân tích bình luận YouTube tiếng Việt. Hệ thống nhận bình luận dưới một video và trả về hai thứ: người xem đang bàn chủ đề gì, và họ khen hay chê. Bài nói khoảng 10 phút theo đúng năm bước của môn, sau đó nhóm em chạy ứng dụng trực tiếp 3 phút. Nếu chỉ nhớ một câu, xin thầy nhớ câu này: mọi con số trong bài đều đọc lại được từ thư mục results của mã nguồn.")

# ---------------------------------------------------------------- 2. Bài toán
s = new("Bài toán", "Hàng nghìn bình luận, hai câu hỏi")
bullets(s, [
    [("Đầu vào: ", {"bold": True, "color": "ink"}), ("toàn bộ bình luận dưới một video YouTube tiếng Việt", {})],
    [("Đầu ra 1: ", {"bold": True, "color": "ink"}), ("các chủ đề người xem bàn tới, mỗi chủ đề là một nhóm từ khóa và vài bình luận tiêu biểu", {})],
    [("Đầu ra 2: ", {"bold": True, "color": "ink"}), ("phân bố cảm xúc theo ba lớp tiêu cực, trung tính, tích cực", {})],
    [("Vì sao khó: ", {"bold": True, "color": "ink"}), ("teencode (ko, đc, j), không dấu, emoji trong 34.5% bình luận, câu ngắn (trung vị 52 ký tự), châm biếm và vừa khen vừa chê", {})],
    [("Tiếng Việt không có khoảng trắng giữa các từ, nên tách từ quyết định chất lượng cả hai đầu ra", {})],
], size=18, gap=22)
notes(s, "Bài toán xuất phát từ nhu cầu thật: một video có vài nghìn bình luận, không ai đọc nổi từng dòng. Nhóm em muốn trả lời hai câu: người xem đang nói về cái gì, và họ thấy thế nào. Dữ liệu này khó hơn văn bản báo chí. Một phần ba bình luận có emoji. Trung vị chỉ 52 ký tự, tức là một câu ngắn, ít ngữ cảnh. Và rất nhiều câu kiểu \"hay mà tiếc\", vừa khen vừa chê. Thêm cái khó riêng của tiếng Việt: từ ghép không có ranh giới, nên tách từ sai thì từ khóa chủ đề sai và đặc trưng phân loại cũng sai.")

# ---------------------------------------------------------------- 3. Pipeline
s = full_bleed(IMG["pipeline"])
notes(s, "Nhóm em xếp đồ án theo đúng quy trình năm bước của bài 5 trong môn: thu thập và phân tích dữ liệu, tiền xử lý, biểu diễn, thuật toán, rồi đánh giá và phân tích lỗi. Sơ đồ này cho thấy mỗi bước nằm ở module nào trong mã nguồn. Điểm khác so với bài phân loại thuần: từ bước biểu diễn, pipeline tách làm hai nhánh. Nhánh cảm xúc dùng TF-IDF và LinearSVC. Nhánh chủ đề dùng vector câu và BERTopic. Hai nhánh gặp lại nhau ở ứng dụng, khi cảm xúc được hiển thị theo từng chủ đề. Phần còn lại của bài đi lần lượt qua năm bước này.")

# ---------------------------------------------------------------- 4. Dữ liệu
s = new("Bước 1 · Dữ liệu", "20.000 bình luận có nhãn về một chương trình")
table(s, [["Nhãn", "Số dòng", "Tỉ lệ"],
          ["Tiêu cực (0)", "7.000", "35,0%"],
          ["Trung tính (1)", "3.409", "17,0%"],
          ["Tích cực (2)", "9.591", "48,0%"],
          ["Tổng, sau tiền xử lý", "19.997", "3 dòng rỗng"]],
      x=ML, y=2.05, w=5.5, col_w=[2.5, 1.5, 1.5], size=15, row_h=0.60, align_right_from=1)
bullets(s, [
    "Gói ATSH-NLP-20k, dự án ATSH-ABSA (Phạm Xuân Vĩnh Hà, UIT); bình luận về Anh Trai Say Hi mùa 1, tập 1 đến 14",
    "Nhãn silver do mô hình ngôn ngữ lớn gán theo đối tượng và khía cạnh rồi gộp; chưa kiểm tay, trung tính nhiễu nhất",
    "Tác giả lấy mẫu lại từ dữ liệu gốc khoảng 88% tích cực; tỉ lệ nhãn không phải tỉ lệ thật trên YouTube",
    "Một chương trình duy nhất; chỉ dùng cho học tập, không công bố lại",
], x=6.85, y=2.05, w=5.7, size=17.5, gap=20)
notes(s, "Tập dữ liệu có 20.000 bình luận, hai cột: văn bản và nhãn. Đây là hai cột text và label của gói ATSH-NLP-20k, trích từ dự án ATSH-ABSA của Phạm Xuân Vĩnh Hà ở UIT; nhóm em chỉ dùng cho học tập theo đúng điều kiện của tác giả. Ba lớp lệch nhau: tích cực gần một nửa, trung tính chỉ 17%. Con số 17% này sẽ quay lại ở phần kết quả, vì trung tính là lớp yếu nhất. Ba điều nhóm em nói thẳng. Thứ nhất, mọi bình luận đều về một chương trình, nên mô hình học cả tên thí sinh làm tín hiệu. Thứ hai, tỉ lệ ba lớp là do tác giả lấy mẫu lại: dữ liệu gốc khoảng 88% tích cực, tác giả giữ hết trung tính, lấy đúng 7.000 tiêu cực rồi bù tích cực cho đủ 20.000; vì vậy số 7.000 tròn, và tỉ lệ này không phải tỉ lệ thật trên YouTube. Thứ ba, nhãn là nhãn silver do mô hình ngôn ngữ lớn gán theo từng đối tượng và khía cạnh rồi gộp lại, chưa có người kiểm từng dòng; chính tác giả ghi trung tính là lớp nhiễu nhất. Phần đọc tay 150 bình luận ở cuối bài cũng là để kiểm chất lượng nhãn.")

# ---------------------------------------------------------------- 5. Tiền xử lý
s = new("Bước 2 · Tiền xử lý", "Tách từ trước, hạ chữ thường sau")
bullets(s, [
    [("Thứ tự trong mã: ", {"bold": True, "color": "ink"}), ("chuẩn hóa NFC, bỏ URL, bỏ ký hiệu @ và #, thay dấu câu và emoji bằng khoảng trắng, rút chữ lặp (luônnnn thành luôn), chuẩn hóa teencode, tách từ bằng pyvi, cuối cùng mới hạ chữ thường", {})],
    [("Vì sao thứ tự đó: ", {"bold": True, "color": "ink"}), ("pyvi phân biệt chữ hoa khi ghép tên riêng. ", {}), ("Đông Hùng hát", {"italic": True, "color": "teal_dk"}), (" cho ", {}), ("Đông_Hùng hát", {"italic": True, "color": "teal_dk"}), ("; đã hạ chữ thường thì thành ba token rời", {})],
    [("Từ điển teencode chỉ sửa chính tả ", {}), ("(ko, hok, khong thành không)", {"italic": True}), (", không dịch tiếng lóng sang từ cảm xúc. Bản cũ có ok thành tốt, vcl thành rất: đó là gán nhãn trước khi mô hình học", {})],
    [("236 từ dừng chỉ dùng cho từ khóa chủ đề, vì ", {}), ("không", {"italic": True, "color": "teal_dk"}), (" nằm trong danh sách và là đặc trưng mạnh nhất của lớp tiêu cực (+2,93)", {})],
], size=17, gap=20)
notes(s, "Bước tiền xử lý có ba quyết định. Một, tách từ trước rồi mới hạ chữ thường. Nhóm em kiểm tra trực tiếp: đưa \"Đông Hùng hát\" vào pyvi thì được tên riêng một token; đưa bản đã hạ chữ thường thì tên bị tách làm hai. Phiên bản đầu của đồ án làm ngược thứ tự, nên tên riêng chưa bao giờ được ghép. Hai, từ điển teencode. Bản cũ dịch \"vcl\" thành \"rất\", \"ok\" thành \"tốt\". Nghe tiện, nhưng thực ra là mình gán sắc thái cho dữ liệu trước khi mô hình được học. Nhóm em bỏ hết, chỉ giữ sửa chính tả, để bộ phân loại tự học từ 20.000 mẫu. Ba, từ dừng. Danh sách có chữ \"không\". Nếu loại từ dừng khỏi bộ phân loại cảm xúc thì mất luôn tín hiệu phủ định, nên danh sách này chỉ đi vào bước chọn từ khóa chủ đề.")

# ---------------------------------------------------------------- 6. A/B tiền xử lý
s = new("Bước 2 · Thí nghiệm", "Đổi tiền xử lý không đổi điểm cảm xúc, nhưng đổi từ khóa chủ đề")
table(s, [["Tiền xử lý", "C", "Tỉ lệ dự đoán đúng", "Macro-F1", "F1 trung tính"],
          ["Cũ", "0,3", "77,58% ± 0,47", "0,7199 ± 0,0052", "0,5218"],
          ["Mới", "0,3", "77,58% ± 0,50", "0,7200 ± 0,0041", "0,5222"],
          ["Cũ", "1,0", "76,61% ± 0,40", "0,7079 ± 0,0052", "0,5004"],
          ["Mới", "1,0", "76,57% ± 0,43", "0,7075 ± 0,0053", "0,5007"]],
      x=ML, y=2.05, w=CW, col_w=[2.0, 0.9, 3.1, 3.1, 2.673], size=13.5, row_h=0.46, align_right_from=2)
bullets(s, [
    "Cùng TF-IDF và LinearSVC, năm hạt giống chia tập (0 đến 4), trên bình luận của tập dữ liệu",
    "Mọi chênh lệch nằm trong một độ lệch chuẩn: với bộ phân loại, thay đổi là trung tính",
    "Giữ bản mới vì tên riêng thành một token trong từ khóa chủ đề và từ điển không áp sắc thái lên dữ liệu",
], y=4.78, size=16, gap=14)
notes(s, "Bài 5 của môn có cặp thí nghiệm có tách từ và không tách từ để xem tiền xử lý đổi kết quả thế nào. Nhóm em làm cặp thí nghiệm tương tự cho hai phiên bản tiền xử lý của mình. Kết quả là bảng này. Mời thầy nhìn hai dòng đầu: 77,58% và 77,58%, giống nhau đến hai chữ số. Nói thẳng: với bộ phân loại cảm xúc, đổi tiền xử lý không tăng điểm. Nhóm em vẫn giữ bản mới, nhưng vì lý do khác: từ khóa chủ đề có tên riêng đúng, và từ điển không còn tự gán cảm xúc. Điểm tăng thật sự ở đồ án đến từ chỗ khác, em sẽ nói ở phần mô hình.")

# ---------------------------------------------------------------- 7. Biểu diễn
s = new("Bước 3 · Biểu diễn", "TF-IDF cho cảm xúc, vector câu 768 chiều cho chủ đề")
bullets(s, [
    [("Cảm xúc: ", {"bold": True, "color": "ink"}), ("TF-IDF unigram và bigram, 15.000 đặc trưng, sublinear_tf", {})],
    [("Bigram là bắt buộc vì phủ định đứng trước từ bị phủ định: ", {}), ("không hay", {"italic": True, "color": "neg"}), (", ", {}), ("không thích", {"italic": True, "color": "neg"}), (" là đặc trưng của lớp tiêu cực; ", {}), ("hay mà", {"italic": True, "color": "muted"}), (", ", {}), ("hay nhưng", {"italic": True, "color": "muted"}), (" là đặc trưng của lớp trung tính", {})],
    [("Chủ đề: ", {"bold": True, "color": "ink"}), ("mỗi bình luận đã tách từ thành một vector 768 chiều từ keepitreal/vietnamese-sbert (Sentence-BERT tiếng Việt; theo config.json trên Hugging Face, mô hình được tinh chỉnh từ PhoBERT-base, nên cần tách từ trước)", {})],
    [("Từ khóa của mỗi cụm chọn bằng c-TF-IDF: từ nào nhiều trong cụm này, ít ở cụm khác thì lên đầu (ví dụ số ở phụ lục)", {})],
], size=17, gap=20)
notes(s, "Hai bài toán con cần hai cách biểu diễn. Với cảm xúc, nhóm em dùng TF-IDF có bigram. Nếu chỉ dùng unigram, chữ \"hay\" sẽ kéo cả \"không hay\" lẫn \"hay nhưng\" về lớp tích cực. Bigram giữ được cặp phủ định. Với chủ đề, mỗi bình luận thành một vector 768 chiều từ một mô hình Sentence-BERT tiếng Việt. Thẻ mô hình không ghi mô hình gốc, nhóm em đọc file config trên Hugging Face và suy ra nó tinh chỉnh từ PhoBERT-base. Điều đó có hệ quả thực tế: PhoBERT học trên văn bản đã tách từ, nên bước pyvi phía trước vừa phục vụ từ khóa, vừa là định dạng đầu vào mà mô hình nhúng mong đợi.")

# ---------------------------------------------------------------- 8. Quy trình đánh giá
s = full_bleed(IMG["evaluation"])
notes(s, "Trước khi xem con số, em nói cách chấm. Dữ liệu chia phân tầng 80 trên 20 với hạt giống 42. Mọi việc chọn lựa, so bốn mô hình nền và dò tham số C, chỉ chạy bằng cross-validation năm phần trên 15.997 dòng huấn luyện. Tập kiểm tra 4.000 dòng để dành, chấm đúng một lần với mô hình cuối. Sau đó nhóm em chia lại với năm hạt giống khác để xem con số có ổn định không. Quy trình này là lý do nhóm em tin các số ở hai slide sau.")

# ---------------------------------------------------------------- 9. Mô hình cảm xúc
s = new("Bước 4 · Mô hình", "Bốn mô hình nền, dò C, chỉ trên tập huấn luyện")
table(s, [["Mô hình", "Tỉ lệ dự đoán đúng (CV)", "Macro-F1 (CV)"],
          ["MostFrequent (đoán lớp đa số)", "47,95%", "0,2161"],
          ["MultinomialNB", "73,82%", "0,5620"],
          ["LogisticRegression", "76,25%", "0,7170 ± 0,0054"],
          ["LinearSVC, C = 1", "76,56%", "0,7054 ± 0,0095"],
          ["LinearSVC, C = 0,3 (đã dò)", "77,49%", "0,7180 ± 0,0066"]],
      x=ML, y=2.02, w=9.6, col_w=[4.2, 2.7, 2.7], size=13.5, row_h=0.44, align_right_from=1, highlight_row=5)
bullets(s, [
    "Cross-validation 5-fold phân tầng trên 15.997 dòng huấn luyện, class_weight balanced",
    "Lưới C theo macro-F1: 0,1 cho 0,7124 · 0,3 cho 0,7180 · 1,0 cho 0,7054 · 3,0 cho 0,6875",
    "LinearSVC hơn LogisticRegression 0,0010, nhỏ hơn độ lệch chuẩn giữa các fold: hai mô hình ngang nhau",
], y=4.98, size=16, gap=14)
notes(s, "Nhóm em so bốn mô hình nền bằng cross-validation, chỉ trên tập huấn luyện, tập kiểm tra để dành. Dòng đầu là mốc sàn: đoán toàn lớp đa số được 47,95% nhưng macro-F1 chỉ 0,22, nên nhóm em không dùng accuracy làm chỉ tiêu chính. Hai mô hình tuyến tính vượt Naive Bayes rõ. Rồi nhóm em dò C cho LinearSVC trên bốn giá trị: từ C bằng 1 xuống 0,3, macro-F1 tăng từ 0,7054 lên 0,7180. Đây là phần tăng đo được của đồ án, không phải tiền xử lý. Còn LinearSVC so với hồi quy logistic: chênh một phần nghìn, nhỏ hơn nhiễu giữa các fold. Nhóm em giữ LinearSVC theo thiết kế ban đầu, không phải vì nó thắng.")

# ---------------------------------------------------------------- 10. Kết quả test
s = new("Bước 5 · Kết quả", "Macro-F1 0,7161; lớp trung tính là điểm yếu")
table(s, [["Lớp", "Độ chính xác", "Độ phủ", "F1"],
          ["Tiêu cực (1.400)", "0,7458", "0,7564", "0,7511"],
          ["Trung tính (682)", "0,5270", "0,5440", "0,5354"],
          ["Tích cực (1.918)", "0,8715", "0,8525", "0,8619"]],
      x=ML, y=2.08, w=5.6, col_w=[2.2, 1.25, 1.1, 1.05], size=14.5, row_h=0.56, align_right_from=1, highlight_row=2)
picture(s, IMG["cm"], x=6.68, y=1.98, w=5.75)
bullets(s, [
    "Tỉ lệ dự đoán đúng 76,62% và macro-F1 0,7161 trên 4.000 dòng, chấm một lần; độ chính xác là precision, độ phủ là recall",
    "Năm hạt giống 0 đến 4: 77,58% ± 0,44 và macro-F1 0,7200 ± 0,0037",
    "Trung tính: đúng 371/682; 203 bị gán tiêu cực, 108 bị gán tích cực",
], x=ML, y=4.52, w=5.95, size=15, gap=16)
notes(s, "Đây là kết quả trên tập kiểm tra, chấm đúng một lần với mô hình đã chọn. Tỉ lệ dự đoán đúng 76,62%, macro-F1 0,7161. Chạy lại với năm hạt giống chia tập khác thì được 77,58%, nên con số 76,62% là ước lượng thận trọng. Mời thầy nhìn hàng giữa của ma trận nhầm lẫn bên phải: lớp trung tính chỉ nhận ra 54%, còn lại chia đều về hai phía, 203 sang tiêu cực, 108 sang tích cực. Hai lớp tiêu cực và tích cực hiếm khi nhầm sang nhau, chỉ 133 và 158 trên gần 3.300 mẫu. Nói gọn: mô hình phân biệt khen với chê tốt, nhưng không chắc đâu là \"không khen không chê\".")

# ---------------------------------------------------------------- 11. Chủ đề
s = new("Bước 4 · Chủ đề", "Tham số HDBSCAN quyết định hơn cả từ dừng")
table(s, [["Chọn cụm", "Cụm tối thiểu", "min_samples", "Chủ đề", "Nhiễu (-1)", "Cụm lớn nhất"],
          ["eom (mặc định BERTopic)", "10", "mặc định", "23", "735 (49,2%)", "81 (5,4%)"],
          ["eom", "20", "5", "3", "0 (0%)", "1.359 (91,0%)"],
          ["eom (chọn dùng)", "15", "1", "26", "497 (33,3%)", "117 (7,8%)"],
          ["leaf", "10", "mặc định", "27", "793 (53,1%)", "80 (5,4%)"],
          ["leaf", "20", "5", "19", "620 (41,5%)", "94 (6,3%)"]],
      x=ML, y=2.02, w=CW, col_w=[3.1, 1.75, 1.75, 1.35, 1.9, 1.923], size=13, row_h=0.43, align_right_from=1, highlight_row=3)
bullets(s, [
    "Khảo sát trên 1.493 bình luận của tập dữ liệu (1.500 dòng đầu), không phải bình luận của một video thật",
    "Mặc định đẩy gần nửa vào nhiễu; min_samples 5 gom 91% vào một cụm; 15/1 cân bằng nhất và là mặc định của ứng dụng",
    "Bật hay tắt từ dừng cho phép gán chủ đề giống hệt từng dòng, chỉ từ khóa đổi",
], y=4.88, size=15.5, gap=14)
notes(s, "Sang nhánh chủ đề. BERTopic gồm UMAP giảm 768 chiều xuống 5, HDBSCAN gom cụm theo mật độ, rồi c-TF-IDF chọn từ khóa. Tham số mặc định không dùng được cho bình luận cùng một chương trình: 49% bình luận bị xếp vào nhiễu. Đổi min_samples lên 5 thì ngược lại, 91% dồn vào một cụm, không có nhiễu, nhưng vô nghĩa. Cấu hình 15 và 1 nằm giữa: 26 chủ đề, một phần ba là nhiễu, và các bình luận nhiễu vẫn được giữ trong bảng dưới tên nhóm -1, không bị loại. Một phát hiện nhóm em thấy đáng nói: tắt từ dừng thì số chủ đề và tỉ lệ nhiễu không đổi một dòng nào, vì từ dừng chỉ đi vào bước chọn từ khóa, sau khi HDBSCAN đã gom xong. Xin nhấn mạnh là bảng này chạy trên bình luận trong tập dữ liệu; nó có chuyển sang một video thật hay không, nhóm em chưa đo.")

# ---------------------------------------------------------------- 12. Ứng dụng
s = new("Ứng dụng", "Năm tab, ba nguồn dữ liệu, không cần API key để chấm")
pic = picture(s, IMG["app"], x=6.10, y=2.02, w=6.43)
pic.line.color.rgb = rgb("line"); pic.line.width = Pt(0.75)
bullets(s, [
    [("Ba nguồn: ", {"bold": True, "color": "ink"}), ("Link YouTube (cần API key), Tệp CSV, Dữ liệu mẫu", {})],
    [("Năm tab: ", {"bold": True, "color": "ink"}), ("Tổng quan, Chủ đề, Cảm xúc, Dữ liệu, Mô hình", {})],
    [("Bình luận và vector nhúng được cache; đổi tab hay lọc bảng không chạy lại", {})],
    [("Ảnh: 1.000 dòng mẫu, 23 chủ đề, 20,6% nhiễu, 47,4% tích cực (tập huấn luyện, nên tỉ lệ cảm xúc lạc quan hơn thực tế)", {})],
], x=ML, y=2.08, w=5.05, size=16, gap=20)
notes(s, "Toàn bộ pipeline đóng thành một ứng dụng Streamlit. Bây giờ nhóm em chạy trực tiếp khoảng 3 phút. (mở ứng dụng, ở thanh bên chọn Dữ liệu mẫu, 1.000 dòng, hoặc Tệp CSV nếu nhóm đã cào sẵn bình luận của một video; bấm Bắt đầu phân tích) Trong lúc chạy em nói qua sáu bước của ứng dụng: lấy dữ liệu, làm sạch, nhúng câu, gom cụm, phân loại, tóm tắt tùy chọn. (khi xong, mở tab Tổng quan) Đây là số chủ đề, tỉ lệ nhiễu và tỉ lệ tích cực. (mở tab Chủ đề, chỉ vào một cụm) Mỗi chủ đề có từ khóa và bình luận tiêu biểu; bản đồ khoảng cách cho thấy cụm nào gần nhau. (mở tab Cảm xúc) Cảm xúc theo từng chủ đề: chủ đề nào bị chê nhiều nhất. (mở tab Dữ liệu, lọc một từ khóa) Bảng có lọc theo chủ đề, cảm xúc, từ khóa và tải CSV. Tab Mô hình chỉ đọc lại thẻ mô hình và các bảng em vừa trình bày, em không mở để tiết kiệm thời gian.")

# ---------------------------------------------------------------- 13. Phân tích lỗi
s = new("Bước 5 · Phân tích lỗi", "150 bình luận sai, xếp theo hiện tượng ngôn ngữ")
table(s, [["Hiện tượng", "n", "Tiêu cực", "Trung tính", "Tích cực"],
          ["Cần ngữ cảnh chương trình", "36", "15", "11", "10"],
          ["Thương cảm, tiếc nuối", "31", "11", "9", "11"],
          ["Vừa khen vừa chê", "23", "7", "12", "4"],
          ["Phủ định", "19", "6", "6", "7"],
          ["Teencode ngoài từ điển", "16", "5", "4", "7"],
          ["Tên riêng lấn át hoặc mất emoji", "11", "4", "4", "3"],
          ["Nhãn gốc đáng ngờ", "7", "3", "2", "2"],
          ["Châm biếm, không dấu, khác", "7", "4", "2", "1"]],
      x=ML, y=1.98, w=7.9, col_w=[3.5, 0.9, 1.2, 1.2, 1.1], size=14, row_h=0.50, align_right_from=1)
bullets(s, [
    "Mẫu phân tầng từ 935 dòng sai; một người đọc, một lượt",
    "Lớp trung tính: dẫn đầu là vừa khen vừa chê (12 trong 50 dòng), đúng với F1 0,5354",
    "Mô hình học cả chủ đề và tên riêng làm tín hiệu cảm xúc, đúng trong miền này nhưng không chuyển được sang video khác",
], x=8.95, y=1.98, w=3.6, size=15.5, gap=18)
notes(s, "Nhóm em không dừng ở ma trận nhầm lẫn. Nhóm xuất toàn bộ 935 bình luận mà mô hình đoán sai, lấy mẫu phân tầng 150 dòng rồi đọc tay từng dòng, mỗi dòng xếp vào một hiện tượng. Hai nhóm lớn nhất chiếm gần một nửa và cùng nói một điều: TF-IDF thấy từ nhưng không thấy thái độ hướng về ai. \"Tiếc cho team của Captain\" mang từ buồn nhưng là đồng cảm, không phải chê. Nhóm thứ ba là vừa khen vừa chê, và đây là nhóm lớn nhất của riêng lớp trung tính: một nhãn cho cả câu thì buộc phải chọn một bên, nên lớp trung tính khó ngay từ định nghĩa. Bảy dòng là nhãn gốc mà người đọc không đồng ý; con số này đo trên mẫu lỗi nên không suy ra được tỉ lệ nhãn sai của cả bộ dữ liệu. Về đặc trưng, mô hình học được rằng ai nhắc quảng cáo thì thường chê, còn nhắc tên hai thí sinh thì thường khen: đúng trong chương trình này, không chuyển được sang video khác.")

# ---------------------------------------------------------------- 14. Kết luận
s = new("Kết luận", "Pipeline chạy được, tái lập được, và biết mình yếu ở đâu")
bullets(s, [
    "Đủ năm bước của môn, mọi con số tái lập từ results/ bằng lệnh trong phụ lục báo cáo",
    "Macro-F1 0,72 trên tập kiểm tra và năm hạt giống; lớp trung tính (F1 0,5354) là giới hạn của cách tiếp cận TF-IDF tuyến tính",
    "Với chủ đề, tham số HDBSCAN quyết định kết quả nhiều hơn danh sách từ dừng",
    [("Tiếp theo: ", {"bold": True, "color": "ink"}), ("giữ emoji làm token và đo lại theo cùng quy trình năm hạt giống; dùng vector câu cho hồi quy logistic hoặc tinh chỉnh PhoBERT ba lớp; tính coherence c_v để chọn cấu hình chủ đề theo số đo thay vì cảm nhận", {})],
], size=17.5, gap=22)
_, tf = textbox(s, ML, 6.25, CW, 0.6)
p = tf.paragraphs[0]
run(p, "Số đẹp thì dễ, số đọc lại được mới khó.", 17, "teal_dk", font=TITLE_FONT, bold=True)
notes(s, "Ba điều nhóm em rút ra. Một, pipeline đi đủ năm bước và ai cũng chạy lại được: một lệnh huấn luyện 12 giây sinh lại toàn bộ kết quả. Hai, macro-F1 dừng ở 0,72, và cái kéo nó xuống là lớp trung tính; TF-IDF tuyến tính không đủ để hiểu \"hay mà tiếc\" là không khen không chê. Ba, ở phần chủ đề, thứ đáng dò là tham số gom cụm, không phải danh sách từ dừng. Ba việc tiếp theo xếp theo chi phí tăng dần: giữ emoji, thử vector câu cho phân loại, và đo coherence. Câu nhóm em muốn để lại: số đẹp thì dễ, số đọc lại được mới khó, và nhóm em chọn cái khó. Nhóm em xin cảm ơn thầy và các bạn.")

# ---------------------------------------------------------------- 15. Câu hỏi
s = new(number=True)
_, tf = textbox(s, ML, 2.75, CW, 1.0)
p = tf.paragraphs[0]
run(p, "Câu hỏi", 36, "ink", font=TITLE_FONT, bold=True)
b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(ML), Inches(3.72), Inches(1.15), Inches(0.06))
b.fill.solid(); b.fill.fore_color.rgb = rgb("teal"); b.line.fill.background(); b.shadow.inherit = False
_, tf = textbox(s, ML, 4.05, 9.5, 1.2)
p = tf.paragraphs[0]; p.space_after = Pt(9)
run(p, "Nhóm em xin cảm ơn thầy và các bạn.", 16, "body")
p = tf.add_paragraph(); p.line_spacing = 1.2
run(p, "Mã nguồn, 146 kiểm thử và toàn bộ tệp kết quả nằm trong repo nộp kèm báo cáo. "
       "Ba slide phụ lục phía sau: tài liệu tham khảo, ví dụ số c-TF-IDF, và một bẫy kỹ thuật của BERTopic.", 14, "muted")
page_number(s, 15)
notes(s, "Nhóm em xin dừng ở đây và sẵn sàng trả lời câu hỏi. (nếu được hỏi về c-TF-IDF hoặc về BERTopic với tiếng Việt, chuyển sang slide phụ lục tương ứng)")

# ---------------------------------------------------------------- 16. B1 references
s = new("Phụ lục B1", "Tài liệu tham khảo")
refs = ["Slide môn CS221 Xử lý ngôn ngữ tự nhiên, TS. Đặng Văn Thìn, UIT, bài 3, 4, 5, 6.",
        "Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794.",
        "McInnes, L., Healy, J., Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426.",
        "Campello, R. J. G. B., Moulavi, D., Sander, J. (2013). Density-Based Clustering Based on Hierarchical Density Estimates. PAKDD 2013, LNCS 7819.",
        "Nguyen, D. Q., Nguyen, A. T. (2020). PhoBERT: Pre-trained language models for Vietnamese. Findings of EMNLP 2020.",
        "Thẻ mô hình keepitreal/vietnamese-sbert, Hugging Face Hub.",
        "pyvi 0.1.1; scikit-learn 1.9.1 (Pedregosa và cộng sự, JMLR 2011); Streamlit 1.64.0; BERTopic 0.17.4; tài liệu YouTube Data API v3.",
        "Dữ liệu được cung cấp bởi dự án ATSH-ABSA (Phạm Xuân Vĩnh Hà, UIT), chỉ dùng cho mục đích học tập. Gói ATSH-NLP-20k."]
_, tf = textbox(s, ML, 1.98, CW, 5.0)
for i, r in enumerate(refs):
    p = para(tf, first=(i == 0)); p.space_after = Pt(13); p.line_spacing = 1.14
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(0.3 * 914400))); pPr.set("indent", str(-int(0.3 * 914400)))
    run(p, f"{i+1}.  ", 14, "teal", bold=True)
    run(p, r, 14, "body")
notes(s, "Danh sách tài liệu tham khảo, trùng với mục 10 của báo cáo. Slide bài giảng của môn là nguồn cho pipeline năm bước, cách tách từ và cách báo kết quả theo lớp; các bài báo là nguồn cho BERTopic, UMAP, HDBSCAN và PhoBERT; dòng cuối là câu trích dẫn mà tác giả bộ dữ liệu yêu cầu.")

# ---------------------------------------------------------------- 17. B2 c-TF-IDF
s = new("Phụ lục B2", "c-TF-IDF: vì sao “hay” không bao giờ đứng đầu từ khóa")
_, tf = textbox(s, ML, 1.98, CW, 0.4)
p = tf.paragraphs[0]
run(p, "W(t, c) = tf(t, c) × log(1 + A / f(t)),  với A là số từ trung bình của một cụm", 14.5, "teal_dk", italic=True)
table(s, [["Từ", "Cụm 1", "Cụm 2", "Cụm 3", "f(t)", "log(1 + 10/f(t))"],
          ["hát", "6", "0", "0", "6", "log(2,667) = 0,981"],
          ["hạng", "0", "6", "0", "6", "0,981"],
          ["quảng_cáo", "1", "1", "6", "8", "log(2,25) = 0,811"],
          ["hay", "3", "3", "4", "10", "log(2) = 0,693"]],
      x=ML, y=2.55, w=9.3, col_w=[2.0, 1.2, 1.2, 1.2, 1.1, 2.6], size=13.5, row_h=0.43, align_right_from=1, highlight_row=4)
bullets(s, [
    "Ba cụm, mỗi cụm 10 từ, nên A = 10; logarit tự nhiên",
    "Cụm 1: hát 6 × 0,981 = 5,88; hay 3 × 0,693 = 2,08; quảng_cáo 0,81. Cụm 3: quảng_cáo 4,87; hay 2,77",
    "hay có mặt ở cả ba cụm nên thành phần logarit thấp nhất, luôn đứng sau từ đặc trưng riêng. Hư từ như là, của, mà có tần suất thô quá cao nên vẫn cần danh sách từ dừng",
], y=5.02, size=15, gap=13)
notes(s, "Ví dụ số cho c-TF-IDF. BERTopic nối mọi bình luận của một cụm thành một văn bản rồi chấm từng từ: tần suất trong cụm nhân với logarit của một cộng A chia f. Từ \"hát\" chỉ có ở cụm 1 nên điểm 5,88, đứng đầu. Từ \"hay\" rải khắp ba cụm, thành phần logarit chỉ 0,693, nên dù xuất hiện nhiều vẫn đứng sau. Nhưng hư từ như \"là\", \"của\" thì tần suất thô lớn đến mức vẫn chiếm đầu bảng, nên danh sách từ dừng ở bước này vẫn cần.")

# ---------------------------------------------------------------- 18. B3 bẫy BERTopic
s = new("Phụ lục B3", "Bẫy BERTopic với tiếng Việt và cách tái lập kết quả", tsize=28)
bullets(s, [
    [("BERTopic() mặc định language=\"english\"; nếu không truyền embedding_model, bước làm sạch nội bộ xóa mọi ký tự ngoài [A-Za-z0-9 ] trước c-TF-IDF: ", {}), ("không", {"italic": True, "color": "rose"}), (" thành ", {}), ("khng", {"italic": True, "color": "rose"}), (", ", {}), ("chương_trình", {"italic": True, "color": "rose"}), (" thành ", {}), ("chngtrnh", {"italic": True, "color": "rose"})],
    "Pipeline của nhóm luôn truyền mô hình nhúng vào BERTopic, kể cả khi vector đã tính sẵn; một kiểm thử xác nhận language của mô hình là None",
    "Tái lập: python src/train_sentiment.py (khoảng 12 giây, sinh lại results/ và models/); python experiments/dataset_stats.py, ab_preprocess.py, topic_ablation.py cho ba bảng thực nghiệm; pytest chạy 146 kiểm thử trong khoảng 16 giây",
    "Lệnh gom cụm dòng lệnh trên 1.500 dòng đầu cho 1.493 bình luận hợp lệ, 26 chủ đề, 33,3% nhiễu; UMAP có hạt giống cố định nên cùng máy cho cùng kết quả",
], size=16.5, gap=20)
notes(s, "Một bài học kỹ thuật nhóm em xác minh trong mã nguồn thư viện. BERTopic khởi tạo với ngôn ngữ mặc định là tiếng Anh, và nếu mình không truyền mô hình nhúng, nó âm thầm xóa hết dấu tiếng Việt trước khi tính từ khóa. Chữ \"không\" thành \"khng\". Nhóm em vá bằng cách luôn truyền mô hình nhúng và viết một kiểm thử canh chỗ đó. Phần tái lập: một lệnh huấn luyện, ba script thực nghiệm, và pytest. Không lệnh nào cần API key hay mạng, trừ lần đầu tải mô hình nhúng về cache.")

out = str(REPO / ARGS.out)
prs.save(out)
print(f"Đã ghi {out}\n  theme {ARGS.theme} · {len(prs.slides._sldIdLst)} slide · {os.path.getsize(out)//1024} KB")
