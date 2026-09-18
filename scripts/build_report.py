"""Dựng file báo cáo Word (.docx) từ bản Markdown nguồn.

Chạy từ thư mục gốc của repo:

    .venv/bin/python scripts/build_report.py
    .venv/bin/python scripts/build_report.py --src report/BAO_CAO.md --out report/BAO_CAO.docx

Script chỉ hiểu một tập con Markdown đủ dùng cho báo cáo đồ án: tiêu đề `#`, `##`, `###`,
đoạn văn, in đậm, in nghiêng, mã lệnh trong nháy ngược, liên kết, danh sách gạch đầu dòng,
danh sách đánh số, bảng dạng ống, hình ảnh, trích dẫn `>`, khối mã ba nháy ngược và dòng
`\\newpage` để ngắt trang. Front matter YAML và chú thích HTML bị bỏ qua.

Thiếu file hình thì script vẫn chạy hết, chỗ thiếu được chèn một dòng chữ đỏ và tên file
được liệt kê lại ở cuối, mã thoát vẫn là 0 để khâu dựng báo cáo không bị gãy giữa chừng.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docx import Document  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.shared import Cm, Pt, RGBColor  # noqa: E402

DEFAULT_SRC = "report/BAO_CAO.md"
DEFAULT_OUT = "report/BAO_CAO.docx"

BASE_FONT = "Times New Roman"
BASE_SIZE = Pt(13)
CODE_FONT = "Consolas"
INLINE_CODE_SIZE = Pt(10)
BLOCK_CODE_SIZE = Pt(9)
TABLE_FONT_SIZE = Pt(9)
MAX_IMAGE_WIDTH = Cm(15.5)
CAPTION_PREFIX_TABLE = "Bảng"
CAPTION_MAX_CHARS = 120

PAGE_BREAK_TOKEN = "\\newpage"

# Chú thích HTML, kể cả loại nhiều dòng của Marp.
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$")
BULLET_RE = re.compile(r"^[-*]\s+(.*)$")
NUMBER_RE = re.compile(r"^\d+[.)]\s+(.*)$")
QUOTE_RE = re.compile(r"^>\s?(.*)$")
FENCE_RE = re.compile(r"^\s*```")
IMAGE_RE = re.compile(r"^!\[(?P<caption>[^\]]*)\]\((?P<path>[^)]+)\)\s*$")
RULE_RE = re.compile(r"^(\*{3,}|-{3,}|_{3,})$")
TABLE_SEP_RE = re.compile(r"^\|?[\s:|-]*-[\s:|-]*\|?$")

INLINE_RE = re.compile(
    r"(?P<code>`[^`]+`)"
    r"|(?P<link>\[(?P<ltext>[^\]]*)\]\((?P<lurl>[^)\s]+)\))"
    r"|(?P<bold>\*\*(?P<btext>.+?)\*\*)"
    r"|(?P<italic>\*(?P<itext>[^*]+)\*)"
)


@dataclass
class InlineRun:
    """Một đoạn chữ liền mạch cùng kiểu định dạng."""

    text: str
    bold: bool = False
    italic: bool = False
    code: bool = False


@dataclass
class Block:
    """Một khối nội dung đã tách khỏi Markdown, chờ đổ vào tài liệu Word."""

    kind: str
    lines: list[str] = field(default_factory=list)
    level: int = 0
    text: str = ""
    rows: list[list[str]] = field(default_factory=list)
    caption: str = ""
    path: str = ""


@dataclass
class BuildStats:
    """Số liệu tóm tắt in ra cuối lần dựng."""

    headings: int = 0
    paragraphs: int = 0
    tables: int = 0
    images: int = 0
    missing_images: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- tách khối


def strip_front_matter(text: str) -> str:
    """Bỏ khối front matter YAML nằm ngay đầu file, nếu có."""
    lines = text.splitlines()
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    if start >= len(lines) or lines[start].strip() != "---":
        return text
    for idx in range(start + 1, len(lines)):
        if lines[idx].strip() in {"---", "..."}:
            return "\n".join(lines[idx + 1 :])
    # Không tìm được dấu đóng thì coi như không có front matter.
    return text


def parse_blocks(text: str) -> list[Block]:
    """Đổi Markdown thành danh sách khối theo thứ tự xuất hiện."""
    text = strip_front_matter(text)
    text = COMMENT_RE.sub("", text)
    lines = text.replace("\r\n", "\n").split("\n")

    blocks: list[Block] = []
    buffer: list[str] = []

    def flush_paragraph() -> None:
        if buffer:
            blocks.append(Block(kind="paragraph", lines=list(buffer)))
            buffer.clear()

    idx = 0
    total = len(lines)
    while idx < total:
        raw = lines[idx]
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            idx += 1
            continue

        if stripped == PAGE_BREAK_TOKEN:
            flush_paragraph()
            blocks.append(Block(kind="pagebreak"))
            idx += 1
            continue

        if FENCE_RE.match(line):
            flush_paragraph()
            idx += 1
            code_lines: list[str] = []
            while idx < total and not FENCE_RE.match(lines[idx]):
                code_lines.append(lines[idx])
                idx += 1
            idx += 1  # bỏ dòng đóng khối
            blocks.append(Block(kind="code", lines=code_lines))
            continue

        heading = HEADING_RE.match(stripped)
        if heading:
            flush_paragraph()
            title = heading.group(2).strip().rstrip("#").strip()
            blocks.append(Block(kind="heading", level=len(heading.group(1)), text=title))
            idx += 1
            continue

        if RULE_RE.match(stripped):
            # Đường kẻ ngang không có chỗ dùng trong báo cáo, bỏ qua.
            flush_paragraph()
            idx += 1
            continue

        image = IMAGE_RE.match(stripped)
        if image:
            flush_paragraph()
            blocks.append(
                Block(kind="image", caption=image.group("caption").strip(), path=image.group("path").strip())
            )
            idx += 1
            continue

        if stripped.startswith("|") and idx + 1 < total and TABLE_SEP_RE.match(lines[idx + 1].strip()):
            flush_paragraph()
            rows = [split_table_row(stripped)]
            idx += 2  # bỏ dòng phân cách
            while idx < total and lines[idx].strip().startswith("|"):
                rows.append(split_table_row(lines[idx].strip()))
                idx += 1
            blocks.append(Block(kind="table", rows=rows))
            continue

        quote = QUOTE_RE.match(stripped)
        if quote:
            flush_paragraph()
            quote_lines = [quote.group(1).strip()]
            idx += 1
            while idx < total:
                nxt = QUOTE_RE.match(lines[idx].strip())
                if not nxt:
                    break
                quote_lines.append(nxt.group(1).strip())
                idx += 1
            blocks.append(Block(kind="quote", lines=quote_lines))
            continue

        bullet = BULLET_RE.match(stripped)
        if bullet:
            flush_paragraph()
            items = [bullet.group(1).strip()]
            idx += 1
            while idx < total:
                nxt = BULLET_RE.match(lines[idx].strip())
                if not nxt:
                    break
                items.append(nxt.group(1).strip())
                idx += 1
            blocks.append(Block(kind="bullets", lines=items))
            continue

        number = NUMBER_RE.match(stripped)
        if number:
            flush_paragraph()
            items = [number.group(1).strip()]
            idx += 1
            while idx < total:
                nxt = NUMBER_RE.match(lines[idx].strip())
                if not nxt:
                    break
                items.append(nxt.group(1).strip())
                idx += 1
            blocks.append(Block(kind="numbers", lines=items))
            continue

        buffer.append(stripped)
        idx += 1

    flush_paragraph()
    return blocks


def split_table_row(line: str) -> list[str]:
    """Tách một dòng bảng dạng ống thành danh sách ô."""
    trimmed = line.strip()
    if trimmed.startswith("|"):
        trimmed = trimmed[1:]
    if trimmed.endswith("|"):
        trimmed = trimmed[:-1]
    return [cell.strip() for cell in trimmed.split("|")]


# ------------------------------------------------------------ chữ bên trong


def iter_inline_runs(text: str, bold: bool = False, italic: bool = False) -> Iterator[InlineRun]:
    """Tách một chuỗi thành các đoạn chữ kèm định dạng in đậm, in nghiêng, mã lệnh."""
    pos = 0
    for match in INLINE_RE.finditer(text):
        if match.start() < pos:
            continue
        if match.start() > pos:
            yield InlineRun(text[pos : match.start()], bold, italic)
        if match.group("code") is not None:
            yield InlineRun(match.group("code")[1:-1], bold, italic, code=True)
        elif match.group("link") is not None:
            label = match.group("ltext")
            url = match.group("lurl")
            yield from iter_inline_runs(label, bold, italic)
            if label.strip() != url.strip():
                yield InlineRun(f" ({url})", bold, italic)
        elif match.group("bold") is not None:
            yield from iter_inline_runs(match.group("btext"), True, italic)
        else:
            yield from iter_inline_runs(match.group("itext"), bold, True)
        pos = match.end()
    if pos < len(text):
        yield InlineRun(text[pos:], bold, italic)


def add_inline(paragraph, text: str, bold: bool = False, italic: bool = False, size: Pt | None = None):
    """Đổ chuỗi Markdown vào một đoạn văn Word, giữ nguyên định dạng bên trong."""
    for piece in iter_inline_runs(text, bold, italic):
        if not piece.text:
            continue
        run = paragraph.add_run(piece.text)
        run.bold = piece.bold or None
        run.italic = piece.italic or None
        if piece.code:
            set_run_font(run, CODE_FONT, size or INLINE_CODE_SIZE)
        elif size is not None:
            run.font.size = size
    return paragraph


def set_run_font(run, name: str, size: Pt | None = None) -> None:
    """Đặt font cho một run, kể cả nhánh east-asia để Word không tự đổi phông."""
    run.font.name = name
    if size is not None:
        run.font.size = size
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), name)


def shade_paragraph(paragraph, fill: str = "F4F4F4") -> None:
    """Tô nền xám nhạt cho một đoạn, dùng cho khối mã."""
    ppr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    ppr.append(shd)


# ---------------------------------------------------------------- tài liệu


def setup_document(doc) -> None:
    """Đặt khổ giấy, lề, phông chữ nền và giãn dòng theo quy ước báo cáo."""
    for section in doc.sections:
        section.page_width = Cm(21)
        section.page_height = Cm(29.7)
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    normal = doc.styles["Normal"]
    normal.font.name = BASE_FONT
    normal.font.size = BASE_SIZE
    rpr = normal.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), BASE_FONT)
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(6)

    for name in ("Title", "Heading 1", "Heading 2", "Heading 3", "Caption", "Quote", "Intense Quote"):
        try:
            style = doc.styles[name]
        except KeyError:
            continue
        style.font.name = BASE_FONT
        element_rpr = style.element.get_or_add_rPr()
        style_fonts = element_rpr.find(qn("w:rFonts"))
        if style_fonts is None:
            style_fonts = OxmlElement("w:rFonts")
            element_rpr.append(style_fonts)
        for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            style_fonts.set(qn(attr), BASE_FONT)


def has_style(doc, name: str) -> bool:
    """Trả về True nếu mẫu tài liệu có sẵn style tên này."""
    try:
        doc.styles[name]
    except KeyError:
        return False
    return True


def add_picture_block(doc, block: Block, base_dir: Path, stats: BuildStats, figure_no: int) -> int:
    """Chèn một hình kèm chú thích, trả về số thứ tự hình kế tiếp."""
    target = (base_dir / block.path).resolve()
    if not target.exists():
        stats.missing_images.append(block.path)
        print(f"  Cảnh báo: thiếu hình {block.path}")
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(f"[Thiếu hình: {block.path}]")
        run.bold = True
        run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
        return figure_no

    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    picture = paragraph.add_run().add_picture(str(target))
    if picture.width > MAX_IMAGE_WIDTH:
        ratio = MAX_IMAGE_WIDTH / picture.width
        picture.width = int(picture.width * ratio)
        picture.height = int(picture.height * ratio)
    stats.images += 1

    caption = doc.add_paragraph()
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    label = f"Hình {figure_no}. {block.caption}" if block.caption else f"Hình {figure_no}."
    caption_run = caption.add_run(label)
    caption_run.italic = True
    caption_run.font.size = Pt(11)
    return figure_no + 1


def add_table_block(doc, block: Block, stats: BuildStats) -> None:
    """Chèn một bảng viền đầy đủ, dòng đầu in đậm."""
    rows = block.rows
    if not rows:
        return
    n_cols = max(len(row) for row in rows)
    table = doc.add_table(rows=len(rows), cols=n_cols)
    if has_style(doc, "Table Grid"):
        table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    for r_idx, row in enumerate(rows):
        for c_idx in range(n_cols):
            cell = table.cell(r_idx, c_idx)
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(2)
            paragraph.paragraph_format.line_spacing = 1.0
            value = row[c_idx] if c_idx < len(row) else ""
            add_inline(paragraph, value, bold=(r_idx == 0), size=TABLE_FONT_SIZE)
    stats.tables += 1


def add_quote_block(doc, block: Block, stats: BuildStats) -> None:
    """Chèn một khối trích dẫn."""
    text = " ".join(line for line in block.lines if line)
    if has_style(doc, "Intense Quote"):
        paragraph = doc.add_paragraph(style="Intense Quote")
        add_inline(paragraph, text)
    else:
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.left_indent = Cm(1)
        add_inline(paragraph, text, italic=True)
    stats.paragraphs += 1


def add_code_block(doc, block: Block) -> None:
    """Chèn một khối mã, mỗi dòng là một đoạn văn giữ nguyên khoảng trắng."""
    lines = block.lines or [""]
    for line in lines:
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.line_spacing = 1.0
        paragraph.paragraph_format.left_indent = Cm(0.5)
        run = paragraph.add_run(line if line else " ")
        set_run_font(run, CODE_FONT, BLOCK_CODE_SIZE)
        shade_paragraph(paragraph)


def is_table_caption(block: Block, next_block: Block | None) -> bool:
    """Đoạn ngắn mở đầu bằng "Bảng" và đứng ngay trước một bảng thì coi là chú thích bảng."""
    if next_block is None or next_block.kind != "table":
        return False
    text = " ".join(block.lines).strip()
    return text.startswith(CAPTION_PREFIX_TABLE) and len(text) < CAPTION_MAX_CHARS


def build_document(blocks: list[Block], base_dir: Path, stats: BuildStats):
    """Đổ toàn bộ khối vào một tài liệu Word mới và trả về tài liệu đó."""
    doc = Document()
    setup_document(doc)

    doc_title = ""
    title_seen = False
    in_title_block = False
    figure_no = 1

    for position, block in enumerate(blocks):
        next_block = blocks[position + 1] if position + 1 < len(blocks) else None

        if block.kind == "heading":
            stats.headings += 1
            if block.level == 1 and not title_seen:
                title_seen = True
                in_title_block = True
                doc_title = block.text
                paragraph = doc.add_paragraph(style="Title" if has_style(doc, "Title") else None)
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_inline(paragraph, block.text)
            else:
                in_title_block = False
                paragraph = doc.add_heading(level=block.level)
                add_inline(paragraph, block.text)
            continue

        if block.kind == "pagebreak":
            in_title_block = False
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
            continue

        if block.kind == "paragraph":
            text = " ".join(block.lines).strip()
            if not text:
                continue
            stats.paragraphs += 1
            paragraph = doc.add_paragraph()
            if in_title_block:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_inline(paragraph, text)
            elif is_table_caption(block, next_block):
                paragraph.paragraph_format.space_after = Pt(3)
                add_inline(paragraph, text, bold=True)
            else:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                add_inline(paragraph, text)
            continue

        in_title_block = False

        if block.kind == "bullets":
            for item in block.lines:
                paragraph = doc.add_paragraph(style="List Bullet" if has_style(doc, "List Bullet") else None)
                add_inline(paragraph, item)
                stats.paragraphs += 1
            continue

        if block.kind == "numbers":
            for item in block.lines:
                paragraph = doc.add_paragraph(style="List Number" if has_style(doc, "List Number") else None)
                add_inline(paragraph, item)
                stats.paragraphs += 1
            continue

        if block.kind == "table":
            add_table_block(doc, block, stats)
            continue

        if block.kind == "image":
            figure_no = add_picture_block(doc, block, base_dir, stats, figure_no)
            continue

        if block.kind == "quote":
            add_quote_block(doc, block, stats)
            continue

        if block.kind == "code":
            add_code_block(doc, block)
            continue

    if doc_title:
        doc.core_properties.title = doc_title
    return doc


def build_report(src: Path, out: Path) -> BuildStats:
    """Đọc Markdown, dựng tài liệu và ghi ra file .docx."""
    if not src.exists():
        raise SystemExit(f"Không tìm thấy file nguồn: {src}")

    text = src.read_text(encoding="utf-8")
    blocks = parse_blocks(text)
    stats = BuildStats()
    doc = build_document(blocks, src.parent, stats)

    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    return stats


def resolve(path_arg: str) -> Path:
    """Đường dẫn tương đối được tính từ thư mục gốc của repo, không phải thư mục đang đứng."""
    path = Path(path_arg).expanduser()
    return path if path.is_absolute() else (ROOT / path)


def print_summary(src: Path, out: Path, stats: BuildStats) -> None:
    """In tóm tắt kết quả dựng báo cáo."""
    size_kb = out.stat().st_size / 1024
    print("\nTóm tắt lần dựng:")
    print(f"  Nguồn        : {src}")
    print(f"  Tiêu đề      : {stats.headings}")
    print(f"  Đoạn văn     : {stats.paragraphs}")
    print(f"  Bảng         : {stats.tables}")
    print(f"  Hình         : {stats.images}")
    print(f"  Hình bị thiếu: {len(stats.missing_images)}")
    for path in stats.missing_images:
        print(f"      - {path}")
    print(f"  Kết quả      : {out}  ({size_kb:.1f} KB)")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dựng báo cáo Word từ Markdown.")
    parser.add_argument("--src", default=DEFAULT_SRC, help=f"File Markdown nguồn (mặc định {DEFAULT_SRC}).")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"File .docx kết quả (mặc định {DEFAULT_OUT}).")
    args = parser.parse_args(list(argv) if argv is not None else None)

    src = resolve(args.src)
    out = resolve(args.out)

    print(f"Đang dựng báo cáo từ {src}")
    stats = build_report(src, out)
    print_summary(src, out, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
