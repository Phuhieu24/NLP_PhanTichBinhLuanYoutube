# -*- coding: utf-8 -*-
"""Sinh các hình bản nền tối dùng cho bài thuyết trình PowerPoint.

    .venv/bin/python scripts/build_dark_assets.py

Cờ --theme chọn nền cho ma trận nhầm lẫn: dark (#0B1220, mặc định) hoặc slate (#1E293B),
khớp theme cùng tên của scripts/build_slides_pptx.py. Cờ --skip-svg chỉ vẽ lại ma trận và
không ghi đè hai tệp SVG đã commit.

Ba đầu ra trong docs/diagrams/: pipeline_dark.svg, evaluation_dark.svg (đổi màu từ hai sơ đồ
bản sáng, giữ nguyên hình học) và confusion_dark.png (ma trận nhầm lẫn vẽ lại từ
results/metrics.json). Bản sáng trong docs/diagrams/ và results/ không bị đụng tới vì báo cáo
Word vẫn dùng chúng.

Bước đổi SVG sang PNG 2560x1440 dùng playwright, và lùi về rsvg-convert (brew install librsvg)
nếu không có playwright; thiếu cả hai thì các tệp PNG đã commit vẫn dùng được.

Cuối cùng script ghép bản "thân sơ đồ" `<tên>[_dark|_slate]_body.png` cho slide: bỏ tiêu đề
nướng sẵn trong ảnh (trên slide tiêu đề nằm ở vùng tiêu đề, ảnh chỉ lấp vùng nội dung bên dưới),
bỏ dải trống và dòng ghi chú cuối để sơ đồ phóng được to nhất. Cờ --crop-only chỉ chạy bước này
trên các PNG đã có.
"""
import argparse
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]

# Nền ảnh phải khớp nền slide của theme tương ứng trong scripts/build_slides_pptx.py,
# nếu không ma trận hiện thành một vệt chữ nhật khác màu giữa slide.
CM_THEMES = {
    "dark":  dict(bg="#0B1220", fg="#E8EDF5", muted="#93A6C0",
                  ramp=["#0E1B2C", "#12484A", "#15776E", "#2DD4BF"],
                  out="docs/diagrams/confusion_dark.png"),
    "slate": dict(bg="#1E293B", fg="#F8FAFC", muted="#94A3B8",
                  ramp=["#243247", "#1B4F52", "#177C72", "#2DD4BF"],
                  out="docs/diagrams/confusion_slate.png"),
}
_ap = argparse.ArgumentParser(description="Sinh hình nền tối cho bài thuyết trình.")
_ap.add_argument("--theme", choices=sorted(CM_THEMES), default="dark",
                 help="nền của ma trận nhầm lẫn: dark (mặc định) hoặc slate")
_ap.add_argument("--skip-svg", action="store_true",
                 help="chỉ vẽ lại ma trận nhầm lẫn, không ghi đè hai SVG bản tối")
_ap.add_argument("--crop-only", action="store_true",
                 help="chỉ cắt bản thân sơ đồ (_body.png) từ các PNG đã có")
ARGS = _ap.parse_args()
T = CM_THEMES[ARGS.theme]

# Bản thân sơ đồ cho slide, tính trên ảnh 2560x1440: bỏ tiêu đề và phụ đề (hàng 53 đến 149),
# bỏ dải trống giữa khối và chú giải, bỏ dòng ghi chú cuối; giữ các khối và chú giải màu, ghép
# lại với khe 40 px. Nội dung nằm trong cột 62 đến 2497. Đo lại các dải hàng nếu sửa SVG.
BODY_COLS = (40, 2520)
BODY_BANDS = {
    "pipeline": [(190, 1105), (1255, 1350)],    # khối và mũi tên phản hồi; chú giải màu
    "evaluation": [(160, 1180), (1240, 1335)],  # khối và mũi tên; chú giải màu
}
BODY_GAP = 40


def crop_bodies():
    """Ghép bản thân sơ đồ cho slide từ mọi bản màu (sáng, dark, slate) đang có."""
    from PIL import Image
    for name, bands in BODY_BANDS.items():
        for suffix in ("", "_dark", "_slate"):
            src = REPO / f"docs/diagrams/{name}{suffix}.png"
            if not src.exists():
                continue
            im = Image.open(src).convert("RGB")
            sx, sy = im.width / 2560, im.height / 1440
            left, right = (int(v * sx) for v in BODY_COLS)
            parts = [im.crop((left, int(a * sy), right, int(b * sy))) for a, b in bands]
            gap = int(BODY_GAP * sy)
            out = Image.new("RGB", (right - left, sum(p.height for p in parts) + gap * (len(parts) - 1)),
                            im.getpixel((2, 2)))
            y = 0
            for part in parts:
                out.paste(part, (0, y))
                y += part.height + gap
            dst = src.with_name(f"{name}{suffix}_body.png")
            out.save(dst, optimize=True)
            print("->", dst, out.size)


if ARGS.crop_only:
    crop_bodies()
    raise SystemExit(0)


import re, sys
from pathlib import Path

SHAPE = {  # fill của hình khối
 "FAFAF9": "0B1220", "FFFFFF": "142236", "F1F5F9": "1A2639", "FFFBEB": "2C2111",
 "F5F3FF": "241D40", "FAF5FF": "241D40", "ECFDF5": "102E26", "ECFEFF": "0E2A35",
 "FFF1F2": "301724", "475569": "41567A", "D97706": "C2740B", "7C3AED": "7C4DF0",
 "059669": "0E9F6E", "0891B2": "0E9BC4", "E11D48": "D9214B", "F59E0B": "F59E0B",
 "94A3B8": "94A3B8", "64748B": "7C8DA6",
}
STROKE = {
 "F59E0B": "F5A524", "94A3B8": "8296B0", "7C3AED": "A78BFA", "059669": "34D399",
 "0891B2": "22D3EE", "FB7185": "FB7185", "64748B": "8296B0", "475569": "5E7characters",
 "0D9488": "2DD4BF",
}
STROKE["475569"] = "5E749A"

# Theme slate dùng lại toàn bộ ánh xạ của theme dark, chỉ nâng các tông bề mặt lên cho
# khớp nền slide #1E293B. Nếu giữ tông của theme dark thì các thẻ trong sơ đồ sẽ tối hơn
# chính nền slide, nhìn như bị thụt xuống. Màu nhấn và màu chữ giữ nguyên.
if ARGS.theme == "slate":
    SHAPE.update({
        "FAFAF9": "1E293B", "FFFFFF": "2B3A50", "F1F5F9": "334155",
        "FFFBEB": "3E301A", "F5F3FF": "332A55", "FAF5FF": "332A55",
        "ECFDF5": "20423A", "ECFEFF": "1E3C49", "FFF1F2": "422637",
    })
TEXT = {
 "0F172A": "F1F5F9", "64748B": "9AAABF", "78716C": "8A97A8", "475569": "B7C4D6",
 "FFFFFF": "FFFFFF", "92400E": "FCD34D", "B45309": "FBBF24", "4C1D95": "C4B5FD",
 "6D28D9": "A78BFA", "065F46": "6EE7B7", "047857": "34D399", "155E75": "67E8F9",
 "0E7490": "22D3EE", "9F1239": "FDA4AF", "E11D48": "FB7185", "5B21B6": "C4B5FD",
 "D97706": "FBBF24", "059669": "34D399", "0891B2": "22D3EE", "7C3AED": "A78BFA",
 "F59E0B": "FBBF24", "94A3B8": "9AAABF",
}
TAG = re.compile(r"<(\w+)\b[^>]*>")

def conv(tag_src, is_text):
    def sub_attr(m):
        attr, val = m.group(1), m.group(2).upper().lstrip("#")
        table = TEXT if is_text else (STROKE if attr == "stroke" else SHAPE)
        return f'{attr}="#{table.get(val, m.group(2).lstrip("#"))}"'
    return re.sub(r'(fill|stroke)="#([0-9A-Fa-f]{6})"', sub_attr, tag_src)

for name in (("pipeline", "evaluation") if not ARGS.skip_svg else ()):
    src = REPO / f"docs/diagrams/{name}.svg"
    s = src.read_text(encoding="utf-8")
    out, pos = [], 0
    for m in TAG.finditer(s):
        out.append(s[pos:m.start()])
        out.append(conv(m.group(0), m.group(1) == "text"))
        pos = m.end()
    out.append(s[pos:])
    dst = REPO / f"docs/diagrams/{name}_{ARGS.theme}.svg"
    dst.write_text("".join(out), encoding="utf-8")
    print("->", dst, dst.stat().st_size, "bytes")


import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

m = json.loads((REPO / "results/metrics.json").read_text(encoding="utf-8"))
cm = np.array(m["test"]["confusion_matrix"], dtype=float)
norm = cm / cm.sum(axis=1, keepdims=True)
labels = ["Tiêu cực", "Trung tính", "Tích cực"]
BG, FG, MUTED, ACC = T["bg"], T["fg"], T["muted"], "#2DD4BF"

from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("teal_dark", T["ramp"])

fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=200)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        v = norm[i, j]
        ax.text(j, i - 0.10, f"{v:.2f}", ha="center", va="center",
                color="#06131F" if v > 0.62 else FG, fontsize=19, fontweight="bold")
        ax.text(j, i + 0.19, f"{int(cm[i, j]):,}".replace(",", "."), ha="center", va="center",
                color="#06131F" if v > 0.62 else MUTED, fontsize=12)
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(labels, fontsize=13, color=FG)
ax.set_yticklabels(labels, fontsize=13, color=FG)
ax.set_xlabel("Nhãn dự đoán", fontsize=13.5, color=MUTED, labelpad=10)
ax.set_ylabel("Nhãn thật", fontsize=13.5, color=MUTED, labelpad=10)
ax.set_title("Ma trận nhầm lẫn, chuẩn hóa theo hàng", fontsize=15, color=FG, pad=16, fontweight="bold")
for sp in ax.spines.values(): sp.set_visible(False)
ax.tick_params(length=0, colors=FG)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.outline.set_visible(False); cb.ax.tick_params(colors=MUTED, labelsize=10)
fig.tight_layout()
out = str(REPO / T["out"])
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.22)
print("->", out)


def _pairs():
    for name in ("pipeline", "evaluation"):
        yield (REPO / f"docs/diagrams/{name}_{ARGS.theme}.svg",
               REPO / f"docs/diagrams/{name}_{ARGS.theme}.png")


def render_rsvg():
    """Đổi SVG sang PNG 2560x1440 bằng rsvg-convert (librsvg, cài qua Homebrew)."""
    import shutil, subprocess
    exe = shutil.which("rsvg-convert")
    if not exe:
        raise FileNotFoundError("rsvg-convert")
    for src, dst in _pairs():
        subprocess.run([exe, "-w", "2560", "-h", "1440", "-o", str(dst), str(src)], check=True)
        print("->", dst)


def render_png():
    """Đổi SVG sang PNG 2560x1440 bằng playwright."""
    from playwright.sync_api import sync_playwright
    import time
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=2)
        for src, dst in _pairs():
            pg.goto(src.as_uri())
            time.sleep(0.8)
            pg.screenshot(path=str(dst))
            print("->", dst)
        b.close()


if __name__ == "__main__":
    if ARGS.skip_svg:
        raise SystemExit(0)
    try:
        render_png()
    except ImportError:
        try:
            render_rsvg()
        except (FileNotFoundError, OSError):
            print("Bỏ qua bước render PNG: không có playwright lẫn rsvg-convert. "
                  "Các tệp PNG đã commit vẫn dùng được.")
    crop_bodies()
