# -*- coding: utf-8 -*-
"""Sinh các hình bản nền tối dùng cho bài thuyết trình PowerPoint.

    .venv/bin/python scripts/build_dark_assets.py

Ba đầu ra trong docs/diagrams/: pipeline_dark.svg, evaluation_dark.svg (đổi màu từ hai sơ đồ
bản sáng, giữ nguyên hình học) và confusion_dark.png (ma trận nhầm lẫn vẽ lại từ
results/metrics.json). Bản sáng trong docs/diagrams/ và results/ không bị đụng tới vì báo cáo
Word vẫn dùng chúng.

Bước đổi SVG sang PNG 2560x1440 cần playwright và chromium; nếu máy không có, hai tệp PNG đã
được commit sẵn nên bài thuyết trình vẫn dựng được.
"""
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]


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

for name in ("pipeline", "evaluation"):
    src = REPO / f"docs/diagrams/{name}.svg"
    s = src.read_text(encoding="utf-8")
    out, pos = [], 0
    for m in TAG.finditer(s):
        out.append(s[pos:m.start()])
        out.append(conv(m.group(0), m.group(1) == "text"))
        pos = m.end()
    out.append(s[pos:])
    dst = REPO / f"docs/diagrams/{name}_dark.svg"
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
BG, FG, MUTED, ACC = "#0B1220", "#E8EDF5", "#93A6C0", "#2DD4BF"

from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("teal_dark", ["#0E1B2C", "#12484A", "#15776E", "#2DD4BF"])

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
out = str(REPO / "docs/diagrams/confusion_dark.png")
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.22)
print("->", out)


def render_png():
    """Đổi hai SVG bản tối sang PNG 2560x1440 (cần playwright)."""
    from playwright.sync_api import sync_playwright
    import time
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=2)
        for name in ("pipeline", "evaluation"):
            pg.goto((REPO / f"docs/diagrams/{name}_dark.svg").as_uri())
            time.sleep(0.8)
            pg.screenshot(path=str(REPO / f"docs/diagrams/{name}_dark.png"))
            print("->", REPO / f"docs/diagrams/{name}_dark.png")
        b.close()


if __name__ == "__main__":
    try:
        render_png()
    except ImportError:
        print("Bỏ qua bước render PNG: chưa cài playwright. Hai tệp PNG đã commit vẫn dùng được.")
