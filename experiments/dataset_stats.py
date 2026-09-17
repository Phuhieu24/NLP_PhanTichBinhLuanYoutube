"""Thống kê mô tả tập dữ liệu `data/dataset_chuan.csv`, ghi ra JSON.

Mọi con số trong phần mô tả dữ liệu của báo cáo đều lấy từ tệp
`results/dataset_stats.json` do script này sinh ra, kèm luôn ĐỊNH NGHĨA đã dùng
(các biểu thức chính quy, cách nhận diện emoji) để người đọc kiểm chứng lại
được thay vì phải đoán.

Chạy:
    python experiments/dataset_stats.py
Thời gian: khoảng 30 giây (phần lớn là pyvi chạy `preprocess_text` trên 20.000 dòng).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from preprocess import preprocess_text  # noqa: E402

LABEL_NAMES = {0: "tiêu cực", 1: "trung tính", 2: "tích cực"}

# Chuỗi 3 ký tự CHỮ CÁI giống nhau trở lên ("hayyy"); [^\W\d_] là chữ cái Unicode.
LETTER_RUN_PATTERN = re.compile(r"([^\W\d_])\1{2,}")
# Chuỗi 3 ký tự BẤT KỲ giống nhau trở lên, tính cả dấu câu và emoji ("!!!", "😂😂😂").
ANY_RUN_PATTERN = re.compile(r"(.)\1{2,}")

# Emoji: ký tự có phạm trù Unicode `So` (Symbol, other), hợp với các khối mã
# emoji không nằm trong `So` (biến thể trình bày U+FE0F, cờ vùng, chữ số khóa).
EMOJI_BLOCKS = (
    (0x1F000, 0x1FAFF),  # Mahjong, domino, emoticons, transport, supplemental symbols
    (0x2600, 0x27BF),    # Miscellaneous symbols, Dingbats
    (0x2B00, 0x2BFF),    # Miscellaneous symbols and arrows
    (0xFE00, 0xFE0F),    # Variation selectors (U+FE0F ép hiển thị dạng emoji)
    (0x1F1E6, 0x1F1FF),  # Regional indicator (cờ quốc gia)
)

EMOJI_DEFINITION = (
    "ký tự có phạm trù Unicode 'So' (Symbol, other), hoặc nằm trong các khối mã emoji "
    "U+1F000-U+1FAFF, U+2600-U+27BF, U+2B00-U+2BFF, U+FE00-U+FE0F, U+1F1E6-U+1F1FF"
)
PUNCTUATION_DEFINITION = (
    "ký tự có phạm trù Unicode bắt đầu bằng 'P' (Pc, Pd, Ps, Pe, Pi, Pf, Po), "
    "tức mọi dấu câu, không tính emoji"
)


def is_emoji(char: str) -> bool:
    if unicodedata.category(char) == "So":
        return True
    code = ord(char)
    return any(low <= code <= high for low, high in EMOJI_BLOCKS)


def has_emoji(text: str) -> bool:
    return any(is_emoji(char) for char in text)


def has_punctuation(text: str) -> bool:
    return any(unicodedata.category(char).startswith("P") for char in text)


def _pct(count: int, total: int) -> float:
    return round(100.0 * count / total, 4) if total else 0.0


def build_stats(df: pd.DataFrame) -> dict:
    texts = df["text"].astype(str)
    total = len(df)
    lengths = texts.str.len()

    emoji_mask = texts.map(has_emoji)
    punct_mask = texts.map(has_punctuation)
    letter_run_mask = texts.map(lambda value: bool(LETTER_RUN_PATTERN.search(value)))
    any_run_mask = texts.map(lambda value: bool(ANY_RUN_PATTERN.search(value)))
    non_nfc = sum(1 for value in texts if unicodedata.normalize("NFC", value) != value)

    print("Đang chạy preprocess_text trên toàn tập (pyvi, khoảng 30 giây)...")
    processed = texts.map(preprocess_text)
    n_empty_after = int((processed.str.strip() == "").sum())

    label_counts = df["label"].value_counts().sort_index()
    return {
        "source": "data/dataset_chuan.csv",
        "n_rows": int(total),
        "columns": list(df.columns),
        "labels": {
            "counts": {str(label): int(count) for label, count in label_counts.items()},
            "names": {str(code): name for code, name in LABEL_NAMES.items()},
            "shares_pct": {str(label): _pct(int(count), total)
                           for label, count in label_counts.items()},
        },
        "duplicates": {
            "n_duplicate_texts": int(texts.duplicated().sum()),
            "definition": "số dòng có cột text trùng với một dòng xuất hiện trước đó",
        },
        "missing": {
            "n_missing_text": int(df["text"].isna().sum()),
            "n_missing_label": int(df["label"].isna().sum()),
        },
        "text_length_chars": {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "median": float(lengths.median()),
            "mean": round(float(lengths.mean()), 4),
            "definition": "số ký tự của cột text gốc, chưa qua tiền xử lý",
        },
        "noise_markers": {
            "n_rows_with_emoji": int(emoji_mask.sum()),
            "pct_rows_with_emoji": _pct(int(emoji_mask.sum()), total),
            "n_rows_with_punctuation": int(punct_mask.sum()),
            "pct_rows_with_punctuation": _pct(int(punct_mask.sum()), total),
            "n_rows_with_emoji_or_punctuation": int((emoji_mask | punct_mask).sum()),
            "pct_rows_with_emoji_or_punctuation": _pct(int((emoji_mask | punct_mask).sum()), total),
            "n_rows_with_letter_run": int(letter_run_mask.sum()),
            "pct_rows_with_letter_run": _pct(int(letter_run_mask.sum()), total),
            "n_rows_with_any_char_run": int(any_run_mask.sum()),
            "pct_rows_with_any_char_run": _pct(int(any_run_mask.sum()), total),
            "n_rows_not_nfc": int(non_nfc),
            "n_rows_empty_after_preprocess": n_empty_after,
            "pct_rows_empty_after_preprocess": _pct(n_empty_after, total),
        },
        "definitions": {
            "emoji": EMOJI_DEFINITION,
            "punctuation": PUNCTUATION_DEFINITION,
            "letter_run_regex": LETTER_RUN_PATTERN.pattern,
            "any_char_run_regex": ANY_RUN_PATTERN.pattern,
            "not_nfc": "unicodedata.normalize('NFC', text) != text",
            "empty_after_preprocess": "preprocess_text(text) rỗng sau khi bỏ khoảng trắng",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Thống kê mô tả tập dữ liệu gán nhãn.")
    parser.add_argument("--data", default=os.path.join(PROJECT_ROOT, "data", "dataset_chuan.csv"),
                        help="Tệp CSV có hai cột text và label.")
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "results"),
                        help="Thư mục ghi kết quả.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.data)
    print(f"Đã đọc {len(df)} dòng từ {args.data}")
    stats = build_stats(df)

    path = os.path.join(args.out_dir, "dataset_stats.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Đã ghi: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
