"""So sánh A/B hai bản tiền xử lý trên cùng một cấu hình phân loại cảm xúc.

"cũ" = bản tiền xử lý gốc ở commit `afb1c3e` (`experiments/preprocess_baseline.py`),
"mới" = bản đang xuất xưởng (`src/preprocess.py`). Mọi thứ còn lại giữ nguyên:
TF-IDF (1,2)-gram, 15.000 đặc trưng, `sublinear_tf`; LinearSVC với
`class_weight="balanced"`, `dual=False`, `random_state=42`; chia phân tầng 80/20
theo từng hạt giống.

Mỗi biến thể tự loại các dòng rỗng sau tiền xử lý của chính nó, vì hai bản làm
sạch bỏ đi số dòng khác nhau.

Kết quả ghi vào `results/ab_preprocess.csv` (một dòng cho mỗi bộ ba
tiền-xử-lý × C × hạt giống) và `results/ab_preprocess.txt` (bảng tổng hợp:
trung bình ± độ lệch chuẩn ddof=1 trên các hạt giống 0-4, kèm riêng hạt giống 42).

Chạy:
    python experiments/ab_preprocess.py
Thời gian: khoảng 1-2 phút (hai lượt pyvi trên 20.000 dòng + 24 lần khớp LinearSVC).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import preprocess as shipped  # noqa: E402  (bản mới, trong src/)
import preprocess_baseline as baseline  # noqa: E402  (bản gốc afb1c3e, đóng băng)

LABELS = [0, 1, 2]
LABEL_NAMES = {0: "tiêu cực", 1: "trung tính", 2: "tích cực"}
TFIDF_PARAMS = {"max_features": 15000, "ngram_range": (1, 2), "sublinear_tf": True}
TEST_SIZE = 0.2
MODEL_SEED = 42  # `random_state` của LinearSVC, cố định để chỉ hạt giống chia dữ liệu thay đổi

VARIANTS = {
    "old": lambda text: baseline.tokenize_vietnamese(baseline.clean_text(text)),
    "new": lambda text: shipped.preprocess_text(text),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A/B hai bản tiền xử lý trên cùng một bộ phân loại.")
    parser.add_argument("--data", default=os.path.join(PROJECT_ROOT, "data", "dataset_chuan.csv"),
                        help="Tệp CSV có hai cột text và label.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 42],
                        help="Các hạt giống chia train/test.")
    parser.add_argument("--c", type=float, nargs="+", default=[0.3, 1.0],
                        help="Các giá trị C của LinearSVC.")
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "results"),
                        help="Thư mục ghi kết quả.")
    return parser


def prepare(df: pd.DataFrame, transform) -> tuple[pd.DataFrame, int]:
    """Tiền xử lý một lần cho cả thí nghiệm, bỏ các dòng rỗng của biến thể đó."""
    out = df.copy()
    out["text_clean"] = out["text"].apply(transform)
    kept = out[out["text_clean"].str.strip() != ""].reset_index(drop=True)
    return kept, len(out) - len(kept)


def unigram_vocab_size(texts) -> int:
    """Số token đơn phân biệt nhau (tách theo khoảng trắng) của cả tập."""
    vocab = set()
    for text in texts:
        vocab.update(str(text).split())
    return len(vocab)


def run_one(frame: pd.DataFrame, c_value: float, seed: int) -> dict:
    train_df, test_df = train_test_split(
        frame, test_size=TEST_SIZE, random_state=seed, stratify=frame["label"]
    )
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    x_train = vectorizer.fit_transform(train_df["text_clean"])
    model = LinearSVC(C=c_value, class_weight="balanced", dual=False, random_state=MODEL_SEED)
    model.fit(x_train, train_df["label"])
    y_true = test_df["label"].to_numpy()
    y_pred = model.predict(vectorizer.transform(test_df["text_clean"]))
    _, _, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "f1_negative": float(per_class_f1[0]),
        "f1_neutral": float(per_class_f1[1]),
        "f1_positive": float(per_class_f1[2]),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }


def _mean_std(values) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        return float(array.mean()) if array.size else float("nan"), float("nan")
    return float(array.mean()), float(array.std(ddof=1))


def write_report(rows, meta, path) -> str:
    frame = pd.DataFrame(rows)
    main_seeds = [seed for seed in meta["seeds"] if seed != 42]
    lines = [
        "A/B TIỀN XỬ LÝ: bản gốc afb1c3e ('old') so với bản xuất xưởng ('new')",
        "=" * 78,
        f"Dữ liệu        : {meta['data']} ({meta['n_rows']} dòng)",
        "Đặc trưng      : TfidfVectorizer(ngram_range=(1,2), max_features=15000, sublinear_tf=True)",
        f"Bộ phân loại   : LinearSVC(class_weight='balanced', dual=False, random_state={MODEL_SEED})",
        f"Chia dữ liệu   : phân tầng {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} theo từng hạt giống",
        f"Hạt giống      : {meta['seeds']}",
        f"scikit-learn   : {meta['sklearn_version']}",
        f"Tổng thời gian : {meta['elapsed_seconds']:.1f}s",
        "",
        "Số dòng bị loại vì rỗng sau tiền xử lý, và số token đơn phân biệt nhau",
        "(tách theo khoảng trắng trên toàn tập, trước khi cắt còn 15.000 đặc trưng TF-IDF):",
    ]
    for name in meta["variants"]:
        info = meta["variants"][name]
        lines.append(
            f"  {name:<3} : loại {info['n_dropped']} dòng, còn {info['n_rows']} dòng, "
            f"{info['unigram_vocab']} token đơn"
        )
    lines += [
        "",
        f"TRUNG BÌNH ± ĐỘ LỆCH CHUẨN (ddof=1) trên các hạt giống {main_seeds}",
        "-" * 78,
        f"{'prep':<5}{'C':>6}{'accuracy':>20}{'macro-F1':>22}{'F1 trung tính':>18}",
    ]
    for name in meta["variants"]:
        for c_value in meta["c_values"]:
            subset = frame[(frame["prep"] == name) & (frame["C"] == c_value)
                           & (frame["seed"].isin(main_seeds))]
            acc_mean, acc_std = _mean_std(subset["accuracy"])
            f1_mean, f1_std = _mean_std(subset["f1_macro"])
            neu_mean, _ = _mean_std(subset["f1_neutral"])
            accuracy = f"{acc_mean * 100:.2f}% ± {acc_std * 100:.2f}"
            macro_f1 = f"{f1_mean:.4f} ± {f1_std:.4f}"
            neutral_f1 = f"{neu_mean:.4f}"
            lines.append(
                f"{name:<5}{c_value:>6g}{accuracy:>20}{macro_f1:>22}{neutral_f1:>18}"
            )
    lines += [
        "",
        "HẠT GIỐNG 42 (đúng lần chia mà src/train_sentiment.py dùng)",
        "-" * 78,
        f"{'prep':<5}{'C':>6}{'accuracy':>20}{'macro-F1':>22}{'F1 trung tính':>18}",
    ]
    for name in meta["variants"]:
        for c_value in meta["c_values"]:
            subset = frame[(frame["prep"] == name) & (frame["C"] == c_value)
                           & (frame["seed"] == 42)]
            if subset.empty:
                continue
            row = subset.iloc[0]
            accuracy = f"{row['accuracy'] * 100:.3f}%"
            macro_f1 = f"{row['f1_macro']:.4f}"
            neutral_f1 = f"{row['f1_neutral']:.4f}"
            lines.append(
                f"{name:<5}{c_value:>6g}{accuracy:>20}{macro_f1:>22}{neutral_f1:>18}"
            )
    lines += [
        "",
        "Chi tiết từng lần chạy nằm trong results/ab_preprocess.csv.",
        "",
    ]
    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return text


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.data)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    print(f"Đã đọc {len(df)} dòng từ {args.data}")

    rows = []
    variants_meta = {}
    for name, transform in VARIANTS.items():
        print(f"Tiền xử lý biến thể '{name}' (pyvi trên toàn tập, mất khoảng 20-30 giây)...")
        frame, n_dropped = prepare(df, transform)
        variants_meta[name] = {
            "n_dropped": int(n_dropped),
            "n_rows": int(len(frame)),
            "unigram_vocab": unigram_vocab_size(frame["text_clean"]),
        }
        for c_value in args.c:
            for seed in args.seeds:
                result = run_one(frame, c_value, seed)
                rows.append({"prep": name, "C": float(c_value), "seed": int(seed), **result})
                print(f"  {name} C={c_value:g} seed={seed}: "
                      f"accuracy {result['accuracy'] * 100:.2f}%, macro-F1 {result['f1_macro']:.4f}")

    csv_path = os.path.join(args.out_dir, "ab_preprocess.csv")
    columns = ["prep", "C", "seed", "accuracy", "f1_macro", "f1_negative", "f1_neutral",
               "f1_positive", "n_train", "n_test"]
    pd.DataFrame(rows, columns=columns).to_csv(csv_path, index=False, encoding="utf-8-sig")

    meta = {
        "data": os.path.relpath(args.data, PROJECT_ROOT),
        "n_rows": int(len(df)),
        "seeds": [int(seed) for seed in args.seeds],
        "c_values": [float(value) for value in args.c],
        "variants": variants_meta,
        "sklearn_version": sklearn.__version__,
        "elapsed_seconds": time.perf_counter() - started,
    }
    txt_path = os.path.join(args.out_dir, "ab_preprocess.txt")
    print()
    print(write_report(rows, meta, txt_path))
    print(f"Đã ghi: {csv_path}")
    print(f"Đã ghi: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
