"""Xuất các bình luận bị phân loại sai trên tập test để phân tích lỗi bằng tay.

Script dựng lại ĐÚNG tập test mà `src/train_sentiment.py` đã dùng (cùng dữ liệu,
cùng tiền xử lý, cùng lần chia phân tầng 80/20 với `random_state=42`), rồi dự đoán
bằng mô hình và vectorizer đã lưu trong `models/`. Trước khi ghi bất cứ thứ gì,
script đối chiếu accuracy tính lại với `results/metrics.json`; lệch một chút là
dừng, vì khi đó danh sách lỗi không còn tương ứng với mô hình trong báo cáo.

`margin` = điểm `decision_function` của lớp mô hình chọn trừ điểm của lớp đúng.
Số này luôn dương với một dòng bị sai; càng lớn thì mô hình càng "tự tin sai",
nên sắp xếp giảm dần sẽ đưa các lỗi đáng xem nhất lên đầu.

Chạy:
    python experiments/export_misclassified.py
Thời gian: khoảng 30 giây (phần lớn là pyvi chạy `preprocess_text` trên 20.000 dòng).
"""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from preprocess import preprocess_text  # noqa: E402

LABEL_NAMES = {0: "tiêu cực", 1: "trung tính", 2: "tích cực"}
TEST_SIZE = 0.2
SPLIT_SEED = 42  # phải trùng `--seed` mặc định của src/train_sentiment.py
SAMPLE_SEED = 42
SAMPLE_SIZE = 150
ACCURACY_TOLERANCE = 1e-9

COLUMNS = ["row_id", "text", "label_true", "label_pred", "margin"]


def build_test_split(data_path):
    """Dựng lại tập test của `train_sentiment.py`, giữ kèm chỉ số dòng gốc.

    `row_id` là chỉ số dòng trong `data/dataset_chuan.csv` (0 cho dòng dữ liệu
    đầu tiên, không tính dòng tiêu đề), gắn vào trước khi `reset_index` nên vẫn
    truy ngược được về file gốc dù thứ tự đã đổi.
    """
    df = pd.read_csv(data_path)
    df["row_id"] = df.index
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    n_loaded = len(df)
    df["text_clean"] = df["text"].apply(preprocess_text)
    df = df[df["text_clean"].str.strip() != ""].reset_index(drop=True)
    n_dropped = n_loaded - len(df)

    _, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=df["label"]
    )
    return test_df, n_dropped


def predict_with_margin(model, vectorizer, test_df):
    """Dự đoán tập test và tính `margin` cho từng dòng."""
    features = vectorizer.transform(test_df["text_clean"])
    scores = model.decision_function(features)
    classes = list(model.classes_)
    pred = model.predict(features)

    true_idx = np.asarray([classes.index(label) for label in test_df["label"]])
    pred_idx = np.asarray([classes.index(label) for label in pred])
    rows = np.arange(len(test_df))
    margin = scores[rows, pred_idx] - scores[rows, true_idx]

    out = pd.DataFrame(
        {
            "row_id": test_df["row_id"].to_numpy(),
            "text": test_df["text"].to_numpy(),
            "label_true": test_df["label"].to_numpy(),
            "label_pred": pred,
            "margin": margin,
        }
    )
    return out, float(accuracy_score(test_df["label"], pred))


def stratified_error_sample(errors, size=SAMPLE_SIZE, seed=SAMPLE_SEED):
    """Lấy mẫu phân tầng theo nhãn ĐÚNG, giữ đúng tỷ lệ các nhãn trong tập lỗi.

    Chia hạn ngạch theo phương pháp phần dư lớn nhất để tổng đúng bằng `size`
    kể cả khi tỷ lệ không chia hết.
    """
    counts = errors["label_true"].value_counts().sort_index()
    exact = counts / counts.sum() * size
    quota = np.floor(exact).astype(int)
    remainder = size - int(quota.sum())
    if remainder > 0:
        order = (exact - quota).sort_values(ascending=False).index
        for label in list(order)[:remainder]:
            quota[label] += 1

    parts = [
        errors[errors["label_true"] == label].sample(
            n=int(quota[label]), random_state=seed
        )
        for label in counts.index
    ]
    sample = pd.concat(parts).sort_values("margin", ascending=False)
    sample["hien_tuong"] = ""  # cột trống để gán nhãn hiện tượng lỗi bằng tay
    return sample


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.path.join(PROJECT_ROOT, "data", "dataset_chuan.csv"))
    parser.add_argument("--models", default=os.path.join(PROJECT_ROOT, "models"))
    parser.add_argument("--out-results", default=os.path.join(PROJECT_ROOT, "results"))
    args = parser.parse_args(argv)

    results_dir = os.path.abspath(args.out_results)
    os.makedirs(results_dir, exist_ok=True)

    print("Đang dựng lại tập test (tiền xử lý + chia phân tầng 80/20, seed 42)...")
    test_df, n_dropped = build_test_split(args.data)
    print(f"Đã loại {n_dropped} dòng rỗng sau tiền xử lý; tập test có {len(test_df)} dòng.")

    model = joblib.load(os.path.join(args.models, "sentiment_model.pkl"))
    vectorizer = joblib.load(os.path.join(args.models, "tfidf_vectorizer.pkl"))
    predictions, accuracy = predict_with_margin(model, vectorizer, test_df)

    with open(os.path.join(results_dir, "metrics.json"), encoding="utf-8") as handle:
        expected = float(json.load(handle)["test"]["accuracy"])
    if abs(accuracy - expected) > ACCURACY_TOLERANCE:
        raise SystemExit(
            f"Accuracy tính lại {accuracy:.6f} khác {expected:.6f} trong results/metrics.json: "
            "tập test hoặc mô hình đã lệch, dừng để khỏi xuất danh sách lỗi sai ngữ cảnh."
        )
    print(f"Accuracy tập test khớp results/metrics.json: {accuracy:.5f}")

    errors = predictions[predictions["label_true"] != predictions["label_pred"]]
    errors = errors.sort_values("margin", ascending=False).reset_index(drop=True)
    n_correct = len(predictions) - len(errors)
    print(f"Đúng {n_correct} / {len(predictions)} dòng, sai {len(errors)} dòng.")

    errors_path = os.path.join(results_dir, "misclassified_test.csv")
    errors.to_csv(errors_path, index=False, encoding="utf-8-sig")

    sample = stratified_error_sample(errors).reset_index(drop=True)
    sample_path = os.path.join(results_dir, "error_sample_150.csv")
    sample.to_csv(sample_path, index=False, encoding="utf-8-sig")

    print("\nSố lỗi theo nhãn đúng (toàn bộ / mẫu):")
    error_counts = errors["label_true"].value_counts().sort_index()
    sample_counts = sample["label_true"].value_counts().sort_index()
    for label in sorted(error_counts.index):
        name = LABEL_NAMES.get(int(label), str(label))
        print(
            f"  {label} ({name}): {int(error_counts[label])} lỗi "
            f"-> {int(sample_counts.get(label, 0))} dòng trong mẫu"
        )
    print(f"  tổng: {len(errors)} lỗi -> {len(sample)} dòng trong mẫu")

    print(f"\nĐã ghi {errors_path}")
    print(f"Đã ghi {sample_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
