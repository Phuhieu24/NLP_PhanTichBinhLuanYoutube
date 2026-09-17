"""Huấn luyện và đánh giá mô hình phân loại cảm xúc 3 lớp cho bình luận tiếng Việt.

Quy trình: tiền xử lý một lần -> chia tập phân tầng 80/20 -> so sánh các mô hình
nền bằng cross-validation trên tập train -> dò tham số C cho LinearSVC -> huấn
luyện mô hình cuối và đánh giá đúng MỘT lần trên tập test -> kiểm tra độ ổn định
theo 5 hạt giống ngẫu nhiên -> lưu mô hình, chỉ số và thẻ mô hình.

Mọi lựa chọn mô hình đều chỉ dùng tập train, tập test không tham gia vào bất kỳ
bước chọn lựa nào (tránh rò rỉ dữ liệu).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")  # phải đặt trước khi import pyplot (chạy không cần màn hình)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import sklearn  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split  # noqa: E402
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

import preprocess

# Tương thích cả bản preprocess cũ (clean_text + tokenize_vietnamese) lẫn bản mới
# (preprocess_text). Hợp đồng giao diện nằm ở PLAN mục 4.1.
_preprocess = getattr(preprocess, "preprocess_text", None) or (
    lambda t: preprocess.tokenize_vietnamese(preprocess.clean_text(t))
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LABELS = [0, 1, 2]
LABEL_NAMES = ["Tiêu cực (0)", "Trung tính (1)", "Tích cực (2)"]
LABEL_MAP = {"0": "Tiêu cực", "1": "Trung tính", "2": "Tích cực"}

TEST_SIZE = 0.2
N_SPLITS = 5
CV_SEED = 42  # hạt giống cố định cho StratifiedKFold khi chọn mô hình
C_GRID = [0.1, 0.3, 1.0, 3.0]
ROBUSTNESS_SEEDS = [0, 1, 2, 3, 4]
TOP_FEATURES = 25

TFIDF_PARAMS = {"max_features": 15000, "ngram_range": (1, 2), "sublinear_tf": True}
PREPROCESSING_DESC = (
    "clean_text (chuẩn hóa NFC, bỏ URL và ký tự đặc biệt, gộp chữ lặp, chuẩn hóa teencode chính tả) "
    "-> pyvi ViTokenizer -> lowercase"
)


# --------------------------------------------------------------------------- #
# Dữ liệu
# --------------------------------------------------------------------------- #
def load_dataset(csv_path):
    """Đọc dữ liệu gán nhãn; yêu cầu hai cột `text` và `label`."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {csv_path}.")
    df = pd.read_csv(csv_path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"File {csv_path} thiếu cột bắt buộc: {sorted(missing)}")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    return df


def preprocess_corpus(df, text_col="text"):
    """Tiền xử lý MỘT lần rồi tái sử dụng cho mọi lần khớp mô hình.

    Tách từ bằng pyvi trên 20.000 dòng mất khoảng 20 giây, nên không lặp lại
    bước này trong từng fold của cross-validation.
    """
    out = df.copy()
    out["text_clean"] = out[text_col].apply(_preprocess)
    out = out[out["text_clean"].str.strip() != ""].reset_index(drop=True)
    return out


def build_vectorizer():
    return TfidfVectorizer(**TFIDF_PARAMS)


def _make_pipeline(clf):
    return Pipeline([("tfidf", build_vectorizer()), ("clf", clf)])


def _linear_svc(c_value, seed):
    return LinearSVC(C=c_value, class_weight="balanced", dual=False, random_state=seed)


# --------------------------------------------------------------------------- #
# Chọn mô hình (chỉ trên tập train)
# --------------------------------------------------------------------------- #
def candidate_models(seed):
    """Các mô hình đem ra so sánh, kể cả mô hình nền đoán theo lớp đa số."""
    return [
        ("MostFrequent", DummyClassifier(strategy="most_frequent")),
        ("MultinomialNB", MultinomialNB()),
        (
            "LogisticRegression",
            LogisticRegression(
                class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=seed
            ),
        ),
        ("LinearSVC", _linear_svc(1.0, seed)),
    ]


def _cv_splitter():
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)


def compare_models(texts, labels, seed=42, n_jobs=-1):
    """So sánh các mô hình ứng viên bằng cross-validation 5-fold trên tập train."""
    records = []
    for name, clf in candidate_models(seed):
        scores = cross_validate(
            _make_pipeline(clf),
            texts,
            labels,
            cv=_cv_splitter(),
            scoring=["accuracy", "f1_macro"],
            n_jobs=n_jobs,
        )
        records.append(
            {
                "model": name,
                "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
                "cv_accuracy_std": float(np.std(scores["test_accuracy"])),
                "cv_f1_macro_mean": float(np.mean(scores["test_f1_macro"])),
                "cv_f1_macro_std": float(np.std(scores["test_f1_macro"])),
                "fit_seconds": float(np.mean(scores["fit_time"])),
            }
        )
    return records


def tune_linear_svc(texts, labels, c_grid=None, seed=42, n_jobs=-1):
    """Dò tham số C của LinearSVC theo macro-F1 cross-validation trên tập train.

    Trả về `(C tốt nhất, danh sách kết quả từng C)`. Mỗi dòng ghi cả macro-F1
    lẫn accuracy kèm độ lệch chuẩn giữa các fold, nhờ vậy bằng chứng lưu trong
    `results/metrics.json` đủ để người đọc tự đối chiếu với bảng so sánh mô hình
    thay vì phải tin vào một con số trên màn hình.
    """
    c_grid = list(c_grid or C_GRID)
    records = []
    for c_value in c_grid:
        scores = cross_validate(
            _make_pipeline(_linear_svc(c_value, seed)),
            texts,
            labels,
            cv=_cv_splitter(),
            scoring=["accuracy", "f1_macro"],
            n_jobs=n_jobs,
        )
        records.append(
            {
                "C": float(c_value),
                "cv_f1_macro_mean": float(np.mean(scores["test_f1_macro"])),
                "cv_f1_macro_std": float(np.std(scores["test_f1_macro"])),
                "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
                "cv_accuracy_std": float(np.std(scores["test_accuracy"])),
                "fit_seconds": float(np.mean(scores["fit_time"])),
            }
        )
    best = max(records, key=lambda r: r["cv_f1_macro_mean"])
    return float(best["C"]), records


def tuned_comparison_row(best_c, grid_records):
    """Dòng bảng so sánh ứng với đúng cấu hình đem đi huấn luyện cuối cùng.

    Bốn dòng đầu của bảng chạy LinearSVC ở C mặc định (1.0), còn mô hình được
    lưu lại dùng C đã dò. Thiếu dòng này thì bảng đang mô tả một mô hình khác
    với mô hình thực sự xuất xưởng.
    """
    for record in grid_records or []:
        if float(record["C"]) == float(best_c):
            return {
                "model": f"LinearSVC (C={best_c:g}, đã dò)",
                "cv_accuracy_mean": record["cv_accuracy_mean"],
                "cv_accuracy_std": record["cv_accuracy_std"],
                "cv_f1_macro_mean": record["cv_f1_macro_mean"],
                "cv_f1_macro_std": record["cv_f1_macro_std"],
                "fit_seconds": record["fit_seconds"],
            }
    return None


# --------------------------------------------------------------------------- #
# Huấn luyện và đánh giá
# --------------------------------------------------------------------------- #
def fit_final_model(train_texts, train_labels, c_value, seed=42):
    """Khớp TF-IDF và LinearSVC trên toàn bộ tập train với C đã chọn."""
    vectorizer = build_vectorizer()
    x_train = vectorizer.fit_transform(train_texts)
    model = _linear_svc(c_value, seed)
    model.fit(x_train, train_labels)
    return vectorizer, model


def evaluate(model, vectorizer, test_texts, test_labels):
    """Đánh giá đúng một lần trên tập test đã giữ riêng."""
    y_true = np.asarray(test_labels)
    y_pred = model.predict(vectorizer.transform(test_texts))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    per_class = {
        str(label): {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(LABELS)
    }
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    report = classification_report(
        y_true, y_pred, labels=LABELS, target_names=LABEL_NAMES, zero_division=0, digits=4
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
        ),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": list(LABEL_NAMES),
        "_report_text": report,
    }


def seed_robustness(df, c_value, seeds=None):
    """Chia lại toàn bộ dữ liệu theo từng hạt giống rồi khớp lại cấu hình đã chọn.

    Mục đích: cho thấy chênh lệch giữa các lần chia ngẫu nhiên, thay vì báo cáo
    một con số duy nhất từ một lần chia may mắn.
    """
    seeds = list(seeds if seeds is not None else ROBUSTNESS_SEEDS)
    accuracies, f1_macros = [], []
    for seed in seeds:
        train_df, test_df = train_test_split(
            df, test_size=TEST_SIZE, random_state=seed, stratify=df["label"]
        )
        vectorizer, model = fit_final_model(
            train_df["text_clean"], train_df["label"], c_value, seed=seed
        )
        y_pred = model.predict(vectorizer.transform(test_df["text_clean"]))
        accuracies.append(float(accuracy_score(test_df["label"], y_pred)))
        f1_macros.append(
            float(
                f1_score(test_df["label"], y_pred, labels=LABELS, average="macro", zero_division=0)
            )
        )
    return {
        "seeds": seeds,
        "test_accuracy": accuracies,
        "test_f1_macro": f1_macros,
        "f1_macro_mean": float(np.mean(f1_macros)),
        "f1_macro_std": float(np.std(f1_macros)),
    }


def top_features_per_class(model, vectorizer, top_n=TOP_FEATURES):
    """Lấy các n-gram có trọng số dương lớn nhất của từng lớp từ `coef_`.

    Tra hàng trọng số theo `model.classes_` chứ không theo vị trí trong
    `LABELS`: khi tập huấn luyện thiếu một lớp, `coef_` chỉ có hàng cho những
    lớp thực sự xuất hiện, lấy theo vị trí sẽ gán nhầm trọng số của lớp này cho
    lớp khác. Lớp vắng mặt bị bỏ qua thay vì lặp lại hàng của lớp khác.
    """
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    coefs = np.atleast_2d(model.coef_)
    classes = [int(value) for value in getattr(model, "classes_", LABELS)]
    result = {}
    for label in LABELS:
        if label not in classes:
            continue
        index = classes.index(label)
        if coefs.shape[0] == 1 and len(classes) == 2:
            # Bài toán hai lớp: scikit-learn chỉ lưu một hàng, ứng với lớp thứ
            # hai; lớp còn lại là hàng đó đổi dấu.
            row = coefs[0] if index == 1 else -coefs[0]
        elif index < coefs.shape[0]:
            row = coefs[index]
        else:  # pragma: no cover - chỉ xảy ra với mô hình có coef_ dị dạng
            continue
        order = np.argsort(row)[::-1][:top_n]
        result[label] = [(str(feature_names[j]), float(row[j])) for j in order]
    return result


# --------------------------------------------------------------------------- #
# Ghi kết quả
# --------------------------------------------------------------------------- #
def _git_commit():
    """Mã commit ngắn của mã nguồn sinh ra kết quả; None nếu không có git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _save_confusion_matrices(cm, results_dir):
    cm = np.asarray(cm, dtype=float)
    counts_path = os.path.join(results_dir, "confusion_matrix.png")
    norm_path = os.path.join(results_dir, "confusion_matrix_normalized.png")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm.astype(int),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
    )
    plt.title("Ma trận nhầm lẫn trên tập test (số lượng)")
    plt.ylabel("Nhãn thực tế")
    plt.xlabel("Nhãn dự đoán")
    plt.savefig(counts_path, bbox_inches="tight")
    plt.close()

    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
    )
    plt.title("Ma trận nhầm lẫn chuẩn hóa theo hàng (tỷ lệ recall)")
    plt.ylabel("Nhãn thực tế")
    plt.xlabel("Nhãn dự đoán")
    plt.savefig(norm_path, bbox_inches="tight")
    plt.close()
    return counts_path, norm_path


def _save_top_features(features, results_dir):
    path = os.path.join(results_dir, "top_features.txt")
    lines = [
        f"Top {TOP_FEATURES} n-gram có trọng số dương lớn nhất theo từng lớp (LinearSVC coef_)",
        "",
    ]
    for label in LABELS:
        lines.append(f"### {LABEL_NAMES[label]}")
        rows = features.get(label, [])
        if not rows:
            lines.append("(lớp này không xuất hiện trong tập huấn luyện)")
        for rank, (term, weight) in enumerate(rows, start=1):
            lines.append(f"{rank:2d}. {term:<30} {weight:+.4f}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def _save_model_comparison(records, results_dir):
    path = os.path.join(results_dir, "model_comparison.csv")
    columns = [
        "model",
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "cv_f1_macro_mean",
        "cv_f1_macro_std",
        "fit_seconds",
    ]
    pd.DataFrame(records, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _relative_path(path):
    """Ghi đường dẫn tương đối so với gốc dự án để kết quả không phụ thuộc máy."""
    try:
        rel = os.path.relpath(os.path.abspath(path), PROJECT_ROOT)
    except ValueError:
        return path
    return path if rel.startswith("..") else rel.replace(os.sep, "/")


def build_metrics(
    df,
    data_path,
    split_info,
    vectorizer,
    comparison,
    selected,
    test_result,
    robustness,
    n_dropped=0,
):
    """Gom toàn bộ chỉ số theo đúng lược đồ ở PLAN mục 4.4.

    `dataset.n_rows` là số dòng thực sự đưa vào mô hình (đã bỏ các dòng rỗng sau
    tiền xử lý), nhờ vậy luôn bằng n_train + n_test.
    """
    label_counts = df["label"].value_counts().sort_index()
    description = PREPROCESSING_DESC
    if n_dropped:
        description += f"; loại {n_dropped} dòng rỗng sau tiền xử lý"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "code_commit": _git_commit(),
        "dataset": {
            "path": _relative_path(data_path),
            "n_rows": int(len(df)),
            "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        },
        "preprocessing": description,
        "split": split_info,
        "features": {
            "vectorizer": "TfidfVectorizer",
            "ngram_range": list(TFIDF_PARAMS["ngram_range"]),
            "max_features": TFIDF_PARAMS["max_features"],
            "sublinear_tf": TFIDF_PARAMS["sublinear_tf"],
            "vocab_size": int(len(vectorizer.vocabulary_)),
        },
        "model_comparison": comparison,
        "selected_model": selected,
        "test": {k: v for k, v in test_result.items() if not k.startswith("_")},
        "seed_robustness": robustness,
    }


def _build_notes(metrics, quick):
    """Ghi chú của thẻ mô hình: nói đúng vai trò của cross-validation.

    Cross-validation không chọn ra họ mô hình (kế hoạch đã cố định LinearSVC),
    nó chỉ so sánh bốn mô hình nền rồi dò tham số C. Ghi như vậy để người đọc
    không hiểu nhầm bảng so sánh là căn cứ chọn mô hình.
    """
    if quick:
        return (
            "Chạy ở chế độ --quick: không so sánh mô hình nền, không dò tham số C "
            "(dùng C mặc định), không kiểm tra độ ổn định theo hạt giống. Tập test "
            "chỉ dùng đúng một lần."
        )
    selected = metrics["selected_model"]
    best = next(
        (row for row in selected.get("c_grid_results", [])
         if float(row["C"]) == float(selected["best_c"])),
        None,
    )
    logistic = next(
        (row for row in metrics.get("model_comparison", [])
         if row["model"] == "LogisticRegression"),
        None,
    )
    notes = (
        "So sánh 4 mô hình nền (MostFrequent, MultinomialNB, LogisticRegression, "
        "LinearSVC) bằng cross-validation 5-fold trên tập train; giữ LinearSVC theo "
        f"thiết kế rồi dò tham số C cũng bằng macro-F1 cross-validation (chọn "
        f"C={selected['best_c']:g})."
    )
    if best is not None and logistic is not None:
        gap = best["cv_f1_macro_mean"] - logistic["cv_f1_macro_mean"]
        spread = max(best["cv_f1_macro_std"], logistic["cv_f1_macro_std"])
        notes += (
            f" Tại C đã chọn, macro-F1 CV của LinearSVC là {best['cv_f1_macro_mean']:.4f} "
            f"± {best['cv_f1_macro_std']:.4f}, so với LogisticRegression "
            f"{logistic['cv_f1_macro_mean']:.4f} ± {logistic['cv_f1_macro_std']:.4f}"
        )
        # Chỉ được nói "ngang nhau" khi con số cho phép nói như vậy.
        if abs(gap) <= spread:
            notes += (
                f" (chênh lệch {gap:+.4f}, nhỏ hơn độ lệch chuẩn giữa các fold, "
                "nên không kết luận được mô hình nào tốt hơn)."
            )
        else:
            notes += f" (chênh lệch {gap:+.4f}, lớn hơn độ lệch chuẩn giữa các fold)."
    notes += (
        " Toàn bộ việc chọn lựa chỉ dùng tập train; tập test chỉ dùng đúng một lần. "
        "Lớp trung tính là lớp yếu nhất, xem results/metrics.json."
    )
    return notes


def build_model_card(metrics, notes):
    return {
        "model": "LinearSVC",
        "vectorizer": "TfidfVectorizer(1,2)-gram, 15000 features",
        "label_map": dict(LABEL_MAP),
        "trained_at": metrics["generated_at"],
        "sklearn_version": metrics["sklearn_version"],
        "preprocessing": metrics["preprocessing"],
        "test_accuracy": metrics["test"]["accuracy"],
        "test_f1_macro": metrics["test"]["f1_macro"],
        "notes": notes,
    }


def _dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    return path


# --------------------------------------------------------------------------- #
# Tóm tắt ra màn hình
# --------------------------------------------------------------------------- #
def print_summary(metrics, grid_records, quick):
    print("\n========== KẾT QUẢ ĐÁNH GIÁ ==========")
    split = metrics["split"]
    print(
        f"Chia dữ liệu: {split['n_train']} train / {split['n_test']} test "
        f"(phân tầng, random_state={split['random_state']})"
    )

    if metrics["model_comparison"]:
        print("\nSo sánh mô hình (cross-validation 5-fold trên tập train):")
        print(f"{'Mô hình':<30}{'CV accuracy':>20}{'CV macro-F1':>22}{'Thời gian khớp':>18}")
        for row in metrics["model_comparison"]:
            acc = f"{row['cv_accuracy_mean'] * 100:.2f}% ± {row['cv_accuracy_std'] * 100:.2f}"
            f1m = f"{row['cv_f1_macro_mean']:.4f} ± {row['cv_f1_macro_std']:.4f}"
            print(f"{row['model']:<30}{acc:>20}{f1m:>22}{row['fit_seconds']:>16.1f}s")
    else:
        print("\nSo sánh mô hình: bỏ qua (chế độ --quick).")

    if grid_records:
        print("\nDò tham số C cho LinearSVC (theo macro-F1 cross-validation):")
        for row in grid_records:
            print(
                f"  C={row['C']:<5} macro-F1 = {row['cv_f1_macro_mean']:.4f} "
                f"± {row['cv_f1_macro_std']:.4f} | accuracy = "
                f"{row['cv_accuracy_mean'] * 100:.2f}% ± {row['cv_accuracy_std'] * 100:.2f}"
            )
    print(f"C được chọn: {metrics['selected_model']['best_c']}")

    test = metrics["test"]
    print("\nTập test (đánh giá một lần duy nhất):")
    print(f"  Accuracy    : {test['accuracy'] * 100:.2f}%")
    print(f"  Macro-F1    : {test['f1_macro']:.4f}")
    print(f"  Weighted-F1 : {test['f1_weighted']:.4f}")
    print(f"  {'Lớp':<18}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")
    for label in LABELS:
        row = test["per_class"][str(label)]
        print(
            f"  {LABEL_NAMES[label]:<18}{row['precision']:>10.4f}{row['recall']:>10.4f}"
            f"{row['f1']:>10.4f}{row['support']:>10d}"
        )

    robustness = metrics["seed_robustness"]
    if robustness["test_f1_macro"]:
        accs = np.asarray(robustness["test_accuracy"])
        print(
            f"\nĐộ ổn định theo {len(robustness['seeds'])} hạt giống {robustness['seeds']}: "
            f"accuracy {accs.mean() * 100:.2f}% ± {accs.std() * 100:.2f}, "
            f"macro-F1 {robustness['f1_macro_mean']:.4f} ± {robustness['f1_macro_std']:.4f}"
        )
    else:
        print("\nĐộ ổn định theo hạt giống: bỏ qua (chế độ --quick).")

    if quick:
        print("\nLưu ý: chế độ --quick bỏ qua so sánh mô hình, dò C và kiểm tra hạt giống.")
    print("======================================\n")


# --------------------------------------------------------------------------- #
# Điểm vào
# --------------------------------------------------------------------------- #
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Huấn luyện và đánh giá mô hình phân loại cảm xúc 3 lớp."
    )
    parser.add_argument(
        "--data",
        default=os.path.join(PROJECT_ROOT, "data", "dataset_chuan.csv"),
        help="Đường dẫn file CSV có hai cột text và label.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Hạt giống cho lần chia train/test chính."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Bỏ qua so sánh mô hình, dò tham số C và kiểm tra độ ổn định theo hạt giống.",
    )
    parser.add_argument(
        "--out-models",
        default=os.path.join(PROJECT_ROOT, "models"),
        help="Thư mục lưu mô hình và thẻ mô hình.",
    )
    parser.add_argument(
        "--out-results",
        default=os.path.join(PROJECT_ROOT, "results"),
        help="Thư mục lưu chỉ số, biểu đồ và báo cáo.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Số tiến trình song song cho cross-validation (đặt 1 khi chạy kiểm thử).",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    started = time.perf_counter()

    models_dir = os.path.abspath(args.out_models)
    results_dir = os.path.abspath(args.out_results)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("--- BẮT ĐẦU HUẤN LUYỆN ---")
    df = load_dataset(args.data)
    print(f"Đã tải {len(df)} dòng từ {args.data}")

    print("Đang tiền xử lý (làm sạch và tách từ, chỉ chạy một lần)...")
    n_loaded = len(df)
    df = preprocess_corpus(df)
    n_dropped = n_loaded - len(df)
    if n_dropped:
        print(f"Đã loại {n_dropped} dòng rỗng sau tiền xử lý; còn {len(df)} dòng.")

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=args.seed, stratify=df["label"]
    )
    split_info = {
        "test_size": TEST_SIZE,
        "random_state": args.seed,
        "stratified": True,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }
    print(f"Chia dữ liệu: {split_info['n_train']} train / {split_info['n_test']} test")

    comparison = []
    grid_records = []
    best_c = 1.0
    if args.quick:
        print("Chế độ --quick: bỏ qua so sánh mô hình và dò tham số C.")
    else:
        print("Đang so sánh các mô hình bằng cross-validation 5-fold trên tập train...")
        comparison = compare_models(
            train_df["text_clean"], train_df["label"], seed=args.seed, n_jobs=args.n_jobs
        )
        print("Đang dò tham số C cho LinearSVC theo macro-F1...")
        best_c, grid_records = tune_linear_svc(
            train_df["text_clean"], train_df["label"], seed=args.seed, n_jobs=args.n_jobs
        )
        tuned_row = tuned_comparison_row(best_c, grid_records)
        if tuned_row is not None:
            comparison.append(tuned_row)

    print(f"Đang huấn luyện mô hình cuối (LinearSVC, C={best_c}) trên toàn bộ tập train...")
    vectorizer, model = fit_final_model(
        train_df["text_clean"], train_df["label"], best_c, seed=args.seed
    )

    print("Đang đánh giá trên tập test...")
    test_result = evaluate(model, vectorizer, test_df["text_clean"], test_df["label"])

    robustness = {
        "seeds": [],
        "test_accuracy": [],
        "test_f1_macro": [],
        "f1_macro_mean": 0.0,
        "f1_macro_std": 0.0,
    }
    if not args.quick:
        print(f"Đang kiểm tra độ ổn định theo các hạt giống {ROBUSTNESS_SEEDS}...")
        robustness = seed_robustness(df, best_c)

    selected = {
        "name": "LinearSVC",
        "params": {
            "C": float(best_c),
            "class_weight": "balanced",
            "dual": False,
            "random_state": args.seed,
        },
        "c_grid": [] if args.quick else list(C_GRID),
        "c_grid_results": grid_records,
        "best_c": float(best_c),
    }
    metrics = build_metrics(
        df,
        args.data,
        split_info,
        vectorizer,
        comparison,
        selected,
        test_result,
        robustness,
        n_dropped=n_dropped,
    )

    # Giữ nguyên tên hai file pickle để ứng dụng Streamlit tải được như trước.
    model_path = os.path.join(models_dir, "sentiment_model.pkl")
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)

    notes = _build_notes(metrics, args.quick)
    _dump_json(metrics, os.path.join(results_dir, "metrics.json"))
    _dump_json(build_model_card(metrics, notes), os.path.join(models_dir, "model_card.json"))

    with open(os.path.join(results_dir, "classification_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(test_result["_report_text"])
    _save_model_comparison(comparison, results_dir)
    _save_confusion_matrices(test_result["confusion_matrix"], results_dir)
    _save_top_features(top_features_per_class(model, vectorizer), results_dir)

    print_summary(metrics, grid_records, args.quick)
    print(f"Đã lưu mô hình: {model_path}")
    print(f"Đã lưu vectorizer: {vec_path}")
    print(f"Đã lưu chỉ số và biểu đồ tại: {results_dir}")
    print(f"--- HOÀN TẤT trong {time.perf_counter() - started:.1f}s ---")
    return metrics


if __name__ == "__main__":
    if sys.stdout is not None and getattr(sys.stdout, "encoding", None) != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
