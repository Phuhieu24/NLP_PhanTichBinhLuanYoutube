"""Kiểm thử quy trình huấn luyện cảm xúc trên một bộ dữ liệu giả nhỏ.

Không đụng tới `data/`, `models/`, `results/` của dự án: mọi thứ ghi vào thư mục
tạm của pytest. Chạy ở chế độ --quick và n_jobs=1 để giữ thời gian dưới 20 giây.
"""

import json
import os

import pandas as pd
import pytest

import train_sentiment as ts

# Mỗi lớp 30 câu ngắn kiểu bình luận YouTube tiếng Việt, đủ tín hiệu để mô hình
# học được mà vẫn chạy rất nhanh.
NEGATIVE = [
    "chương trình dở tệ xem chán quá",
    "tiết mục này thất vọng thật sự",
    "giọng hát yếu nghe không nổi",
    "dàn dựng cẩu thả rất tệ",
    "xem mà buồn ngủ chán ghê",
    "kịch bản nhạt nhẽo thất vọng",
    "âm thanh rè nghe tệ lắm",
    "biên tập kém xem thấy chán",
    "màn trình diễn tệ hại quá",
    "phần thi này dở không xem nữa",
]
NEUTRAL = [
    "chương trình phát sóng lúc tám giờ tối",
    "tập này dài khoảng chín mươi phút",
    "có bao nhiêu thí sinh tham gia vậy",
    "ai biết tên bài hát này không",
    "xem trên kênh nào vậy mọi người",
    "tập sau chiếu vào thứ bảy phải không",
    "danh sách tiết mục ở đâu vậy",
    "bản thu âm có trên ứng dụng nhạc chưa",
    "ban giám khảo gồm những ai vậy",
    "thời lượng tập này bao lâu vậy",
]
POSITIVE = [
    "tiết mục quá hay tôi rất thích",
    "giọng hát tuyệt vời nghe cuốn lắm",
    "sân khấu đẹp xuất sắc luôn",
    "xem đi xem lại vẫn thấy hay",
    "phần trình diễn đỉnh thật sự",
    "dàn dựng công phu rất đáng khen",
    "nghe xong thấy vui cả ngày",
    "bài này hay quá nghe mãi không chán",
    "ủng hộ chương trình hết mình luôn",
    "cảm động rơi nước mắt quá hay",
]
SUFFIXES = ["", " nha mọi người", " thật đó"]


def _make_dataframe():
    rows = []
    for label, samples in ((0, NEGATIVE), (1, NEUTRAL), (2, POSITIVE)):
        for text in samples:
            for suffix in SUFFIXES:
                rows.append({"text": text + suffix, "label": label})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory):
    """Chạy toàn bộ `main` một lần ở chế độ --quick và trả về đường dẫn kết quả."""
    base = tmp_path_factory.mktemp("train")
    data_path = base / "tiny.csv"
    models_dir = base / "models"
    results_dir = base / "results"
    df = _make_dataframe()
    df.to_csv(data_path, index=False, encoding="utf-8")

    metrics = ts.main(
        [
            "--data",
            str(data_path),
            "--out-models",
            str(models_dir),
            "--out-results",
            str(results_dir),
            "--quick",
            "--n-jobs",
            "1",
        ]
    )
    return {
        "df": df,
        "metrics": metrics,
        "models_dir": str(models_dir),
        "results_dir": str(results_dir),
        "data_path": str(data_path),
    }


def test_dataset_is_balanced_three_class():
    df = _make_dataframe()
    assert len(df) == 90
    assert sorted(df["label"].unique().tolist()) == [0, 1, 2]


def test_artifacts_written(quick_run):
    models_dir = quick_run["models_dir"]
    results_dir = quick_run["results_dir"]
    for name in ("sentiment_model.pkl", "tfidf_vectorizer.pkl", "model_card.json"):
        assert os.path.getsize(os.path.join(models_dir, name)) > 0, name
    for name in (
        "metrics.json",
        "classification_report.txt",
        "model_comparison.csv",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "top_features.txt",
    ):
        assert os.path.getsize(os.path.join(results_dir, name)) > 0, name


def test_metrics_json_matches_schema(quick_run):
    with open(os.path.join(quick_run["results_dir"], "metrics.json"), encoding="utf-8") as fh:
        metrics = json.load(fh)

    assert set(metrics) == {
        "generated_at",
        "sklearn_version",
        "code_commit",
        "dataset",
        "preprocessing",
        "split",
        "features",
        "model_comparison",
        "selected_model",
        "test",
        "seed_robustness",
    }
    assert set(metrics["dataset"]) == {"path", "n_rows", "label_counts"}
    assert set(metrics["split"]) == {
        "test_size",
        "random_state",
        "stratified",
        "n_train",
        "n_test",
    }
    assert set(metrics["features"]) == {
        "vectorizer",
        "ngram_range",
        "max_features",
        "sublinear_tf",
        "vocab_size",
    }
    assert set(metrics["selected_model"]) == {
        "name",
        "params",
        "c_grid",
        "c_grid_results",
        "best_c",
    }
    assert set(metrics["test"]) == {
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "per_class",
        "confusion_matrix",
        "labels",
    }
    assert set(metrics["seed_robustness"]) == {
        "seeds",
        "test_accuracy",
        "test_f1_macro",
        "f1_macro_mean",
        "f1_macro_std",
    }
    assert set(metrics["test"]["per_class"]) == {"0", "1", "2"}
    for stats in metrics["test"]["per_class"].values():
        assert set(stats) == {"precision", "recall", "f1", "support"}
    assert metrics["features"]["ngram_range"] == [1, 2]
    assert metrics["split"]["stratified"] is True
    assert metrics["split"]["random_state"] == 42
    assert metrics["test"]["labels"] == ["Tiêu cực (0)", "Trung tính (1)", "Tích cực (2)"]


def test_quick_mode_leaves_optional_sections_empty(quick_run):
    metrics = quick_run["metrics"]
    assert metrics["model_comparison"] == []
    assert metrics["seed_robustness"]["seeds"] == []
    assert metrics["seed_robustness"]["test_accuracy"] == []
    assert metrics["seed_robustness"]["test_f1_macro"] == []
    assert metrics["selected_model"]["best_c"] == 1.0
    assert metrics["selected_model"]["c_grid_results"] == []


def test_confusion_matrix_is_three_by_three_and_totals_test_size(quick_run):
    metrics = quick_run["metrics"]
    cm = metrics["test"]["confusion_matrix"]
    assert len(cm) == 3 and all(len(row) == 3 for row in cm)
    assert sum(sum(row) for row in cm) == metrics["split"]["n_test"]
    assert metrics["split"]["n_train"] + metrics["split"]["n_test"] == len(quick_run["df"])
    support = sum(stats["support"] for stats in metrics["test"]["per_class"].values())
    assert support == metrics["split"]["n_test"]
    # Số dòng của bộ dữ liệu luôn khớp với tổng hai phần đã chia.
    assert metrics["dataset"]["n_rows"] == metrics["split"]["n_train"] + metrics["split"]["n_test"]


def test_relative_path_is_used_for_files_inside_the_project():
    inside = os.path.join(ts.PROJECT_ROOT, "data", "dataset_chuan.csv")
    assert ts._relative_path(inside) == "data/dataset_chuan.csv"
    outside = os.path.join(os.sep, "tmp", "somewhere", "tiny.csv")
    assert ts._relative_path(outside) == outside


def test_model_card_label_map_and_numbers(quick_run):
    with open(os.path.join(quick_run["models_dir"], "model_card.json"), encoding="utf-8") as fh:
        card = json.load(fh)
    assert card["label_map"] == {"0": "Tiêu cực", "1": "Trung tính", "2": "Tích cực"}
    assert card["model"] == "LinearSVC"
    assert card["test_accuracy"] == pytest.approx(quick_run["metrics"]["test"]["accuracy"])
    assert card["test_f1_macro"] == pytest.approx(quick_run["metrics"]["test"]["f1_macro"])


def test_model_comparison_csv_has_schema_header_even_when_quick(quick_run):
    frame = pd.read_csv(os.path.join(quick_run["results_dir"], "model_comparison.csv"))
    assert list(frame.columns) == [
        "model",
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "cv_f1_macro_mean",
        "cv_f1_macro_std",
        "fit_seconds",
    ]
    assert frame.empty


def test_top_features_file_lists_every_class(quick_run):
    with open(os.path.join(quick_run["results_dir"], "top_features.txt"), encoding="utf-8") as fh:
        content = fh.read()
    for name in ts.LABEL_NAMES:
        assert f"### {name}" in content


def test_preprocess_corpus_drops_empty_rows():
    df = pd.DataFrame({"text": ["chương trình rất hay", "   ", None], "label": [2, 1, 0]})
    out = ts.preprocess_corpus(df)
    assert len(out) == 1
    assert out.loc[0, "text_clean"].strip() != ""


def test_compare_models_reports_every_candidate_with_both_scores():
    df = ts.preprocess_corpus(_make_dataframe())
    records = ts.compare_models(df["text_clean"], df["label"], n_jobs=1)
    assert [r["model"] for r in records] == [
        "MostFrequent",
        "MultinomialNB",
        "LogisticRegression",
        "LinearSVC",
    ]
    for row in records:
        assert 0.0 <= row["cv_accuracy_mean"] <= 1.0
        assert 0.0 <= row["cv_f1_macro_mean"] <= 1.0
    dummy = records[0]["cv_f1_macro_mean"]
    assert records[-1]["cv_f1_macro_mean"] > dummy


def test_tune_linear_svc_picks_best_macro_f1():
    """Dò C phải chọn đúng giá trị có macro-F1 cross-validation cao nhất."""
    df = ts.preprocess_corpus(_make_dataframe())
    best_c, records = ts.tune_linear_svc(
        df["text_clean"], df["label"], c_grid=[0.05, 3.0], seed=42, n_jobs=1
    )

    assert [row["C"] for row in records] == [0.05, 3.0]
    for row in records:
        assert set(row) == {
            "C",
            "cv_f1_macro_mean",
            "cv_f1_macro_std",
            "cv_accuracy_mean",
            "cv_accuracy_std",
            "fit_seconds",
        }
        assert 0.0 <= row["cv_f1_macro_mean"] <= 1.0
        assert 0.0 <= row["cv_accuracy_mean"] <= 1.0
    assert best_c == max(records, key=lambda row: row["cv_f1_macro_mean"])["C"]


def test_tuned_comparison_row_reuses_the_numbers_of_the_selected_c():
    """Dòng thứ năm của bảng so sánh phải là chính cấu hình được xuất xưởng."""
    grid = [
        {"C": 0.1, "cv_f1_macro_mean": 0.70, "cv_f1_macro_std": 0.01,
         "cv_accuracy_mean": 0.75, "cv_accuracy_std": 0.02, "fit_seconds": 1.5},
        {"C": 0.3, "cv_f1_macro_mean": 0.72, "cv_f1_macro_std": 0.006,
         "cv_accuracy_mean": 0.76, "cv_accuracy_std": 0.01, "fit_seconds": 1.6},
    ]
    row = ts.tuned_comparison_row(0.3, grid)

    assert row["model"] == "LinearSVC (C=0.3, đã dò)"
    assert row["cv_f1_macro_mean"] == 0.72
    assert row["cv_accuracy_std"] == 0.01
    assert row["fit_seconds"] == 1.6
    assert ts.tuned_comparison_row(0.3, []) is None


def test_seed_robustness_returns_one_number_per_seed():
    df = ts.preprocess_corpus(_make_dataframe())
    result = ts.seed_robustness(df, 1.0, seeds=[0, 1])

    assert result["seeds"] == [0, 1]
    assert len(result["test_accuracy"]) == 2
    assert len(result["test_f1_macro"]) == 2
    assert result["f1_macro_mean"] == pytest.approx(
        sum(result["test_f1_macro"]) / 2, abs=1e-9
    )
    assert result["f1_macro_std"] >= 0.0


def _frame_without_the_neutral_class():
    rows = []
    for label, samples in ((0, NEGATIVE), (2, POSITIVE)):
        for text in samples:
            for suffix in SUFFIXES:
                rows.append({"text": text + suffix, "label": label})
    return pd.DataFrame(rows)


def test_top_features_per_class_follows_model_classes(tmp_path):
    """Thiếu một lớp trong tập huấn luyện thì không được gán nhầm trọng số."""
    df = ts.preprocess_corpus(_frame_without_the_neutral_class())
    vectorizer, model = ts.fit_final_model(df["text_clean"], df["label"], 1.0)
    assert sorted(int(value) for value in model.classes_) == [0, 2]

    features = ts.top_features_per_class(model, vectorizer, top_n=5)

    # Lớp trung tính vắng mặt: bỏ qua hẳn thay vì lặp lại hàng của lớp khác.
    assert set(features) == {0, 2}
    terms_negative = [term for term, _ in features[0]]
    terms_positive = [term for term, _ in features[2]]
    assert terms_negative != terms_positive

    path = ts._save_top_features(features, str(tmp_path))
    content = open(path, encoding="utf-8").read()
    for name in ts.LABEL_NAMES:
        assert content.count(f"### {name}") == 1
    assert "(lớp này không xuất hiện trong tập huấn luyện)" in content


def test_top_features_per_class_gives_each_class_its_own_row():
    df = ts.preprocess_corpus(_make_dataframe())
    vectorizer, model = ts.fit_final_model(df["text_clean"], df["label"], 1.0)
    features = ts.top_features_per_class(model, vectorizer, top_n=5)

    assert set(features) == {0, 1, 2}
    rows = [tuple(term for term, _ in features[label]) for label in (0, 1, 2)]
    assert len(set(rows)) == 3
