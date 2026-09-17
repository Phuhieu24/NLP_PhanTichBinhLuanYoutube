"""Kiểm thử topic_pipeline: các hàm thuần dùng mô hình giả, không nạp mô hình thật."""

import pandas as pd
import pytest

import topic_pipeline as tp


class FakeTopicModel:
    """Mô hình giả chỉ cần trả về bảng thông tin chủ đề giống BERTopic."""

    def __init__(self, info):
        self._info = info

    def get_topic_info(self):
        return self._info.copy()


class DummyEmbedder:
    """Vật thể giả đóng vai mô hình nhúng câu: chỉ cần khác None là đủ."""

    def encode(self, docs, **kwargs):  # pragma: no cover - không được gọi trong kiểm thử
        raise AssertionError("Kiểm thử không được phép nạp mô hình nhúng thật.")


def make_info():
    return pd.DataFrame(
        {
            "Topic": [-1, 0, 1],
            "Count": [3, 5, 2],
            "Name": ["-1_ơi_à", "0_hay_bài_hát", "1_giọng_ấm"],
            "Representation": [
                ["ơi", "à", "ừ", "nha"],
                ["hay", "bài", "hát", "quá"],
                ["giọng", "hát", "ấm"],
            ],
            "Representative_Docs": [
                ["bình_luận nhiễu"],
                ["bài_hát này rất hay", "chuỗi không có trong docs"],
                ["giọng hát ấm_áp"],
            ],
        }
    )


def make_frame_and_docs():
    docs = ["bài_hát này rất hay", "giọng hát ấm_áp", "bình_luận nhiễu"]
    df = pd.DataFrame(
        {
            "text": ["Bài hát này RẤT HAY!!!", "Giọng hát ấm áp quá 😍", "Bình luận nhiễu"],
            "tokenized_text": docs,
        }
    )
    return df, docs


# --------------------------------------------------------------------------
# TopicConfig
# --------------------------------------------------------------------------


def test_topic_config_defaults():
    cfg = tp.TopicConfig()
    assert cfg.min_topic_size == 10
    assert cfg.nr_topics is None
    assert cfg.n_neighbors == 15
    assert cfg.n_components == 5
    assert cfg.top_n_words == 10
    assert cfg.random_state == 42
    assert cfg.use_stopwords is True
    assert cfg.cluster_selection_method == "eom"
    assert cfg.min_samples is None


# --------------------------------------------------------------------------
# prepare_documents (dùng preprocess thật)
# --------------------------------------------------------------------------


def test_prepare_documents_filters_and_reindexes():
    df = pd.DataFrame(
        {
            "text": [
                "Bài hát này rất hay và ý nghĩa",
                "Đông Hùng hát quá đỉnh luôn",
                None,
                "hay",
                "",
                "Chương trình làm tôi xúc động https://youtu.be/abc",
            ],
            "like_count": [10, 9, 8, 7, 6, 5],
        }
    )
    out = tp.prepare_documents(df)

    assert list(out.columns[-2:]) == ["clean_text", "tokenized_text"]
    # NaN, chuỗi rỗng và bình luận một token đều bị loại.
    assert len(out) == 3
    assert list(out.index) == [0, 1, 2]
    assert out["like_count"].tolist() == [10, 9, 5]
    assert all(len(text.split()) >= 2 for text in out["tokenized_text"])
    # Cột văn bản gốc không bị đụng tới.
    assert out.loc[0, "text"] == "Bài hát này rất hay và ý nghĩa"
    # URL bị loại khỏi văn bản đã làm sạch.
    assert "http" not in out.loc[2, "clean_text"].lower()


def test_prepare_documents_requires_text_column():
    with pytest.raises(KeyError):
        tp.prepare_documents(pd.DataFrame({"noi_dung": ["một hai ba"]}))


def test_prepare_documents_accepts_custom_column():
    df = pd.DataFrame({"noi_dung": ["Bài hát này rất hay", "x"]})
    out = tp.prepare_documents(df, text_col="noi_dung")
    assert len(out) == 1


def test_documents_hash_is_stable_and_sensitive():
    assert tp.documents_hash(["a", "b"]) == tp.documents_hash(["a", "b"])
    assert tp.documents_hash(["a", "b"]) != tp.documents_hash(["b", "a"])
    assert tp.documents_hash(["ab", "c"]) != tp.documents_hash(["a", "bc"])


# --------------------------------------------------------------------------
# topic_label_mapping / topic_table / outlier_share
# --------------------------------------------------------------------------


def test_topic_label_mapping_uses_keywords_and_marks_outliers():
    mapping = tp.topic_label_mapping(FakeTopicModel(make_info()))
    assert mapping[-1] == tp.OUTLIER_LABEL
    assert mapping[0] == "0: hay, bài, hát"
    assert mapping[1] == "1: giọng, hát, ấm"


def test_topic_label_mapping_respects_nr_words():
    mapping = tp.topic_label_mapping(FakeTopicModel(make_info()), nr_words=2)
    assert mapping[0] == "0: hay, bài"


def test_topic_label_mapping_falls_back_to_name():
    info = make_info()
    info["Representation"] = [[], [], []]
    mapping = tp.topic_label_mapping(FakeTopicModel(info))
    assert mapping[0] == "0_hay_bài_hát"


def test_topic_table_maps_representatives_back_to_original_text():
    df, docs = make_frame_and_docs()
    table = tp.topic_table(FakeTopicModel(make_info()), df, docs)

    assert list(table.columns) == ["Topic", "Count", "Name", "Keywords", "Representative"]
    assert table["Topic"].tolist() == [-1, 0, 1]
    assert table["Count"].tolist() == [3, 5, 2]

    topic0 = table[table["Topic"] == 0].iloc[0]
    # Tài liệu đại diện khớp vị trí -> trả về văn bản gốc còn nguyên hoa thường và dấu câu.
    assert topic0["Representative"][0] == "Bài hát này RẤT HAY!!!"
    # Tài liệu không tìm thấy trong docs -> giữ nguyên chuỗi đã tách từ thay vì báo lỗi.
    assert topic0["Representative"][1] == "chuỗi không có trong docs"
    assert topic0["Keywords"] == ["hay", "bài", "hát", "quá"]
    assert table[table["Topic"] == -1].iloc[0]["Name"] == tp.OUTLIER_LABEL


def test_topic_table_survives_missing_columns():
    info = make_info().drop(columns=["Representative_Docs", "Representation"])
    df, docs = make_frame_and_docs()
    table = tp.topic_table(FakeTopicModel(info), df, docs)
    assert table["Representative"].tolist() == [[], [], []]
    assert table["Keywords"].tolist() == [[], [], []]


def test_outlier_share():
    assert tp.outlier_share([]) == 0.0
    assert tp.outlier_share([-1, -1, 0, 0]) == 50.0
    assert tp.outlier_share([0, 1, 2]) == 0.0


# --------------------------------------------------------------------------
# build_topic_model (dựng đối tượng thật, không huấn luyện, không nạp mô hình nhúng)
# --------------------------------------------------------------------------


def test_build_topic_model_disables_english_preprocessing():
    cfg = tp.TopicConfig(
        min_topic_size=7,
        top_n_words=8,
        cluster_selection_method="leaf",
        min_samples=3,
        n_neighbors=12,
        n_components=4,
        random_state=7,
    )
    model = tp.build_topic_model(DummyEmbedder(), cfg)

    # Đây là bẫy chính: language != None thì BERTopic xóa sạch dấu tiếng Việt
    # trước khi tính c-TF-IDF.
    assert model.language is None
    assert model.embedding_model is not None
    assert model.vectorizer_model.token_pattern == r"(?u)\b[^\W\d_]\w*\b"
    assert model.hdbscan_model.min_cluster_size == 7
    assert model.hdbscan_model.min_samples == 3
    assert model.hdbscan_model.cluster_selection_method == "leaf"
    assert model.hdbscan_model.prediction_data is True
    assert model.umap_model.n_neighbors == 12
    assert model.umap_model.n_components == 4
    assert model.umap_model.random_state == 7
    assert model.umap_model.metric == "cosine"
    assert model.top_n_words == 8
    assert model.min_topic_size == 7
    assert model.calculate_probabilities is False


def test_build_topic_model_stopwords_toggle():
    off = tp.build_topic_model(DummyEmbedder(), tp.TopicConfig(use_stopwords=False))
    assert off.vectorizer_model.stop_words is None

    on = tp.build_topic_model(DummyEmbedder(), tp.TopicConfig(use_stopwords=True))
    stop_words = on.vectorizer_model.stop_words
    # Phải là danh sách từ dừng thật: nhận `None` ở đây thì mất hẳn tính năng lọc
    # từ dừng mà kiểm thử vẫn xanh.
    assert isinstance(stop_words, list) and len(stop_words) >= 150


def test_token_pattern_keeps_vietnamese_words_and_drops_bare_numbers():
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(token_pattern=tp.TOKEN_PATTERN)
    tokens = vectorizer.build_tokenizer()("không chương_trình được top4 10 5 _x")
    assert tokens == ["không", "chương_trình", "được", "top4"]


# --------------------------------------------------------------------------
# comments_from_dataframe (nguồn dữ liệu ngoại tuyến: CSV tải lên, tệp mẫu)
# --------------------------------------------------------------------------


def test_comments_from_dataframe_defaults_missing_columns():
    df = pd.DataFrame({"text": ["Bài hát rất hay", "Nghe chán quá"]})
    out = tp.comments_from_dataframe(df, "text")

    assert list(out.columns) == tp.COMMENT_COLUMNS
    assert out["comment_id"].tolist() == ["0", "1"]
    assert out["parent_id"].tolist() == [None, None]
    assert out["is_reply"].tolist() == [False, False]
    assert out["author"].tolist() == ["Ẩn danh", "Ẩn danh"]
    assert out["published_at"].tolist() == ["", ""]
    assert out["like_count"].tolist() == [0, 0]
    assert out["reply_count"].tolist() == [0, 0]
    assert out["text"].tolist() == ["Bài hát rất hay", "Nghe chán quá"]
    assert out["is_reply"].dtype == bool
    assert out["like_count"].dtype.kind == "i"
    assert out["reply_count"].dtype.kind == "i"


def test_comments_from_dataframe_keeps_columns_that_exist():
    df = pd.DataFrame(
        {
            "noi_dung": ["Bài hát rất hay", "Nghe chán quá"],
            "author": ["An", "Bình"],
            "like_count": ["7", None],
            "is_reply": [True, False],
            "parent_id": ["Ugx1", None],
            "published_at": ["2024-01-01T00:00:00Z", ""],
            "reply_count": [2, 0],
            "comment_id": ["Ugy0", "Ugy1"],
        }
    )
    out = tp.comments_from_dataframe(df, "noi_dung")

    assert out["comment_id"].tolist() == ["Ugy0", "Ugy1"]
    assert out["author"].tolist() == ["An", "Bình"]
    assert out["is_reply"].tolist() == [True, False]
    # Giá trị số hỏng hoặc thiếu quy về 0 chứ không làm hỏng kiểu dữ liệu.
    assert out["like_count"].tolist() == [7, 0]
    assert out["like_count"].dtype.kind == "i"
    assert out["text"].tolist() == ["Bài hát rất hay", "Nghe chán quá"]


def test_comments_from_dataframe_drops_empty_text_then_applies_limit():
    df = pd.DataFrame({"text": ["Bài hát rất hay", None, "   ", "Nghe chán quá", "Giọng ấm thật"]})
    out = tp.comments_from_dataframe(df, "text", limit=2)

    assert out["text"].tolist() == ["Bài hát rất hay", "Nghe chán quá"]
    # comment_id giữ chỉ số dòng gốc, nên vẫn dò ngược được về tệp đầu vào.
    assert out["comment_id"].tolist() == ["0", "3"]
    assert list(out.index) == [0, 1]


def test_comments_from_dataframe_limit_none_keeps_everything():
    df = pd.DataFrame({"text": [f"bình luận số {index}" for index in range(5)]})
    assert len(tp.comments_from_dataframe(df, "text")) == 5


def test_comments_from_dataframe_requires_the_text_column():
    with pytest.raises(ValueError) as error:
        tp.comments_from_dataframe(pd.DataFrame({"text": ["một hai"]}), "noi_dung")
    assert "noi_dung" in str(error.value)


def test_comments_from_dataframe_rejects_a_frame_without_usable_rows():
    df = pd.DataFrame({"text": [None, "", "  "]})
    with pytest.raises(ValueError):
        tp.comments_from_dataframe(df, "text")


def test_comments_from_dataframe_handles_a_header_read_with_bom():
    """Tệp CSV có BOM: người gọi đọc bằng utf-8-sig, ở đây chỉ nhận bảng đã sạch."""
    import io

    payload = "﻿text,label\nBài hát rất hay,2\nNghe chán quá,0\n".encode("utf-8")
    frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
    out = tp.comments_from_dataframe(frame, "text")

    assert list(frame.columns) == ["text", "label"]
    assert out["text"].tolist() == ["Bài hát rất hay", "Nghe chán quá"]


# --------------------------------------------------------------------------
# Bảng hiển thị: bảng tải về, chỉ số theo lớp, bảng so sánh và lưới C
# --------------------------------------------------------------------------


def _frame_for_export():
    return pd.DataFrame(
        {
            "comment_id": ["Ugy0", "Ugy1"],
            "published_at": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
            "is_reply": [False, True],
            "reply_count": [3, 0],
            "author": ["An", "Bình"],
            "like_count": [7, 1],
            "text": ["Bài hát rất hay", "Nghe chán quá"],
            "Topic_Name": ["0: hay", "1: chán"],
            "Sentiment_Label": ["🟢 Tích cực", "🔴 Tiêu cực"],
        }
    )


def test_comments_table_stays_compact():
    table = tp.comments_table(_frame_for_export())
    assert list(table.columns) == ["Chủ đề", "Cảm xúc", "Tác giả", "Bình luận", "Lượt thích", "Trả lời?"]


def test_comments_export_table_adds_the_traceable_columns():
    export = tp.comments_export_table(_frame_for_export())

    for title in ("Mã bình luận", "Thời điểm đăng", "Số trả lời", "Trả lời?"):
        assert title in export.columns
    assert export["Mã bình luận"].tolist() == ["Ugy0", "Ugy1"]
    assert export["Số trả lời"].tolist() == [3, 0]
    # Bảng hiển thị vẫn gọn như cũ.
    assert len(export.columns) == len(tp.comments_table(_frame_for_export()).columns) + 3


def test_comments_export_table_ignores_columns_that_are_missing():
    df = pd.DataFrame({"text": ["Bài hát rất hay"], "Topic_Name": ["0: hay"], "like_count": [1],
                       "author": ["An"]})
    export = tp.comments_export_table(df, use_sentiment=False)
    assert "Mã bình luận" not in export.columns
    assert "Cảm xúc" not in export.columns


def test_per_class_table_uses_vietnamese_headers_and_class_names():
    per_class = {
        "0": {"precision": 0.75, "recall": 0.76, "f1": 0.75, "support": 1400},
        "1": {"precision": 0.53, "recall": 0.54, "f1": 0.54, "support": 682},
        "2": {"precision": 0.87, "recall": 0.85, "f1": 0.86, "support": 1918},
    }
    table = tp.per_class_table(per_class)

    assert list(table.columns) == ["Lớp", "Precision", "Recall", "F1", "Số mẫu"]
    assert table["Lớp"].tolist() == ["Tiêu cực", "Trung tính", "Tích cực"]
    assert table["Số mẫu"].tolist() == [1400, 682, 1918]
    assert tp.per_class_table(None).empty


def _comparison_records():
    return [
        {"model": "LogisticRegression", "cv_accuracy_mean": 0.7625, "cv_accuracy_std": 0.0049,
         "cv_f1_macro_mean": 0.7170, "cv_f1_macro_std": 0.0054, "fit_seconds": 0.47},
        {"model": "LinearSVC (C=0.3, đã dò)", "cv_accuracy_mean": 0.7749, "cv_accuracy_std": 0.0038,
         "cv_f1_macro_mean": 0.7180, "cv_f1_macro_std": 0.0066, "fit_seconds": 0.34},
    ]


def _grid_records():
    return [
        {"C": 0.1, "cv_f1_macro_mean": 0.7124, "cv_f1_macro_std": 0.0066,
         "cv_accuracy_mean": 0.7703, "cv_accuracy_std": 0.0044, "fit_seconds": 0.30},
        {"C": 0.3, "cv_f1_macro_mean": 0.7180, "cv_f1_macro_std": 0.0066,
         "cv_accuracy_mean": 0.7749, "cv_accuracy_std": 0.0038, "fit_seconds": 0.34},
    ]


def test_comparison_frame_translates_headers_and_scales_accuracy():
    frame = tp.comparison_frame(_comparison_records())
    assert list(frame.columns) == ["Mô hình", "CV accuracy", "± acc", "CV macro-F1", "± F1",
                                   "Thời gian khớp (s)"]
    assert frame["CV accuracy"].tolist() == pytest.approx([76.25, 77.49])
    assert frame["CV macro-F1"].tolist() == pytest.approx([0.7170, 0.7180])


def test_comparison_frame_passes_through_an_unexpected_shape():
    frame = tp.comparison_frame([{"model": "X"}])
    assert list(frame.columns) == ["model"]


def test_c_grid_frame_keeps_the_grid_readable():
    frame = tp.c_grid_frame(_grid_records())
    assert list(frame.columns) == ["C", "CV macro-F1", "± F1", "CV accuracy"]
    assert frame["C"].tolist() == [0.1, 0.3]
    assert frame["CV accuracy"].tolist() == pytest.approx([77.03, 77.49])


def test_c_grid_note_calls_a_tie_a_tie():
    metrics = {"selected_model": {"c_grid_results": _grid_records()},
               "model_comparison": _comparison_records()}
    note = tp.c_grid_note(metrics)
    assert "C=0.3" in note
    assert "+0.0010" in note
    assert "ngang nhau" in note


def test_c_grid_note_reports_a_real_gap_as_a_gap():
    metrics = {"selected_model": {"c_grid_results": _grid_records()},
               "model_comparison": [{"model": "LogisticRegression", "cv_f1_macro_mean": 0.60,
                                     "cv_f1_macro_std": 0.001}]}
    note = tp.c_grid_note(metrics)
    assert "ngang nhau" not in note
    assert "+0.1180" in note


def test_c_grid_note_is_empty_without_grid_results():
    assert tp.c_grid_note({"selected_model": {}}) == ""
    assert tp.c_grid_note({}) == ""


def test_topic_config_line_spells_out_the_defaults():
    line = tp.topic_config_line({"min_topic_size": 15, "min_samples": 1, "nr_topics": None,
                                 "cluster_selection_method": "eom", "use_stopwords": True})
    assert line == "min_topic_size=15 · min_samples=1 · eom · stopwords=bật · nr_topics=tự động"

    other = tp.topic_config_line({"min_topic_size": 10, "min_samples": None, "nr_topics": 12,
                                  "cluster_selection_method": "leaf", "use_stopwords": False})
    assert other == "min_topic_size=10 · min_samples=mặc định · leaf · stopwords=tắt · nr_topics=12"
    assert tp.topic_config_line({}) == ""


def test_short_label_trims_long_topic_names_but_keeps_the_id_prefix():
    name = "12: chương_trình, đội_trưởng, đoàn_kết, dễ_thương"
    short = tp.short_label(name)
    assert len(short) <= tp.MAX_LABEL_CHARS
    assert short.startswith("12: chương_trình")
    assert short.endswith("…")
    assert tp.short_label("0: hay, bài") == "0: hay, bài"


# --------------------------------------------------------------------------
# Công cụ dòng lệnh
# --------------------------------------------------------------------------


def test_cli_defaults_to_the_raw_crawler_output():
    import topic_model

    args = topic_model.build_parser().parse_args([])
    assert args.input.replace("\\", "/") == "data/comments.csv"
    assert args.text_col == "text"

    # `--help` phải nói ra bộ tham số khuyến nghị; so trên description vì
    # argparse ngắt dòng phần in ra màn hình.
    description = topic_model.build_parser().description
    assert "--min-topic-size 15 --min-samples 1" in description
    assert "Khuyến nghị" in topic_model.build_parser().format_help()
