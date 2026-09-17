"""Kiểm thử khói cho giao diện Streamlit: app phải mở được và chưa chạy gì cả.

Phần lớn kiểm thử ở đây không gọi mạng và không nạp mô hình: AppTest chỉ thực
thi phần dựng giao diện (mọi tính toán nặng nằm sau nút "Bắt đầu phân tích").
Riêng kiểm thử nguồn "Dữ liệu mẫu" chạy trọn pipeline trên 100 dòng của
`data/dataset_chuan.csv`, dùng mô hình nhúng câu đã có sẵn trong cache máy.
"""

import os

# Bắt buộc mô hình nhúng câu đọc từ cache: kiểm thử không được gọi mạng. Phải
# đặt trước khi huggingface_hub được import lần đầu thì mới có tác dụng.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(PROJECT_ROOT, "src", "app.py")
SAMPLE_PATH = os.path.join(PROJECT_ROOT, "data", "dataset_chuan.csv")
SOURCE_CHOICES = ["Link YouTube", "Tệp CSV", "Dữ liệu mẫu"]


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(APP_PATH, default_timeout=60).run(timeout=60)


def fake_result():
    """Gói kết quả tối thiểu, đủ để dựng cả 5 tab mà không cần nạp mô hình nào."""
    df = pd.DataFrame(
        {
            "text": ["Bài hát rất hay", "Nghe chán quá", "Cũng bình thường", "Giọng ấm thật"],
            "tokenized_text": ["bài_hát rất hay", "nghe chán quá", "cũng bình_thường", "giọng ấm thật"],
            "author": ["a", "b", "c", "d"],
            "like_count": [10, 2, 1, 7],
            "is_reply": [False, False, True, False],
            "Topic": [0, 1, -1, 0],
            "Topic_Name": ["0: hay, bài", "1: chán", "-1: Nhiễu", "0: hay, bài"],
            "Sentiment": [2, 0, 1, 2],
            "Sentiment_Label": ["🟢 Tích cực", "🔴 Tiêu cực", "⚪ Trung tính", "🟢 Tích cực"],
        }
    )
    topics = pd.DataFrame(
        {
            "Topic": [-1, 0, 1],
            "Count": [1, 2, 1],
            "Name": ["-1: Nhiễu", "0: hay, bài", "1: chán"],
            "Keywords": [["ơi"], ["hay", "bài"], ["chán"]],
            "Representative": [["Cũng bình thường"], ["Bài hát rất hay"], ["Nghe chán quá"]],
        }
    )
    return {
        "video_id": "abcdefghijk",
        "source": "YouTube: abcdefghijk",
        "metadata": None,
        "df": df,
        "topics": topics,
        "summaries": {},
        "figures": {"barchart": None, "intertopic": None},
        "timings": {"thu thập": 1.0, "BERTopic": 2.0},
        "config": {"min_topic_size": 10},
        "use_sentiment": True,
        "n_raw": 6,
    }


@pytest.mark.slow
def test_app_renders_without_exception(app):
    assert not app.exception, [str(error) for error in app.exception]


@pytest.mark.slow
def test_sidebar_has_url_input_and_buttons(app):
    labels = [element.label for element in app.sidebar.text_input]
    assert "Link video YouTube" in labels

    buttons = [element.label for element in app.button]
    assert "Bắt đầu phân tích" in buttons
    assert "Xóa kết quả" in buttons

    sliders = [element.label for element in app.sidebar.slider]
    assert "Số bình luận tối đa" in sliders


@pytest.mark.slow
def test_tabs_are_not_rendered_before_a_run(app):
    assert len(app.tabs) == 0
    assert any("Bắt đầu phân tích" in element.value for element in app.info)


@pytest.mark.slow
def test_clicking_run_without_url_shows_vietnamese_error():
    app = AppTest.from_file(APP_PATH, default_timeout=60).run(timeout=60)
    target = [element for element in app.button if element.label == "Bắt đầu phân tích"][0]
    target.click().run(timeout=60)

    assert not app.exception, [str(error) for error in app.exception]
    assert any("link video" in element.value.lower() for element in app.error)
    assert len(app.tabs) == 0


@pytest.mark.slow
def test_results_survive_a_widget_interaction():
    """Lỗi cũ: mọi thao tác trên giao diện đều xóa sạch kết quả và bắt cào lại."""
    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["result"] = fake_result()
    app.run(timeout=60)

    assert not app.exception, [str(error) for error in app.exception]
    assert len(app.tabs) == 5
    assert [tab.label for tab in app.tabs] == ["Tổng quan", "Chủ đề", "Cảm xúc", "Dữ liệu", "Mô hình"]

    toggle = [element for element in app.sidebar.toggle if "replies" in element.label][0]
    toggle.set_value(not toggle.value).run(timeout=60)

    assert not app.exception, [str(error) for error in app.exception]
    assert "result" in app.session_state
    assert len(app.tabs) == 5


@pytest.mark.slow
def test_all_tabs_render_with_metadata_and_llm_summaries():
    """Đường đi đầy đủ: thẻ video, chỉ số, bảng chủ đề kèm tóm tắt, bộ lọc, tab mô hình."""
    bundle = fake_result()
    bundle["metadata"] = {
        "title": "Anh Trai Say Hi tập 10",
        "channel_title": "VieChannel",
        "published_at": "2024-01-01T00:00:00Z",
        "view_count": 1234567,
        "like_count": 4321,
        "comment_count": 9876,
        "thumbnail_url": "https://i.ytimg.com/vi/abcdefghijk/hqdefault.jpg",
    }
    bundle["summaries"] = {0: "Người xem khen phần trình diễn.", 1: "Người xem chê phần dàn dựng."}

    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["result"] = bundle
    app.run(timeout=60)

    assert not app.exception, [str(error) for error in app.exception]
    metrics = {element.label: element.value for element in app.metric}
    assert metrics["Bình luận thu được"] == "6"
    assert metrics["Hợp lệ sau lọc"] == "4"
    assert metrics["Số chủ đề"] == "2"
    assert metrics["Nhiễu (-1)"] == "25.0%"
    assert metrics["Tích cực"] == "50.0%"
    assert len(app.get("download_button")) == 1
    assert any("Anh Trai Say Hi" in element.value for element in app.markdown)
    assert any("Người xem khen phần trình diễn." in element.value for element in app.info)


@pytest.mark.slow
def test_clear_button_drops_the_stored_result():
    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["result"] = fake_result()
    app.run(timeout=60)
    assert len(app.tabs) == 5

    clear = [element for element in app.button if element.label == "Xóa kết quả"][0]
    clear.click().run(timeout=60)

    assert not app.exception, [str(error) for error in app.exception]
    assert "result" not in app.session_state
    assert len(app.tabs) == 0


# --------------------------------------------------------------------------
# Nguồn dữ liệu: link YouTube, tệp CSV, dữ liệu mẫu
# --------------------------------------------------------------------------


def _slider(app, label):
    return [element for element in app.sidebar.slider if element.label == label][0]


@pytest.mark.slow
def test_source_radio_offers_three_modes_and_defaults_to_youtube(app):
    radios = [element for element in app.sidebar.radio if element.label == "Nguồn"]
    assert len(radios) == 1
    assert list(radios[0].options) == SOURCE_CHOICES
    assert radios[0].value == "Link YouTube"


@pytest.mark.slow
def test_sample_source_shows_the_note_and_hides_the_url_input():
    app = AppTest.from_file(APP_PATH, default_timeout=60).run(timeout=60)
    app.sidebar.radio[0].set_value("Dữ liệu mẫu").run(timeout=60)

    assert not app.exception, [str(error) for error in app.exception]
    assert "Link video YouTube" not in [element.label for element in app.sidebar.text_input]
    captions = [element.value for element in app.sidebar.caption]
    assert any("tập huấn luyện" in value for value in captions)
    assert any("chủ đề" in value for value in captions)


@pytest.mark.slow
def test_sample_source_runs_the_pipeline_end_to_end():
    """Chạy thật: 100 dòng dữ liệu mẫu, không cần API key, không gọi mạng."""
    assert os.path.exists(SAMPLE_PATH), "Thiếu data/dataset_chuan.csv"

    app = AppTest.from_file(APP_PATH, default_timeout=600)
    app.run(timeout=600)
    app.sidebar.radio[0].set_value("Dữ liệu mẫu").run(timeout=600)
    _slider(app, "Số bình luận tối đa").set_value(100).run(timeout=600)
    [element for element in app.button if element.label == "Bắt đầu phân tích"][0].click().run(timeout=600)

    assert not app.exception, [str(error) for error in app.exception]
    assert not app.error, [element.value for element in app.error]
    assert [tab.label for tab in app.tabs] == ["Tổng quan", "Chủ đề", "Cảm xúc", "Dữ liệu", "Mô hình"]

    result = app.session_state["result"]
    assert result["n_raw"] == 100
    assert result["metadata"] is None
    assert result["source"].startswith("Dữ liệu mẫu: data/dataset_chuan.csv")
    assert any(result["source"] in element.value for element in app.caption)

    # Với 100 tài liệu, BERTopic có thể không tách được chủ đề nào; khi đó giao
    # diện vẫn phải dựng đủ 5 tab và hiện một ghi chú tiếng Việt thay cho biểu đồ.
    n_topics = int((result["topics"]["Topic"] != -1).sum())
    if n_topics == 0:
        assert any("chủ đề" in element.value.lower() for element in app.info)


@pytest.mark.slow
def test_tabs_still_render_when_no_topic_was_formed():
    """Đường đi xấu nhất: HDBSCAN xếp mọi bình luận vào nhóm nhiễu."""
    bundle = fake_result()
    bundle["source"] = "CSV: binh_luan.csv"
    bundle["df"]["Topic"] = -1
    bundle["df"]["Topic_Name"] = "-1: Nhiễu"
    bundle["topics"] = pd.DataFrame(
        {
            "Topic": [-1],
            "Count": [4],
            "Name": ["-1: Nhiễu"],
            "Keywords": [["hay"]],
            "Representative": [["Bài hát rất hay"]],
        }
    )

    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["result"] = bundle
    app.run(timeout=60)

    assert not app.exception, [str(error) for error in app.exception]
    assert len(app.tabs) == 5
    assert any("Chưa tách được chủ đề nào" in element.value for element in app.info)
    assert any("CSV: binh_luan.csv" in element.value for element in app.caption)


@pytest.mark.slow
def test_advanced_defaults_follow_the_ablation(app):
    """Mặc định của thanh bên là 15/1, không phải mặc định của BERTopic."""
    assert _slider(app, "Kích thước cụm tối thiểu").value == 15

    numbers = [element for element in app.sidebar.number_input if element.label == "min_samples"]
    assert len(numbers) == 1
    assert numbers[0].value == 1

    helps = [element.help for element in app.sidebar.slider
             if element.label == "Kích thước cụm tối thiểu"]
    assert any("91%" in str(value) for value in helps)


def _fake_youtube_comments(n_rows=60):
    """Bảng đúng 8 cột hợp đồng của crawler, văn bản lấy từ tệp dữ liệu mẫu."""
    source = pd.read_csv(SAMPLE_PATH, nrows=n_rows, encoding="utf-8-sig")
    texts = source["text"].astype(str).tolist()
    return pd.DataFrame(
        {
            "comment_id": [f"Ugy{index:04d}" for index in range(len(texts))],
            "parent_id": [None] * len(texts),
            "is_reply": [False] * len(texts),
            "author": [f"Người xem {index}" for index in range(len(texts))],
            "published_at": ["2024-05-01T10:00:00Z"] * len(texts),
            "like_count": [index % 7 for index in range(len(texts))],
            "reply_count": [0] * len(texts),
            "text": texts,
        }
    )


@pytest.mark.slow
def test_youtube_source_runs_end_to_end_with_a_stubbed_crawler(monkeypatch):
    """Đường đi tốn kém nhất: link YouTube, crawler giả, mô hình thật trên máy."""
    import crawler

    frame = _fake_youtube_comments()
    metadata = {
        "title": "Anh Trai Say Hi tập 10",
        "channel_title": "VieChannel",
        "published_at": "2024-05-01T00:00:00Z",
        "view_count": 1234567,
        "like_count": 4321,
        "comment_count": 9876,
        "thumbnail_url": "",
    }
    monkeypatch.setattr(crawler, "get_video_comments", lambda *args, **kwargs: frame.copy())
    monkeypatch.setattr(crawler, "get_video_metadata", lambda *args, **kwargs: dict(metadata))
    # App đọc khóa API từ biến môi trường; đặt sẵn để không phụ thuộc tệp .env.
    monkeypatch.setenv("YOUTUBE_API_KEY", "khoa-gia-danh-cho-kiem-thu")

    app = AppTest.from_file(APP_PATH, default_timeout=600)
    app.run(timeout=600)
    url_input = [element for element in app.sidebar.text_input
                 if element.label == "Link video YouTube"][0]
    url_input.set_value("https://www.youtube.com/watch?v=abcdefghijk").run(timeout=600)
    [element for element in app.button if element.label == "Bắt đầu phân tích"][0].click().run(timeout=600)

    assert not app.exception, [str(error) for error in app.exception]
    assert not app.error, [element.value for element in app.error]
    assert "result" in app.session_state

    result = app.session_state["result"]
    assert result["video_id"] == "abcdefghijk"
    assert result["n_raw"] == len(frame)
    assert result["source"] == "YouTube: abcdefghijk"
    assert result["metadata"]["title"] == metadata["title"]
    assert [tab.label for tab in app.tabs] == ["Tổng quan", "Chủ đề", "Cảm xúc", "Dữ liệu", "Mô hình"]
    # Thẻ video được dựng từ dữ liệu metadata giả.
    assert any(metadata["title"] in element.value for element in app.markdown)
    assert any("VieChannel" in element.value for element in app.caption)
