"""Kiểm thử khói cho giao diện Streamlit: app phải mở được và chưa chạy gì cả.

Không gọi mạng, không nạp mô hình: AppTest chỉ thực thi phần dựng giao diện
(mọi tính toán nặng nằm sau nút "Bắt đầu phân tích").
"""

import os

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "app.py"
)


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
