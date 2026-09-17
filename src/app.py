"""Giao diện Streamlit: dán link YouTube, nhận về chủ đề và cảm xúc bình luận.

Luồng: thu thập -> làm sạch và tách từ -> nhúng câu bằng `keepitreal/vietnamese-sbert`
-> gom cụm bằng BERTopic -> phân loại cảm xúc bằng LinearSVC + TF-IDF -> (tùy chọn)
tóm tắt chủ đề bằng Ollama. Kết quả nằm trong `st.session_state` nên mọi thao tác
trên giao diện (đổi tab, lọc, tải CSV) không làm chạy lại phần tính toán nặng.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from dotenv import load_dotenv

import crawler
import topic_pipeline as tp
from llm_summary import DEFAULT_BASE_URL, DEFAULT_MODEL, OllamaSummarizer

st.set_page_config(layout="wide", page_title="Phân tích bình luận YouTube", page_icon="🎬")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
ORDER_CHOICES = {"Phổ biến": "relevance", "Mới nhất": "time"}

# Bản `crawler.py` cũ chưa có hai tên này; giữ cho app chạy được với cả hai bản.
CrawlError = getattr(crawler, "CrawlError", Exception)
_get_video_metadata = getattr(crawler, "get_video_metadata", None)


# --- Nạp mô hình và lấy dữ liệu (có bộ nhớ đệm) ---------------------------


@st.cache_resource(show_spinner=False)
def load_embedding_model(name: str):
    return tp.load_embedding_model(name)


@st.cache_resource(show_spinner=False)
def load_sentiment_models():
    """Trả về cặp (mô hình, vectorizer) hoặc (None, None) nếu chưa huấn luyện."""
    import joblib

    paths = [os.path.join(MODELS_DIR, name) for name in ("sentiment_model.pkl", "tfidf_vectorizer.pkl")]
    if not all(os.path.exists(path) for path in paths):
        return None, None
    try:
        return joblib.load(paths[0]), joblib.load(paths[1])
    except Exception:
        return None, None


def _call_crawler(api_key, video_id, max_comments, include_replies, order, on_progress):
    """Gọi `get_video_comments` với đúng những tham số bản crawler hiện có hỗ trợ."""
    extras = {"include_replies": include_replies, "order": order, "on_progress": on_progress}
    try:
        accepted = inspect.signature(crawler.get_video_comments).parameters
        extras = {name: value for name, value in extras.items() if name in accepted}
    except (TypeError, ValueError):
        pass
    try:
        return crawler.get_video_comments(api_key, video_id, max_results=max_comments, **extras)
    except TypeError:
        return crawler.get_video_comments(api_key, video_id, max_results=max_comments)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_comments(video_id, max_comments, include_replies, order, _api_key, _on_progress=None):
    """Khóa đệm chỉ gồm 4 tham số của video; API key không bao giờ nằm trong khóa."""
    return _call_crawler(_api_key, video_id, max_comments, include_replies, order, _on_progress)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_metadata(video_id, _api_key):
    if _get_video_metadata is None:
        return None
    try:
        return _get_video_metadata(_api_key, video_id)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def compute_embeddings(docs_hash, model_name, _model, _docs):
    """Khóa đệm là chuỗi băm của danh sách tài liệu, không phải bản thân danh sách."""
    return tp.embed_documents(_model, list(_docs))


# --- Thanh bên -------------------------------------------------------------


@dataclass
class Settings:
    video_url: str
    max_comments: int
    include_replies: bool
    order: str
    topic_cfg: tp.TopicConfig
    use_sentiment: bool
    use_llm: bool
    llm_model: str
    llm_base_url: str


def render_sidebar() -> tuple[Settings, bool, bool]:
    with st.sidebar:
        st.subheader("Nguồn dữ liệu")
        video_url = st.text_input("Link video YouTube", placeholder="https://www.youtube.com/watch?v=...",
                                  help="Hỗ trợ cả dạng youtu.be, /shorts/, /live/ hoặc ID 11 ký tự.")
        max_comments = st.slider("Số bình luận tối đa", 100, 5000, 1000, step=100)
        include_replies = st.toggle("Lấy cả bình luận trả lời (replies)", value=True)
        order_label = st.selectbox("Thứ tự lấy bình luận", list(ORDER_CHOICES), index=0)

        with st.expander("Cài đặt nâng cao (BERTopic)"):
            min_topic_size = st.slider("Kích thước cụm tối thiểu", 5, 50, 10)
            nr_topics = None
            if not st.checkbox("Số chủ đề: tự động", value=True):
                nr_topics = int(st.number_input("Số chủ đề mong muốn", 2, 50, 10))
            top_n_words = st.slider("Số từ khóa mỗi chủ đề", 5, 20, 10)
            use_stopwords = st.toggle("Loại bỏ từ dừng tiếng Việt", value=True)
            cluster_selection = st.selectbox("Cách chọn cụm (HDBSCAN)", ["eom", "leaf"], index=0,
                                             help="'eom' cho vài cụm lớn, 'leaf' cho nhiều cụm nhỏ hơn.")
            min_samples = None
            if not st.checkbox("min_samples: mặc định", value=True):
                min_samples = int(st.number_input("min_samples", 1, 50, 5))

        with st.expander("Cảm xúc"):
            use_sentiment = st.toggle("Phân loại cảm xúc (LinearSVC + TF-IDF)", value=True)

        with st.expander("Tóm tắt bằng LLM (Ollama)"):
            use_llm = st.toggle("Bật tóm tắt chủ đề", value=False)
            llm_model = st.text_input("Tên mô hình", value=DEFAULT_MODEL)
            llm_base_url = st.text_input("Địa chỉ Ollama", value=DEFAULT_BASE_URL)

        start = st.button("Bắt đầu phân tích", type="primary", use_container_width=True)
        clear = st.button("Xóa kết quả", use_container_width=True)

        with st.expander("Hướng dẫn"):
            st.markdown(
                "1. Tạo file `.env` ở thư mục gốc với dòng `YOUTUBE_API_KEY=...`.\n"
                "2. Dán link video rồi bấm **Bắt đầu phân tích**.\n"
                "3. Lần chạy đầu tiên cần tải mô hình nhúng câu về máy.\n"
                "4. Nếu chủ đề bị gom thành một cụm quá lớn, hãy giảm kích thước cụm tối thiểu "
                "hoặc đổi sang `leaf` trong Cài đặt nâng cao."
            )

    cfg = tp.TopicConfig(min_topic_size=int(min_topic_size), nr_topics=nr_topics,
                         top_n_words=int(top_n_words), use_stopwords=bool(use_stopwords),
                         cluster_selection_method=cluster_selection, min_samples=min_samples)
    settings = Settings(video_url=video_url.strip(), max_comments=int(max_comments),
                        include_replies=bool(include_replies), order=ORDER_CHOICES[order_label],
                        topic_cfg=cfg, use_sentiment=bool(use_sentiment), use_llm=bool(use_llm),
                        llm_model=llm_model.strip() or DEFAULT_MODEL,
                        llm_base_url=llm_base_url.strip() or DEFAULT_BASE_URL)
    return settings, start, clear


# --- Chạy phân tích --------------------------------------------------------


def _lap(timings: dict, key: str, started: float) -> float:
    timings[key] = time.perf_counter() - started
    return timings[key]


def _summarize_topics(settings, topics_df, limit=8) -> dict:
    """Gọi Ollama cho tối đa `limit` chủ đề lớn nhất; trả về {topic: câu tóm tắt}."""
    summarizer = OllamaSummarizer(model=settings.llm_model, base_url=settings.llm_base_url)
    ready, message = summarizer.available()
    if not ready:
        st.warning(message)
        return {}
    targets = topics_df[topics_df["Topic"] != -1].head(limit)
    summaries, progress = {}, st.progress(0.0, text="Đang tóm tắt chủ đề bằng Ollama…")
    for position, (_, row) in enumerate(targets.iterrows(), start=1):
        try:
            summaries[int(row["Topic"])] = summarizer.summarize_topic(row["Keywords"], row["Representative"])
        except Exception as error:
            summaries[int(row["Topic"])] = f"(không tóm tắt được: {error})"
        progress.progress(position / max(len(targets), 1))
    progress.empty()
    return summaries


def run_analysis(settings: Settings, api_key: str) -> dict | None:
    try:
        video_id = crawler.extract_video_id(settings.video_url)
    except ValueError as error:
        st.error(str(error))
        return None

    timings: dict[str, float] = {}
    with st.status("Đang xử lý…", expanded=True) as status:
        try:
            started = time.perf_counter()
            st.write(f"1/6 · Đang lấy tối đa {settings.max_comments} bình luận từ YouTube…")
            note = st.empty()
            raw = fetch_comments(video_id, settings.max_comments, settings.include_replies,
                                 settings.order, _api_key=api_key,
                                 _on_progress=lambda n, message: note.caption(f"{message} ({n} bình luận)"))
            note.empty()
            st.write(f"　　Đã lấy {len(raw)} bình luận ({_lap(timings, 'thu thập', started):.1f}s)")
            if len(raw) == 0:
                status.update(label="Không có dữ liệu", state="error")
                st.warning("Video này không có bình luận nào hoặc đã tắt bình luận.")
                return None

            started = time.perf_counter()
            st.write("2/6 · Đang làm sạch và tách từ tiếng Việt…")
            df = tp.prepare_documents(raw, text_col="text")
            st.write(f"　　Còn {len(df)} bình luận hợp lệ ({_lap(timings, 'tiền xử lý', started):.1f}s)")
            if len(df) < 10:
                status.update(label="Dữ liệu quá ít", state="error")
                st.warning("Cần ít nhất 10 bình luận hợp lệ để gom cụm chủ đề.")
                return None

            started = time.perf_counter()
            st.write("3/6 · Đang nhúng câu bằng vietnamese-sbert…")
            docs = df["tokenized_text"].tolist()
            model = load_embedding_model(tp.DEFAULT_EMBEDDING_MODEL)
            embeddings = compute_embeddings(tp.documents_hash(docs), tp.DEFAULT_EMBEDDING_MODEL,
                                            _model=model, _docs=docs)
            st.write(f"　　Xong {len(docs)} vector ({_lap(timings, 'embedding', started):.1f}s)")

            started = time.perf_counter()
            st.write("4/6 · Đang gom cụm chủ đề bằng BERTopic…")
            topic_model = tp.build_topic_model(model, settings.topic_cfg)
            topics, topic_model = tp.fit_topics(topic_model, docs, embeddings)
            df["Topic"] = topics
            topics_df = tp.topic_table(topic_model, df, docs)
            df["Topic_Name"] = df["Topic"].map(dict(zip(topics_df["Topic"], topics_df["Name"])))
            n_topics = int((topics_df["Topic"] != -1).sum())
            st.write(f"　　Tìm được {n_topics} chủ đề ({_lap(timings, 'BERTopic', started):.1f}s)")

            started, use_sentiment = time.perf_counter(), settings.use_sentiment
            if use_sentiment:
                st.write("5/6 · Đang phân loại cảm xúc…")
                s_model, s_vectorizer = load_sentiment_models()
                if s_model is None or s_vectorizer is None:
                    st.warning("Chưa có mô hình cảm xúc trong thư mục models/. "
                               "Hãy chạy `python src/train_sentiment.py` trước.")
                    use_sentiment = False
                else:
                    features = s_vectorizer.transform(df["tokenized_text"])
                    df["Sentiment"] = [int(value) for value in s_model.predict(features)]
                    df["Sentiment_Label"] = df["Sentiment"].map(tp.SENTIMENT_ICONS)
            else:
                st.write("5/6 · Bỏ qua bước phân loại cảm xúc.")
            _lap(timings, "cảm xúc", started)

            started, summaries = time.perf_counter(), {}
            if settings.use_llm:
                st.write("6/6 · Đang tóm tắt chủ đề bằng Ollama…")
                summaries = _summarize_topics(settings, topics_df)
            else:
                st.write("6/6 · Bỏ qua bước tóm tắt bằng LLM.")
            _lap(timings, "LLM", started)

            figures = tp.bertopic_figures(topic_model)
        except CrawlError as error:
            status.update(label="Lỗi khi lấy bình luận", state="error")
            st.error(str(error))
            return None
        except Exception as error:  # lỗi ngoài dự kiến: báo gọn, không đổ traceback ra giao diện
            status.update(label="Lỗi khi phân tích", state="error")
            st.error(f"Đã xảy ra lỗi: {error}")
            return None
        status.update(label="Phân tích hoàn tất", state="complete", expanded=False)

    return {"video_id": video_id, "metadata": fetch_metadata(video_id, _api_key=api_key),
            "df": df, "topics": topics_df, "summaries": summaries, "figures": figures,
            "timings": timings, "config": asdict(settings.topic_cfg),
            "use_sentiment": use_sentiment, "n_raw": int(len(raw))}


# --- Các tab kết quả -------------------------------------------------------


def render_overview(result):
    metadata = result.get("metadata")
    if metadata:
        left, right = st.columns([1, 3])
        if metadata.get("thumbnail_url"):
            left.image(metadata["thumbnail_url"], use_container_width=True)
        right.markdown(f"#### {metadata.get('title', '(không có tiêu đề)')}")
        right.caption(f"Kênh: {metadata.get('channel_title', '?')} · "
                      f"{metadata.get('view_count', '?')} lượt xem · "
                      f"{metadata.get('comment_count', '?')} bình luận trên YouTube")

    df, topics_df = result["df"], result["topics"]
    n_topics = int((topics_df["Topic"] != -1).sum())
    positive = "Đang tắt"
    if result["use_sentiment"] and "Sentiment" in df.columns and len(df):
        positive = f"{100.0 * float((df['Sentiment'] == 2).mean()):.1f}%"

    columns = st.columns(5)
    columns[0].metric("Bình luận thu được", f"{result['n_raw']:,}")
    columns[1].metric("Hợp lệ sau lọc", f"{len(df):,}")
    columns[2].metric("Số chủ đề", n_topics)
    columns[3].metric("Nhiễu (-1)", f"{tp.outlier_share(df['Topic']):.1f}%")
    columns[4].metric("Tích cực", positive)

    st.markdown("##### Quy mô từng chủ đề")
    if n_topics:
        st.plotly_chart(tp.topic_size_figure(topics_df), use_container_width=True)
    else:
        st.info("Chưa tách được chủ đề nào ngoài nhóm nhiễu. Hãy giảm kích thước cụm tối thiểu.")

    timings = result["timings"]
    st.caption("Thời gian xử lý: " + " · ".join(f"{k} {v:.1f}s" for k, v in timings.items())
               + f" · tổng {sum(timings.values()):.1f}s")


def render_topics(result):
    topics_df, summaries, df = result["topics"], result["summaries"], result["df"]
    left, right = st.columns(2)
    left.markdown("##### Từ khóa theo chủ đề")
    if result["figures"].get("barchart") is not None:
        left.plotly_chart(result["figures"]["barchart"], use_container_width=True)
    else:
        left.info("Chưa đủ chủ đề để vẽ biểu đồ từ khóa.")
    right.markdown("##### Bản đồ khoảng cách giữa các chủ đề")
    if result["figures"].get("intertopic") is not None:
        right.plotly_chart(result["figures"]["intertopic"], use_container_width=True)
    else:
        right.info("Cần ít nhất vài chủ đề để vẽ bản đồ khoảng cách.")

    table = topics_df[["Topic", "Count", "Name", "Keywords"]].copy()
    config = {"Topic": st.column_config.NumberColumn("Chủ đề", width="small"),
              "Count": st.column_config.NumberColumn("Số bình luận", width="small"),
              "Name": st.column_config.TextColumn("Nhãn", width="medium"),
              "Keywords": st.column_config.ListColumn("Từ khóa", width="large")}
    if summaries:
        table["Summary"] = table["Topic"].map(summaries).fillna("")
        config["Summary"] = st.column_config.TextColumn("Tóm tắt LLM", width="large")
    st.dataframe(table, use_container_width=True, hide_index=True, column_config=config)

    st.markdown("##### Bình luận tiêu biểu")
    for _, row in topics_df[topics_df["Topic"] != -1].head(10).iterrows():
        topic_id = int(row["Topic"])
        with st.expander(f"{row['Name']}  ·  {int(row['Count'])} bình luận"):
            if summaries.get(topic_id):
                st.info(summaries[topic_id])
            for comment in row["Representative"][:5]:
                st.markdown(f"> {comment}")
            if result["use_sentiment"] and "Sentiment" in df.columns:
                st.caption("Cảm xúc trong chủ đề: " + tp.sentiment_summary_line(df, topic_id))


def render_sentiment(result):
    df = result["df"]
    if not result["use_sentiment"] or "Sentiment" not in df.columns:
        st.info("Bước phân loại cảm xúc đang tắt hoặc chưa có mô hình đã huấn luyện.")
        return

    left, right = st.columns([1, 2])
    left.markdown("##### Tỷ lệ ba sắc thái")
    left.plotly_chart(tp.sentiment_donut_figure(tp.sentiment_counts(df)),
                      use_container_width=True, theme=None)
    right.markdown("##### Cảm xúc theo chủ đề (10 chủ đề lớn nhất)")
    right.plotly_chart(tp.sentiment_by_topic_figure(df, result["topics"]),
                       use_container_width=True, theme=None)

    st.markdown("##### Bình luận nhiều lượt thích nhất theo sắc thái")
    columns = st.columns(3)
    for index, code in enumerate((2, 1, 0)):
        columns[index].caption(tp.SENTIMENT_LABELS[code])
        subset = df[df["Sentiment"] == code].nlargest(5, "like_count")
        if len(subset) == 0:
            columns[index].write("(không có)")
            continue
        columns[index].dataframe(
            subset[["text", "like_count"]].rename(columns={"text": "Bình luận", "like_count": "Lượt thích"}),
            use_container_width=True, hide_index=True)


def render_data(result):
    df, topics_df = result["df"], result["topics"]
    left, middle, right = st.columns([2, 2, 3])
    chosen_topics = left.multiselect("Lọc theo chủ đề", topics_df["Name"].tolist(), default=[])
    chosen_sentiment = middle.multiselect("Lọc theo cảm xúc",
                                          [tp.SENTIMENT_LABELS[code] for code in (0, 1, 2)],
                                          default=[], disabled=not result["use_sentiment"])
    keyword = right.text_input("Tìm trong nội dung bình luận", value="")

    view = tp.filter_comments(df, chosen_topics, chosen_sentiment, keyword)
    table = tp.comments_table(view, result["use_sentiment"])
    st.caption(f"Đang hiển thị {len(table):,} / {len(df):,} bình luận.")
    st.dataframe(table, use_container_width=True, hide_index=True, height=460)
    st.download_button("Tải CSV đang lọc", data=table.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"binh_luan_{result['video_id']}.csv", mime="text/csv")


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def render_model(_result=None):
    """Tab này đọc các tệp trong models/ và results/ chứ không phụ thuộc lần chạy hiện tại."""
    st.markdown("##### Mô hình nhúng câu")
    st.markdown(
        "`keepitreal/vietnamese-sbert` là mô hình Sentence-BERT tinh chỉnh từ PhoBERT-base "
        "(kiến trúc RoBERTa, 12 tầng, hidden size 768, bộ tách từ của PhoBERT). Mỗi bình luận "
        "được mã hóa thành một vector 768 chiều bằng cách lấy trung bình (mean pooling) vector "
        "các token, độ dài tối đa 256 token. Mô hình làm việc trên văn bản đã tách từ nên bước "
        "`pyvi` phía trước là bắt buộc."
    )

    st.markdown("##### Mô hình phân loại cảm xúc")
    card = _read_json(os.path.join(MODELS_DIR, "model_card.json"))
    metrics = _read_json(os.path.join(RESULTS_DIR, "metrics.json"))
    if card:
        st.markdown(f"**{card.get('model', 'LinearSVC')}** trên đặc trưng {card.get('vectorizer', 'TF-IDF')}. "
                    f"Tiền xử lý: {card.get('preprocessing', 'clean_text + pyvi')}. "
                    f"Huấn luyện lúc {card.get('trained_at', '?')} với scikit-learn "
                    f"{card.get('sklearn_version', '?')}.")
    else:
        st.info("Chưa có models/model_card.json. Hãy chạy `python src/train_sentiment.py` "
                "để sinh mô hình và thẻ mô tả.")

    if not metrics:
        st.info("Chưa có results/metrics.json nên chưa hiển thị được kết quả đánh giá.")
    else:
        test = metrics.get("test", {})
        columns = st.columns(3)
        columns[0].metric("Accuracy (test)", f"{test.get('accuracy', 0):.3f}")
        columns[1].metric("Macro-F1 (test)", f"{test.get('f1_macro', 0):.3f}")
        columns[2].metric("Weighted-F1 (test)", f"{test.get('f1_weighted', 0):.3f}")
        if metrics.get("model_comparison"):
            st.markdown("##### So sánh các mô hình (cross-validation trên tập huấn luyện)")
            st.dataframe(pd.DataFrame(metrics["model_comparison"]), use_container_width=True, hide_index=True)
        per_class = tp.per_class_table(test.get("per_class"))
        if len(per_class):
            st.markdown("##### Kết quả theo từng lớp")
            st.dataframe(per_class, use_container_width=True, hide_index=True)

    image_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    if os.path.exists(image_path):
        st.markdown("##### Ma trận nhầm lẫn trên tập kiểm tra")
        st.image(image_path, width=560)


# --- Điểm vào --------------------------------------------------------------


def main():
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    api_key = os.getenv("YOUTUBE_API_KEY")

    st.title("🎬 Phân tích bình luận YouTube")
    st.caption("Gom cụm chủ đề bằng vietnamese-sbert (Sentence-BERT tinh chỉnh từ PhoBERT-base) "
               "kết hợp BERTopic, phân loại cảm xúc bằng LinearSVC trên đặc trưng TF-IDF.")

    settings, start, clear = render_sidebar()
    if clear:
        st.session_state.pop("result", None)
    if start:
        if not settings.video_url:
            st.error("Vui lòng nhập link video YouTube.")
        elif not api_key:
            st.error("Không tìm thấy YOUTUBE_API_KEY. Hãy tạo file .env ở thư mục gốc dự án.")
        else:
            result = run_analysis(settings, api_key)
            if result is not None:
                st.session_state["result"] = result

    result = st.session_state.get("result")
    if result is None:
        st.info("Nhập link video ở thanh bên rồi bấm **Bắt đầu phân tích**. "
                "Kết quả sẽ được giữ lại khi bạn chuyển tab hoặc đổi bộ lọc.")
        return

    tabs = st.tabs(["Tổng quan", "Chủ đề", "Cảm xúc", "Dữ liệu", "Mô hình"])
    renderers = (render_overview, render_topics, render_sentiment, render_data, render_model)
    for tab, render in zip(tabs, renderers):
        with tab:
            render(result)


main()
