"""Pipeline gom cụm chủ đề cho bình luận tiếng Việt.

Module này gom toàn bộ logic BERTopic vào một chỗ để giao diện Streamlit
(`app.py`) và công cụ dòng lệnh (`topic_model.py`) dùng chung một đường đi,
không còn hai bản sao lệch nhau.

Quy ước quan trọng: các thư viện nặng (torch, sentence-transformers, bertopic,
umap, hdbscan, plotly) chỉ được import bên trong hàm. Nhờ vậy `import
topic_pipeline` chỉ tốn vài phần mười giây, giúp kiểm thử và lúc mở app nhanh.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import pandas as pd

from preprocess import clean_text, tokenize_vietnamese

try:  # bước B có thể chưa bổ sung danh sách stopwords
    from preprocess import get_vietnamese_stopwords
except ImportError:  # pragma: no cover - chỉ xảy ra với bản preprocess cũ
    get_vietnamese_stopwords = None


# Mô hình nhúng câu: Sentence-BERT tiếng Việt, tinh chỉnh từ PhoBERT-base.
DEFAULT_EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"

# Mẫu token cho c-TF-IDF: token phải bắt đầu bằng một chữ cái (mọi bảng chữ
# Unicode), sau đó cho phép chữ, số và dấu gạch dưới. Nhờ vậy từ ghép tiếng
# Việt (`chương_trình`) được giữ nguyên còn số trần (`10`, `2024`) không còn
# lọt vào danh sách từ khóa chủ đề.
TOKEN_PATTERN = r"(?u)\b[^\W\d_]\w*\b"

SENTIMENT_LABELS = {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"}
SENTIMENT_COLORS = {0: "#EF4444", 1: "#9CA3AF", 2: "#22C55E"}
SENTIMENT_ICONS = {0: "🔴 Tiêu cực", 1: "⚪ Trung tính", 2: "🟢 Tích cực"}

OUTLIER_LABEL = "-1: Nhiễu (không thuộc chủ đề nào)"

# Tên cột tiếng Việt cho bảng chỉ số theo lớp trong tab "Mô hình".
PER_CLASS_HEADERS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "support": "Số mẫu",
}

# Các cột chỉ xuất hiện trong tệp CSV tải về, không hiện trên bảng cho gọn.
EXPORT_EXTRA_COLUMNS = {
    "comment_id": "Mã bình luận",
    "published_at": "Thời điểm đăng",
    "reply_count": "Số trả lời",
}

# Nhãn chủ đề dài hơn mức này bị cắt bớt khi vẽ trục y của biểu đồ.
MAX_LABEL_CHARS = 28

# Hợp đồng cột của `crawler.get_video_comments`. Mọi nguồn dữ liệu khác (tệp CSV
# người dùng tải lên, tệp mẫu trong `data/`) phải đưa về đúng thứ tự cột này thì
# phần còn lại của pipeline mới dùng chung được một đường đi.
COMMENT_COLUMNS = [
    "comment_id",
    "parent_id",
    "is_reply",
    "author",
    "published_at",
    "like_count",
    "reply_count",
    "text",
]


@dataclass
class TopicConfig:
    """Các tham số người phân tích có thể chỉnh từ giao diện hoặc CLI."""

    min_topic_size: int = 10
    nr_topics: int | None = None
    n_neighbors: int = 15
    n_components: int = 5
    top_n_words: int = 10
    random_state: int = 42
    use_stopwords: bool = True
    cluster_selection_method: str = "eom"  # 'eom' gom cụm lớn, 'leaf' cụm nhỏ hơn
    min_samples: int | None = None  # None: HDBSCAN lấy bằng min_cluster_size


# --------------------------------------------------------------------------
# Chuẩn bị dữ liệu
# --------------------------------------------------------------------------


def prepare_documents(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Làm sạch, tách từ và bỏ những bình luận quá ngắn.

    Thêm hai cột `clean_text` và `tokenized_text`, loại các dòng còn dưới 2
    token rồi đánh lại chỉ số. Chỉ số mới chính là vị trí của tài liệu trong
    danh sách đưa vào BERTopic, nên `topic_table` ánh xạ ngược được về văn bản
    gốc.
    """
    if text_col not in df.columns:
        raise KeyError(f"Không tìm thấy cột văn bản '{text_col}' trong dữ liệu.")

    out = df.copy()
    out["clean_text"] = out[text_col].apply(clean_text)
    out["tokenized_text"] = out["clean_text"].apply(tokenize_vietnamese)
    keep = out["tokenized_text"].apply(lambda value: len(str(value).split()) >= 2)
    return out[keep].reset_index(drop=True)


def _int_column(source: pd.DataFrame, name: str) -> pd.Series:
    """Cột số nguyên: thiếu cột thì trả về 0, giá trị hỏng cũng quy về 0."""
    if name not in source.columns:
        return pd.Series(0, index=source.index, dtype="int64")
    return pd.to_numeric(source[name], errors="coerce").fillna(0).astype("int64")


def comments_from_dataframe(
    df: pd.DataFrame,
    text_col: str,
    limit: int | None = None,
) -> pd.DataFrame:
    """Đưa một bảng bất kỳ về đúng hợp đồng cột của `crawler.get_video_comments`.

    Dùng cho hai nguồn dữ liệu ngoại tuyến: tệp CSV người dùng tải lên và tệp
    mẫu trong `data/`. Cột nào có trong bảng thì lấy nguyên, cột nào thiếu thì
    điền giá trị mặc định (`author` là "Ẩn danh", `comment_id` là chỉ số dòng,
    `like_count` và `reply_count` là 0). Dòng có nội dung trống hoặc NaN bị loại
    trước, rồi mới cắt theo `limit`, nên số dòng trả về đúng bằng số bình luận
    dùng được mà người dùng yêu cầu.

    Ném `ValueError` kèm thông điệp tiếng Việt khi thiếu cột văn bản hoặc không
    còn dòng nào dùng được; phía giao diện bắt lỗi này và hiển thị bằng
    `st.error` thay vì đổ traceback.
    """
    if text_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột văn bản '{text_col}' trong dữ liệu.")

    source = df[df[text_col].notna()]
    source = source[source[text_col].astype(str).str.strip() != ""]
    if limit is not None and int(limit) > 0:
        source = source.head(int(limit))
    if source.empty:
        raise ValueError("Không có dòng bình luận nào dùng được trong dữ liệu.")

    out = pd.DataFrame(index=source.index)
    out["comment_id"] = (
        source["comment_id"].astype(str) if "comment_id" in source.columns
        else source.index.astype(str)
    )
    out["parent_id"] = source["parent_id"] if "parent_id" in source.columns else None
    out["is_reply"] = source["is_reply"] if "is_reply" in source.columns else False
    out["author"] = source["author"].astype(str) if "author" in source.columns else "Ẩn danh"
    out["published_at"] = (
        source["published_at"].astype(str) if "published_at" in source.columns else ""
    )
    out["like_count"] = _int_column(source, "like_count")
    out["reply_count"] = _int_column(source, "reply_count")
    out["text"] = source[text_col].astype(str)
    out["is_reply"] = out["is_reply"].fillna(False).astype(bool)
    return out[COMMENT_COLUMNS].reset_index(drop=True)


def documents_hash(docs: Sequence[str]) -> str:
    """Chuỗi băm ổn định của một danh sách tài liệu, dùng làm khóa bộ nhớ đệm."""
    digest = hashlib.sha1()
    digest.update(str(len(docs)).encode("utf-8"))
    for doc in docs:
        digest.update(b"\x00")
        digest.update(str(doc).encode("utf-8"))
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Nhúng câu và mô hình chủ đề
# --------------------------------------------------------------------------


def load_embedding_model(name: str = DEFAULT_EMBEDDING_MODEL):
    """Nạp mô hình Sentence-BERT tiếng Việt (tải về lần đầu, sau đó dùng cache)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def embed_documents(model, docs: list[str], batch_size: int = 64):
    """Mã hóa danh sách bình luận thành ma trận vector 768 chiều."""
    import numpy as np

    if not docs:
        return np.zeros((0, 0), dtype="float32")
    return model.encode(list(docs), batch_size=batch_size, show_progress_bar=False)


def _stopword_list(use_stopwords: bool) -> list[str] | None:
    if not use_stopwords or get_vietnamese_stopwords is None:
        return None
    words = get_vietnamese_stopwords()
    return list(words) if words else None


def build_topic_model(embedding_model, cfg: TopicConfig):
    """Dựng BERTopic với UMAP, HDBSCAN và CountVectorizer khai báo tường minh.

    `embedding_model` luôn phải được truyền vào, kể cả khi đã có sẵn vector.
    BERTopic mặc định `language="english"` và khi đó bước tiền xử lý nội bộ
    của nó xóa mọi ký tự ngoài `[A-Za-z0-9 ]`, biến `không` thành `khng` và
    `chương_trình` thành `chngtrnh` trước khi tính c-TF-IDF. Truyền
    `embedding_model` khiến BERTopic đặt `language = None` và bỏ qua bước đó.
    """
    import hdbscan
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    umap_model = UMAP(
        n_neighbors=cfg.n_neighbors,
        n_components=cfg.n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=cfg.random_state,
    )
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=cfg.min_topic_size,
        min_samples=cfg.min_samples,
        metric="euclidean",
        cluster_selection_method=cfg.cluster_selection_method,
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        token_pattern=TOKEN_PATTERN,
        stop_words=_stopword_list(cfg.use_stopwords),
    )
    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=cfg.min_topic_size,
        nr_topics=cfg.nr_topics,
        top_n_words=cfg.top_n_words,
        calculate_probabilities=False,
    )


def fit_topics(topic_model, docs: list[str], embeddings=None) -> tuple[list[int], Any]:
    """Huấn luyện mô hình chủ đề và gắn nhãn rút gọn cho các biểu đồ của BERTopic."""
    topics, _probabilities = topic_model.fit_transform(list(docs), embeddings)
    try:
        labels = topic_model.generate_topic_labels(nr_words=3, topic_prefix=True, separator=", ")
        topic_model.set_topic_labels(labels)
    except Exception:  # pragma: no cover - nhãn chỉ để hiển thị, lỗi thì bỏ qua
        pass
    return [int(topic) for topic in topics], topic_model


# --------------------------------------------------------------------------
# Đọc kết quả (thuần dữ liệu, kiểm thử được bằng mô hình giả)
# --------------------------------------------------------------------------


def _keywords_of(row, info_columns) -> list[str]:
    if "Representation" not in info_columns:
        return []
    value = row["Representation"]
    if isinstance(value, (list, tuple)):
        return [str(word) for word in value if str(word)]
    return []


def topic_label_mapping(topic_model, nr_words: int = 3) -> dict[int, str]:
    """Bảng tra `topic id -> nhãn ngắn` dạng `0: từ1, từ2, từ3`."""
    info = topic_model.get_topic_info()
    columns = list(info.columns)
    mapping: dict[int, str] = {}
    for _, row in info.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            mapping[topic_id] = OUTLIER_LABEL
            continue
        keywords = _keywords_of(row, columns)[:nr_words]
        if keywords:
            mapping[topic_id] = f"{topic_id}: " + ", ".join(keywords)
        else:
            mapping[topic_id] = str(row["Name"]) if "Name" in columns else str(topic_id)
    return mapping


def topic_table(
    topic_model,
    df: pd.DataFrame,
    docs: Sequence[str],
    text_col: str = "text",
) -> pd.DataFrame:
    """Bảng tổng hợp một dòng cho mỗi chủ đề, kể cả chủ đề nhiễu -1.

    Cột `Representative` chứa văn bản GỐC của người xem chứ không phải chuỗi
    đã tách từ: BERTopic trả về tài liệu đại diện lấy từ `docs` (đã làm sạch,
    viết thường), nên ở đây ta dò vị trí của chúng trong `docs` rồi lấy đúng
    dòng tương ứng của `df`.
    """
    info = topic_model.get_topic_info()
    columns = list(info.columns)
    labels = topic_label_mapping(topic_model)

    position: dict[str, int] = {}
    for index, doc in enumerate(docs):
        position.setdefault(str(doc), index)
    originals = df[text_col].astype(str).tolist() if text_col in df.columns else []

    rows = []
    for _, row in info.iterrows():
        topic_id = int(row["Topic"])
        representative = []
        raw_docs = row["Representative_Docs"] if "Representative_Docs" in columns else []
        if isinstance(raw_docs, str):
            raw_docs = [raw_docs]
        elif not isinstance(raw_docs, (list, tuple)):
            raw_docs = []
        for doc in raw_docs:
            if not isinstance(doc, str):
                continue
            index = position.get(doc)
            if index is not None and 0 <= index < len(originals):
                representative.append(originals[index])
            else:
                representative.append(doc)
        rows.append(
            {
                "Topic": topic_id,
                "Count": int(row["Count"]),
                "Name": labels.get(topic_id, str(topic_id)),
                "Keywords": _keywords_of(row, columns),
                "Representative": representative,
            }
        )
    return pd.DataFrame(rows, columns=["Topic", "Count", "Name", "Keywords", "Representative"])


def outlier_share(topics: Iterable[int]) -> float:
    """Tỷ lệ phần trăm bình luận bị xếp vào nhóm nhiễu (-1)."""
    values = [int(topic) for topic in topics]
    if not values:
        return 0.0
    return 100.0 * sum(1 for topic in values if topic == -1) / len(values)


def sentiment_counts(df: pd.DataFrame) -> dict[int, int]:
    """Đếm số bình luận theo từng lớp cảm xúc, luôn trả về đủ ba khóa 0/1/2."""
    if "Sentiment" not in df.columns:
        return {code: 0 for code in (0, 1, 2)}
    return {code: int((df["Sentiment"] == code).sum()) for code in (0, 1, 2)}


def sentiment_summary_line(df: pd.DataFrame, topic_id: int) -> str:
    """Chuỗi `Tiêu cực: 3 · Trung tính: 1 · Tích cực: 7` cho một chủ đề."""
    subset = df[df["Topic"] == topic_id]
    counts = sentiment_counts(subset)
    return " · ".join(f"{SENTIMENT_LABELS[code]}: {counts[code]}" for code in (0, 1, 2))


def filter_comments(df: pd.DataFrame, topic_names=(), sentiment_names=(), keyword="") -> pd.DataFrame:
    """Lọc bảng bình luận theo chủ đề, cảm xúc và từ khóa tìm kiếm."""
    view = df
    if topic_names:
        view = view[view["Topic_Name"].isin(list(topic_names))]
    if sentiment_names and "Sentiment" in view.columns:
        codes = [code for code, name in SENTIMENT_LABELS.items() if name in sentiment_names]
        view = view[view["Sentiment"].isin(codes)]
    keyword = str(keyword).strip()
    if keyword:
        view = view[view["text"].astype(str).str.contains(keyword, case=False, na=False, regex=False)]
    return view


def comments_table(df: pd.DataFrame, use_sentiment: bool = True) -> pd.DataFrame:
    """Chọn và đổi tên cột sang tiếng Việt để hiển thị hoặc tải về CSV."""
    columns = {"Topic_Name": "Chủ đề"}
    if use_sentiment and "Sentiment_Label" in df.columns:
        columns["Sentiment_Label"] = "Cảm xúc"
    columns.update({"author": "Tác giả", "text": "Bình luận", "like_count": "Lượt thích"})
    if "is_reply" in df.columns:
        columns["is_reply"] = "Trả lời?"
    present = [name for name in columns if name in df.columns]
    return df[present].rename(columns={name: columns[name] for name in present})


def comments_export_table(df: pd.DataFrame, use_sentiment: bool = True) -> pd.DataFrame:
    """Bảng dành cho tệp CSV tải về.

    Giữ nguyên các cột của bảng hiển thị rồi thêm những cột truy vết mà bảng
    trên màn hình cố tình bỏ bớt cho gọn: mã bình luận, thời điểm đăng và số
    lượt trả lời. Nhờ vậy người nhận tệp dò ngược được về bình luận gốc trên
    YouTube.
    """
    table = comments_table(df, use_sentiment)
    for column, title in EXPORT_EXTRA_COLUMNS.items():
        if column in df.columns:
            table[title] = df[column]
    return table


def topic_config_line(config) -> str:
    """Một dòng mô tả cấu hình BERTopic đã sinh ra kết quả đang hiển thị.

    Kết quả cố tình không bị xóa khi người dùng chỉnh thanh bên, nên phải nói rõ
    kết quả trên màn hình được tính với cấu hình nào.
    """
    if not config:
        return ""
    min_samples = config.get("min_samples")
    nr_topics = config.get("nr_topics")
    parts = [
        f"min_topic_size={config.get('min_topic_size', '?')}",
        f"min_samples={min_samples if min_samples is not None else 'mặc định'}",
        str(config.get("cluster_selection_method", "eom")),
        "stopwords=" + ("bật" if config.get("use_stopwords", True) else "tắt"),
        f"nr_topics={nr_topics if nr_topics is not None else 'tự động'}",
    ]
    return " · ".join(parts)


COMPARISON_HEADERS = {
    "model": "Mô hình",
    "cv_accuracy_mean": "CV accuracy",
    "cv_accuracy_std": "± acc",
    "cv_f1_macro_mean": "CV macro-F1",
    "cv_f1_macro_std": "± F1",
    "fit_seconds": "Thời gian khớp (s)",
}


def comparison_frame(records) -> pd.DataFrame:
    """Bảng so sánh mô hình với tiêu đề tiếng Việt, accuracy quy về phần trăm."""
    frame = pd.DataFrame(records)
    if not set(COMPARISON_HEADERS).issubset(frame.columns):
        return frame
    return pd.DataFrame(
        {
            "Mô hình": frame["model"],
            "CV accuracy": frame["cv_accuracy_mean"] * 100,
            "± acc": frame["cv_accuracy_std"] * 100,
            "CV macro-F1": frame["cv_f1_macro_mean"],
            "± F1": frame["cv_f1_macro_std"],
            "Thời gian khớp (s)": frame["fit_seconds"],
        }
    )


def c_grid_frame(records) -> pd.DataFrame:
    """Bảng kết quả dò tham số C, cùng quy ước hiển thị với bảng so sánh."""
    frame = pd.DataFrame(records)
    needed = {"C", "cv_f1_macro_mean", "cv_f1_macro_std", "cv_accuracy_mean"}
    if not needed.issubset(frame.columns):
        return frame
    return pd.DataFrame(
        {
            "C": frame["C"],
            "CV macro-F1": frame["cv_f1_macro_mean"],
            "± F1": frame["cv_f1_macro_std"],
            "CV accuracy": frame["cv_accuracy_mean"] * 100,
        }
    )


def c_grid_note(metrics) -> str:
    """Một câu đọc thẳng từ số liệu: C nào được chọn và cách biệt tới đâu."""
    grid = (metrics.get("selected_model") or {}).get("c_grid_results") or []
    if not grid:
        return ""
    best = max(grid, key=lambda row: row["cv_f1_macro_mean"])
    note = (f"C={best['C']:g} cho macro-F1 cross-validation cao nhất "
            f"({best['cv_f1_macro_mean']:.4f} ± {best['cv_f1_macro_std']:.4f}) "
            "và đó là cấu hình được lưu lại.")
    other = next((row for row in metrics.get("model_comparison") or []
                  if row["model"] == "LogisticRegression"), None)
    if other is None:
        return note
    gap = best["cv_f1_macro_mean"] - other["cv_f1_macro_mean"]
    spread = max(best["cv_f1_macro_std"], other["cv_f1_macro_std"])
    if abs(gap) <= spread:
        return note + (f" Chênh lệch so với LogisticRegression là {gap:+.4f}, nhỏ hơn độ "
                       "lệch chuẩn giữa các fold, nên hai mô hình coi như ngang nhau.")
    return note + f" Chênh lệch so với LogisticRegression là {gap:+.4f}."


def per_class_table(per_class: dict) -> pd.DataFrame:
    """Bảng precision/recall/F1 theo từng lớp, đọc từ results/metrics.json.

    Tên cột và tên lớp đều để tiếng Việt cho khớp phần còn lại của giao diện.
    """
    rows = []
    for code, values in (per_class or {}).items():
        try:
            name = SENTIMENT_LABELS.get(int(code), str(code))
        except (TypeError, ValueError):
            name = str(code)
        row = {"Lớp": name}
        for key, title in PER_CLASS_HEADERS.items():
            if key in (values or {}):
                row[title] = values[key]
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Biểu đồ dùng chung cho giao diện (gom ở đây để app.py chỉ lo phần bố cục)
# --------------------------------------------------------------------------


def short_label(name, limit: int = MAX_LABEL_CHARS) -> str:
    """Rút gọn nhãn chủ đề cho trục y.

    Nhãn quá dài khiến plotly thu hẹp phần chữ đến mức chỉ còn một ký tự; cắt
    sẵn ở đây rồi bật `automargin` thì nhãn luôn đọc được. Tiền tố `id:` được
    giữ lại nên hai chủ đề không bao giờ bị rút gọn thành cùng một nhãn.
    """
    text = str(name)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def topic_size_figure(topics_df: pd.DataFrame, top_n: int = 15):
    """Biểu đồ cột ngang: số bình luận của từng chủ đề, bỏ nhóm nhiễu."""
    import plotly.graph_objects as go

    data = topics_df[topics_df["Topic"] != -1].sort_values("Count", ascending=False).head(top_n)
    data = data.iloc[::-1]
    figure = go.Figure(
        go.Bar(
            x=data["Count"],
            y=[short_label(name) for name in data["Name"]],
            orientation="h",
            marker=dict(color="#F43F5E"),
            customdata=data["Name"],
            hovertemplate="%{customdata}<br>%{x} bình luận<extra></extra>",
        )
    )
    figure.update_layout(
        height=max(240, 34 * len(data) + 90),
        margin=dict(l=10, r=8, t=30, b=8),
        xaxis_title="Số bình luận",
        yaxis=dict(automargin=True, title=None),
        showlegend=False,
    )
    return figure


def bertopic_figures(topic_model) -> dict:
    """Dựng sẵn hai biểu đồ của BERTopic; chủ đề nào quá ít dữ liệu thì trả về None."""
    builders = {
        "barchart": lambda: topic_model.visualize_barchart(top_n_topics=8),
        "intertopic": lambda: topic_model.visualize_topics(),
    }
    figures = {}
    for name, builder in builders.items():
        try:
            figures[name] = builder()
        except Exception:
            figures[name] = None
    return figures


def sentiment_donut_figure(counts: dict[int, int]):
    """Biểu đồ vành khuyên cho ba lớp cảm xúc, màu cố định."""
    import plotly.graph_objects as go

    present = [code for code in (0, 1, 2) if counts.get(code, 0) > 0]
    figure = go.Figure(
        go.Pie(
            labels=[SENTIMENT_LABELS[code] for code in present],
            values=[counts.get(code, 0) for code in present],
            hole=0.45,
            sort=False,
            marker=dict(colors=[SENTIMENT_COLORS[code] for code in present]),
            textinfo="percent+label",
        )
    )
    figure.update_layout(height=360, margin=dict(l=8, r=8, t=30, b=8), showlegend=True)
    return figure


def sentiment_by_topic_figure(df: pd.DataFrame, topics_df: pd.DataFrame, top_n: int = 10):
    """Cột chồng ngang: tỷ lệ ba lớp cảm xúc trong từng chủ đề."""
    import plotly.graph_objects as go

    top_topics = (
        topics_df[topics_df["Topic"] != -1].sort_values("Count", ascending=False).head(top_n)
    )
    full_names = top_topics["Name"].tolist()[::-1]
    names = [short_label(name) for name in full_names]
    ids = top_topics["Topic"].tolist()[::-1]

    figure = go.Figure()
    for code in (0, 1, 2):
        values = []
        for topic_id in ids:
            subset = df[df["Topic"] == topic_id]
            values.append(int((subset["Sentiment"] == code).sum()) if len(subset) else 0)
        figure.add_trace(
            go.Bar(
                x=values,
                y=names,
                name=SENTIMENT_LABELS[code],
                orientation="h",
                marker=dict(color=SENTIMENT_COLORS[code]),
                customdata=full_names,
                hovertemplate="%{customdata}<br>" + SENTIMENT_LABELS[code] + ": %{x}<extra></extra>",
            )
        )
    figure.update_layout(
        barmode="stack",
        height=max(260, 34 * len(names) + 110),
        # `automargin` để plotly tự chừa đủ chỗ cho nhãn chủ đề bên trái thay vì
        # cắt chúng còn một ký tự khi cửa sổ rộng.
        margin=dict(l=10, r=8, t=30, b=8),
        xaxis_title="Số bình luận",
        yaxis=dict(automargin=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return figure
