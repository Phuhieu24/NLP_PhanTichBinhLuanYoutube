import streamlit as st
import pandas as pd
import sys
import os

# Thêm thư mục hiện tại vào sys.path để tránh lỗi ModuleNotFoundError khi chạy Streamlit
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
import joblib
import plotly.graph_objects as go
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

# Font cho Word Cloud (hỗ trợ tiếng Việt trên Windows)
_wc_font_path = None
for _fp in ['C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/segoeui.ttf']:
    if os.path.exists(_fp):
        _wc_font_path = _fp
        break

from crawler import get_video_comments, extract_video_id
from preprocess import clean_text, tokenize_vietnamese

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

# Cấu hình giao diện toàn trang
st.set_page_config(page_title="YouTube NLP Topic Modeling", page_icon="🎥", layout="wide")

# Load API Key
project_root = os.path.dirname(os.path.dirname(__file__))
load_dotenv(os.path.join(project_root, '.env'))
API_KEY = os.getenv("YOUTUBE_API_KEY")

# --- Danh sách Stopword tiếng Việt ---
VIETNAMESE_STOPWORDS = [
    "là", "và", "của", "có", "trong", "được", "cho", "với", "các", "này",
    "đã", "một", "những", "không", "như", "khi", "vào", "về", "từ", "thì",
    "mà", "để", "hay", "ra", "lại", "cũng", "đây", "đó", "thế", "nên",
    "rất", "cần", "bị", "vì", "tuy", "nhưng", "nếu", "hơn", "nhất", "rồi",
    "đến", "lên", "theo", "bởi", "qua", "sau", "trước", "trên", "dưới",
    "nào", "ai", "gì", "sao", "vậy", "đâu", "đi", "thôi", "thật", "ơi",
    "ạ", "à", "ừ", "uh", "uh", "ha", "he", "hihi", "haha", "hehe",
    "tôi", "mình", "bạn", "em", "anh", "chị", "họ", "nó", "ta", "chúng",
    "mọi", "người", "cái", "con", "bao", "nhiêu", "lắm", "quá", "vẫn",
    "thêm", "chỉ", "cả", "đều", "sẽ", "đang", "còn", "nữa", "mới", "đã",
    "phải", "muốn", "thấy", "biết", "làm", "nói", "nghĩ", "thích", "dùng",
]

@st.cache_resource(show_spinner="Đang tải mô hình PhoBERT...")
def load_embedding_model():
    return SentenceTransformer("keepitreal/vietnamese-sbert")

@st.cache_resource(show_spinner="Đang nạp mô hình Phân tích cảm xúc (Sentiment)...")
def _load_sentiment_cached():
    """Chỉ được gọi khi file model tồn tại — tránh cache kết quả None."""
    model_path = os.path.join(project_root, 'models', 'sentiment_model.pkl')
    vec_path = os.path.join(project_root, 'models', 'tfidf_vectorizer.pkl')
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer

def load_sentiment_models():
    """Kiểm tra file trước khi load — chỉ cache khi model thực sự tồn tại."""
    model_path = os.path.join(project_root, 'models', 'sentiment_model.pkl')
    vec_path = os.path.join(project_root, 'models', 'tfidf_vectorizer.pkl')
    if os.path.exists(model_path) and os.path.exists(vec_path):
        return _load_sentiment_cached()
    return None, None

@st.cache_data(show_spinner="Đang mã hóa bình luận thành vector (kết quả sẽ được cache)...")
def compute_embeddings(_model, docs):
    """Cache embeddings để không phải tính lại khi phân tích cùng danh sách bình luận."""
    return _model.encode(docs)

# Header
st.title("🎥 Hệ thống Phân tích Chủ đề Bình luận YouTube (NLP)")
st.markdown("Đồ án Xử lý ngôn ngữ tự nhiên - Ứng dụng **BERTopic** và **PhoBERT** để gom cụm chủ đề.")

# Sidebar cho Input
with st.sidebar:
    st.header("⚙️ Cài đặt Phân tích")
    video_url = st.text_input("Nhập Link YouTube:", placeholder="Ví dụ: https://www.youtube.com/watch?v=exR2qh0zFCA")
    max_comments = st.number_input("Số lượng bình luận tối đa:", min_value=100, max_value=5000, value=1000, step=100)
    start_btn = st.button("🚀 Bắt đầu Phân tích", use_container_width=True)

    st.markdown("---")
    st.markdown("**Hướng dẫn:**\n1. Copy link một video YouTube bất kỳ.\n2. Dán vào ô trên.\n3. Nhấn Bắt đầu và đợi vài phút để hệ thống tự động cào dữ liệu và chạy AI.")

    st.markdown("---")
    st.header("🤖 Tích hợp AI Tóm tắt")
    use_ollama = st.checkbox("Bật AI tóm tắt chủ đề (Ollama)", value=False)
    ollama_model = st.text_input("Tên Model:", value="qwen2", help="Mô hình phải được tải sẵn trong máy qua lệnh 'ollama pull <tên_model>'")

    st.markdown("---")
    st.header("❤️ Phân tích Cảm xúc (ML)")
    use_sentiment = st.checkbox("Bật AI Phân tích Cảm xúc (Mô hình Tự Train)", value=True, help="Sử dụng mô hình Máy véc-tơ hỗ trợ (LinearSVC) đã train bằng file train_sentiment.py")

# Khu vực chính
if start_btn:
    if not video_url:
        st.error("Vui lòng nhập Link YouTube!")
    elif not API_KEY:
        st.error("Lỗi: Không tìm thấy YOUTUBE_API_KEY trong file .env")
    else:
        try:
            video_id = extract_video_id(video_url)

            # --- Progress UI ---
            st.markdown("### 🔄 Tiến trình xử lý")
            progress_bar = st.progress(0)
            status_text = st.empty()

            # --- Bước 1: Crawl ---
            status_text.info(f"Đang kết nối YouTube API để tải tối đa {max_comments} bình luận...")
            df = get_video_comments(API_KEY, video_id, max_results=max_comments)
            progress_bar.progress(25)

            if len(df) == 0:
                st.warning("Không tìm thấy bình luận nào cho video này.")
                st.stop()

            # --- Bước 2: Preprocess ---
            status_text.info(f"Đang làm sạch và tách từ (Word Tokenization) cho {len(df)} bình luận...")
            df['clean_text'] = df['text'].apply(clean_text)
            df = df[df['clean_text'].str.strip().astype(bool)]  # Lọc bình luận rỗng sau clean
            df = df.drop_duplicates(subset=['text'])  # Loại bỏ bình luận trùng lặp (spam)
            df['tokenized_text'] = df['clean_text'].apply(tokenize_vietnamese)

            df = df.dropna(subset=['tokenized_text'])
            df = df[df['tokenized_text'].apply(lambda x: len(str(x).split()) >= 2)]
            progress_bar.progress(50)

            if len(df) < 10:
                st.warning("Số lượng bình luận hợp lệ quá ít để chạy AI (Cần tối thiểu 10 bình luận).")
                st.stop()

            # --- Bước 3: Model ---
            status_text.info("Đang khởi tạo mô hình AI PhoBERT...")
            sentence_model = load_embedding_model()

            status_text.info(f"Đang chạy thuật toán BERTopic trên {len(df)} bình luận (Quá trình này có thể tốn 1-2 phút)...")
            docs = df['tokenized_text'].tolist()
            embeddings = compute_embeddings(sentence_model, docs)
            progress_bar.progress(75)

            # CountVectorizer với stopword tiếng Việt để lọc từ không có nghĩa
            vectorizer_model = CountVectorizer(
                token_pattern=r'(?u)\b\w+\b',
                stop_words=VIETNAMESE_STOPWORDS,
                min_df=2  # Bỏ từ chỉ xuất hiện ở 1 bình luận
            )

            # Tự động xác định số chủ đề, min_topic_size giúp tránh chủ đề quá nhỏ lẻ
            min_size = max(5, len(df) // 100)  # ~1% tổng bình luận, tối thiểu 5

            # Cố định random_state để kết quả luôn nhất quán giữa các lần chạy
            umap_model = UMAP(
                n_neighbors=10,
                n_components=5,
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )
            hdbscan_model = HDBSCAN(
                min_cluster_size=min_size,
                min_samples=3,
                metric='euclidean',
                prediction_data=True
            )

            topic_model = BERTopic(
                embedding_model=sentence_model,
                vectorizer_model=vectorizer_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
            )

            topics, _ = topic_model.fit_transform(docs, embeddings)

            # Gắn nhãn
            topic_labels = topic_model.generate_topic_labels(nr_words=3, topic_prefix=True, separator=", ")
            topic_model.set_topic_labels(topic_labels)

            topic_info = topic_model.get_topic_info()
            if 'CustomName' in topic_info.columns:
                label_mapping = dict(zip(topic_info['Topic'], topic_info['CustomName']))
            else:
                label_mapping = dict(zip(topic_info['Topic'], topic_info['Name']))

            df['Topic'] = topics
            df['Topic_Name'] = df['Topic'].map(label_mapping)

            # --- Bước 4: Sentiment Analysis (Optional) ---
            if use_sentiment:
                status_text.info("Đang chạy mô hình Học máy để phân tích cảm xúc (Sentiment Analysis)...")
                s_model, s_vectorizer = load_sentiment_models()
                if s_model and s_vectorizer:
                    X_pred = s_vectorizer.transform(df['tokenized_text'])
                    preds = s_model.predict(X_pred)
                    df['Sentiment'] = preds
                    df['Sentiment_Label'] = df['Sentiment'].map({0: '🔴 Tiêu cực', 1: '⚪ Trung tính', 2: '🟢 Tích cực'}).fillna('❓ Không xác định')
                else:
                    st.warning("Không tìm thấy mô hình Cảm xúc. Vui lòng chạy file train_sentiment.py trước!")
                    use_sentiment = False

            progress_bar.progress(100)
            status_text.success("✅ Phân tích hoàn tất!")

            # --- Result UI ---
            st.markdown("---")

            # === Dashboard Metric Cards ===
            valid_topics_info = topic_info[topic_info['Topic'] != -1]
            outlier_count = len(df[df['Topic'] == -1])
            outlier_pct = outlier_count / len(df) * 100

            st.success(f"🎉 Đã phân tích xong **{len(df)}** bình luận hợp lệ!")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("💬 Tổng bình luận", f"{len(df):,}")
            m2.metric("📌 Số chủ đề tìm được", f"{len(valid_topics_info)}")
            m3.metric("🔵 Bình luận có chủ đề", f"{len(df) - outlier_count:,}")
            m4.metric("⚫ Bình luận ngoại lệ (outlier)", f"{outlier_count:,}", delta=f"-{outlier_pct:.1f}%", delta_color="inverse")

            st.markdown("---")

            # === Biểu đồ từ khóa ===
            st.subheader("📊 Biểu đồ Từ khóa theo Chủ đề")
            try:
                fig_barchart = topic_model.visualize_barchart(top_n_topics=8)
                st.plotly_chart(fig_barchart, use_container_width=True)
            except Exception as e:
                st.info(f"Chưa đủ dữ liệu chủ đề để vẽ biểu đồ Barchart. (Chi tiết: {e})")

            # === Biểu đồ phân bố số bình luận theo chủ đề ===
            st.subheader("📈 Phân bố Bình luận theo Chủ đề")
            try:
                dist_df = valid_topics_info[['Name', 'Count']].copy()
                dist_df = dist_df.sort_values('Count', ascending=True).tail(15)

                fig_dist = go.Figure(go.Bar(
                    x=dist_df['Count'],
                    y=dist_df['Name'],
                    orientation='h',
                    marker=dict(
                        color=dist_df['Count'],
                        colorscale='Viridis',
                        showscale=False
                    ),
                    text=dist_df['Count'],
                    textposition='outside'
                ))
                fig_dist.update_layout(
                    xaxis_title="Số bình luận",
                    yaxis_title="",
                    height=max(300, len(dist_df) * 40),
                    margin=dict(l=10, r=40, t=20, b=20),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            except Exception as e:
                st.info(f"Không thể vẽ biểu đồ phân bố chủ đề. (Chi tiết: {e})")

            # === Sentiment Pie ===
            if use_sentiment:
                st.markdown("---")
                st.subheader("❤️ Trạng thái Cảm xúc chung (Sentiment)")
                sentiment_counts = df['Sentiment_Label'].value_counts().reset_index()
                sentiment_counts.columns = ['label', 'count']

                color_map = {'🔴 Tiêu cực': 'red', '⚪ Trung tính': 'gray', '🟢 Tích cực': 'green'}
                labels_list = sentiment_counts['label'].tolist()
                counts_list = [int(x) for x in sentiment_counts['count'].tolist()]
                pie_colors = [color_map.get(l, 'blue') for l in labels_list]

                fig_pie = go.Figure(data=[go.Pie(
                    labels=labels_list,
                    values=counts_list,
                    hole=0.4,
                    marker=dict(colors=pie_colors),
                    textinfo='percent+label'
                )])
                st.plotly_chart(fig_pie, use_container_width=True, theme=None)

            # === Bảng chi tiết với bộ lọc outlier ===
            st.markdown("---")
            st.subheader("📋 Bảng dữ liệu phân loại chi tiết")

            hide_outliers = st.checkbox("Ẩn bình luận ngoại lệ (outlier / không xác định chủ đề)", value=True)

            cols_to_show = ['Topic_Name', 'author', 'text', 'like_count']
            if use_sentiment:
                cols_to_show.insert(1, 'Sentiment_Label')

            df_display = df.copy()
            if hide_outliers:
                df_display = df_display[df_display['Topic'] != -1]

            df_display = df_display[cols_to_show].sort_values(by=['Topic_Name'])
            st.caption(f"Hiển thị **{len(df_display):,}** bình luận" + (" (đã ẩn outlier)" if hide_outliers else " (bao gồm outlier)"))
            st.dataframe(df_display, use_container_width=True, height=500)

            # Nút tải xuống kết quả CSV
            csv_data = df_display.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(
                label="📥 Tải xuống kết quả phân tích (CSV)",
                data=csv_data,
                file_name="ket_qua_phan_tich.csv",
                mime="text/csv",
                use_container_width=True
            )

            # --- Ollama Summarization ---
            if use_ollama:
                from openai import OpenAI

                st.markdown("---")
                st.subheader("🤖 AI Giải nghĩa Chủ đề (Sinh bởi Ollama)")

                ollama_status = st.empty()
                ollama_status.info("Đang gọi Ollama để đọc bình luận và tóm tắt... Quá trình này nhanh hay chậm tùy thuộc vào cấu hình máy của bạn.")

                client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

                valid_topics = topic_info[topic_info['Topic'] != -1].head(5)

                for idx, row in valid_topics.iterrows():
                    t_id = row['Topic']
                    t_name = label_mapping.get(t_id, row['Name'])

                    rep_docs = row['Representative_Docs']
                    if not isinstance(rep_docs, list):
                        rep_docs = [rep_docs]

                    comments_text = "\n- ".join([d for d in rep_docs if isinstance(d, str)][:10])
                    prompt = f"Bạn là một chuyên gia phân tích cảm xúc mạng xã hội. Hãy đọc các bình luận YouTube sau đây và viết ĐÚNG 1 CÂU NGẮN GỌN bằng tiếng Việt để tóm tắt xem mọi người đang bàn luận chung về vấn đề gì.\n\nCác bình luận:\n- {comments_text}\n\nCâu tóm tắt:"

                    try:
                        response = client.chat.completions.create(
                            model=ollama_model,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=150,
                            temperature=0.3
                        )
                        summary = response.choices[0].message.content.strip()
                        st.success(f"**Chủ đề: [{t_name}]**\n\n💡 {summary}")
                    except Exception as e:
                        st.error(f"Lỗi khi xử lý chủ đề {t_name} với Ollama: {e}. Vui lòng kiểm tra xem Ollama có đang chạy không và tên model có đúng không.")

                ollama_status.success("✅ Đã hoàn tất việc giải nghĩa bằng AI!")

        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")
