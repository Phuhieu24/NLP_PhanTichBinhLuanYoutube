import pandas as pd
import os
import sys
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

if __name__ == "__main__":
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
        
    project_root = os.path.dirname(os.path.dirname(__file__))
    input_path = os.path.join(project_root, 'data', 'comments_clean.csv')
    
    if not os.path.exists(input_path):
        print(f"Lỗi: Không tìm thấy file dữ liệu tại {input_path}")
        sys.exit(1)
        
    print("1. Đang đọc dữ liệu...")
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['tokenized_text', 'clean_text', 'text'])
    docs = df['tokenized_text'].tolist()
    
    # Sử dụng biến thể của PhoBERT được tối ưu riêng cho Sentence Embedding
    print("2. Khởi tạo mô hình nhúng PhoBERT (vietnamese-sbert)...")
    sentence_model = SentenceTransformer("keepitreal/vietnamese-sbert")
    
    print("3. Đang mã hóa câu thành Vector (Quá trình này có thể tốn 1-3 phút)...")
    embeddings = sentence_model.encode(docs, show_progress_bar=True)
    
    print("4. Cấu hình vectorizer để giữ nguyên các từ ghép tiếng Việt...")
    VIETNAMESE_STOPWORDS = [
        "là", "và", "của", "có", "trong", "được", "cho", "với", "các", "này",
        "đã", "một", "những", "không", "như", "khi", "vào", "về", "từ", "thì",
        "mà", "để", "hay", "ra", "lại", "cũng", "đây", "đó", "thế", "nên",
        "rất", "cần", "bị", "vì", "tuy", "nhưng", "nếu", "hơn", "nhất", "rồi",
        "đến", "lên", "theo", "bởi", "qua", "sau", "trước", "trên", "dưới",
        "nào", "ai", "gì", "sao", "vậy", "đâu", "đi", "thôi", "thật", "ơi",
        "ạ", "à", "ừ", "uh", "ha", "he", "hihi", "haha", "hehe",
        "tôi", "mình", "bạn", "em", "anh", "chị", "họ", "nó", "ta", "chúng",
        "mọi", "người", "cái", "con", "bao", "nhiêu", "lắm", "quá", "vẫn",
        "thêm", "chỉ", "cả", "đều", "sẽ", "đang", "còn", "nữa", "mới",
        "phải", "muốn", "thấy", "biết", "làm", "nói", "nghĩ", "thích", "dùng",
    ]
    vectorizer_model = CountVectorizer(
        token_pattern=r'(?u)\b\w+\b',
        stop_words=VIETNAMESE_STOPWORDS,
        min_df=2
    )

    min_size = max(5, len(docs) // 100)

    print("5. Bắt đầu chạy thuật toán BERTopic (random_state=42 để kết quả nhất quán)...")
    umap_model = UMAP(
        n_neighbors=10, n_components=5, min_dist=0.0,
        metric='cosine', random_state=42
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_size, min_samples=3,
        metric='euclidean', prediction_data=True
    )
    topic_model = BERTopic(
        embedding_model=sentence_model,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=True
    )
    
    topics, probs = topic_model.fit_transform(docs, embeddings)
    
    print("\n--- Đã gom cụm xong! Thống kê các chủ đề ---")
    topic_info = topic_model.get_topic_info()
    print(topic_info.head(10))
    
    # Gắn nhãn tự động cho dễ nhìn
    topic_labels = topic_model.generate_topic_labels(nr_words=3, topic_prefix=True, separator=", ")
    topic_model.set_topic_labels(topic_labels)
    
    df['Topic'] = topics
    if 'CustomName' in topic_info.columns:
        label_mapping = dict(zip(topic_info['Topic'], topic_info['CustomName']))
    else:
        label_mapping = dict(zip(topic_info['Topic'], topic_info['Name']))
    df['Topic_Name'] = df['Topic'].map(label_mapping)
    
    output_csv = os.path.join(project_root, 'data', 'comments_with_topics.csv')
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\nĐã lưu kết quả phân loại từng bình luận vào: {output_csv}")
    
    print("\n6. Đang tạo các biểu đồ tương tác dạng Web (HTML)...")
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Biểu đồ cột từ khóa
    fig_barchart = topic_model.visualize_barchart(top_n_topics=10)
    fig_barchart.write_html(os.path.join(results_dir, "topic_barchart.html"))
    
    # Bản đồ không gian 2D của các chủ đề
    fig_intertopic = topic_model.visualize_topics()
    fig_intertopic.write_html(os.path.join(results_dir, "intertopic_distance_map.html"))
    
    # Cây phân tầng chủ đề
    fig_hierarchy = topic_model.visualize_hierarchy(top_n_topics=15)
    fig_hierarchy.write_html(os.path.join(results_dir, "topic_hierarchy.html"))
    
    print(f"Hoàn thành! Bạn có thể mở các file HTML trong thư mục {results_dir} bằng trình duyệt web để xem.")
