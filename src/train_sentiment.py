import os
import pandas as pd
import numpy as np
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.pipeline import make_pipeline
from preprocess import clean_text, tokenize_vietnamese


def load_dataset(csv_path):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Đã tải {len(df)} dòng từ {csv_path}")
        return df
    else:
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {csv_path}.")

def main():
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(project_root, 'data')
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, 'dataset_chuan.csv')
    
    print("--- BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN (TRAINING) ---")
    
    # 1. Load Data
    df = load_dataset(csv_path)
    
    # 2. Tiền xử lý (Preprocessing)
    print("Đang tiền xử lý ngôn ngữ tự nhiên (Làm sạch và Tách từ)...")
    df['text_clean'] = df['text'].apply(clean_text).apply(tokenize_vietnamese)
    
    # 3. Chia tập dữ liệu (Train/Test Split) trước khi biến đổi đặc trưng để tránh rò rỉ dữ liệu (Data Leakage)
    print("Đang chia tập dữ liệu (Train/Test Split)...")
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    print(f"Số lượng mẫu huấn luyện (Train): {df_train.shape[0]}")
    print(f"Số lượng mẫu kiểm thử (Test): {df_test.shape[0]}")
    
    # 4. Trích xuất đặc trưng (Feature Extraction)
    print("Đang chuyển đổi văn bản thành Vector (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = vectorizer.fit_transform(df_train['text_clean'])
    y_train = df_train['label']
    
    X_test = vectorizer.transform(df_test['text_clean'])
    y_test = df_test['label']
    
    # 5. Huấn luyện mô hình (Training)
    print("Đang huấn luyện mô hình LinearSVC...")
    model = LinearSVC(class_weight='balanced', random_state=42, dual=False)
    model.fit(X_train, y_train)
    
    # 6. Đánh giá mô hình (Evaluation)
    print("Đang đánh giá mô hình bằng Cross-Validation (K-Fold = 5)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline_cv = make_pipeline(
        TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True),
        LinearSVC(class_weight='balanced', random_state=42, dual=False)
    )
    cv_scores = cross_val_score(pipeline_cv, df['text_clean'], df['label'], cv=cv, scoring='accuracy', n_jobs=-1)
    
    print("\n========== KẾT QUẢ ĐÁNH GIÁ (EVALUATION) ==========")
    print(f"Độ chính xác trung bình (Cross-Validation Accuracy): {cv_scores.mean() * 100:.2f}% (± {cv_scores.std() * 100:.2f}%)")

    print("\nĐang đánh giá chi tiết trên tập Test...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Độ chính xác trên tập Test (Accuracy): {acc * 100:.2f}%")
    print("Báo cáo phân loại chi tiết (Classification Report):")
    target_names = ["Tiêu cực (0)", "Trung tính (1)", "Tích cực (2)"]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print("Đang vẽ biểu đồ Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix trên tập Test')
    plt.ylabel('Nhãn thực tế')
    plt.xlabel('Nhãn dự đoán')
    
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu biểu đồ Confusion Matrix tại: {cm_path}")
    print("===================================================\n")
    
    # 7. Lưu mô hình (Export)
    model_path = os.path.join(models_dir, 'sentiment_model.pkl')
    vec_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    
    print(f"✅ Đã lưu mô hình thành công tại:\n - {model_path}\n - {vec_path}")
    print("--- HOÀN TẤT ---")

if __name__ == "__main__":
    main()
