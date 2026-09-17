import pandas as pd
import re
import os
import sys
from pyvi import ViTokenizer

TEENCODE_DICT = {
    "ko": "không",
    "k": "không",
    "khong": "không",
    "đc": "được",
    "dc": "được",
    "ok": "tốt",
    "oke": "tốt",
    "cx": "cũng",
    "sp": "sản phẩm",
    "mk": "mình",
    "mik": "mình",
    "ntn": "như thế nào",
    "vs": "với",
    "ms": "mới",
    "kb": "không biết",
    "vcl": "rất",
    "vl": "rất",
    "đkm": "tồi",
    "vlol": "tồi"
}

def normalize_teencode(text):
    words = text.split()
    normalized_words = [TEENCODE_DICT.get(word, word) for word in words]
    return " ".join(normalized_words)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Chuyển về chữ thường
    text = text.lower()
    
    # 2. Xóa URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 3. Xóa ký tự đặc biệt, dấu câu, emoji (chỉ giữ lại chữ cái, số và khoảng trắng)
    # \w trong Python 3 mặc định hỗ trợ Unicode tiếng Việt
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 4. Chuẩn hóa teencode
    text = normalize_teencode(text)
    
    # 5. Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_vietnamese(text):
    if not text:
        return ""
    # Tách từ tiếng Việt bằng pyvi
    return ViTokenizer.tokenize(text)

if __name__ == "__main__":
    # Fix unicode printing on Windows terminal
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
        
    project_root = os.path.dirname(os.path.dirname(__file__))
    input_path = os.path.join(project_root, 'data', 'comments.csv')
    output_path = os.path.join(project_root, 'data', 'comments_clean.csv')
    
    if not os.path.exists(input_path):
        print(f"Lỗi: Không tìm thấy file dữ liệu tại {input_path}")
        sys.exit(1)
        
    print("Đang đọc dữ liệu từ comments.csv...")
    df = pd.read_csv(input_path)
    
    initial_len = len(df)
    print(f"Số lượng bình luận ban đầu: {initial_len}")
    
    print("Đang làm sạch văn bản (xóa emoji, url, dấu câu)...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    print("Đang tách từ (Word Tokenization)...")
    df['tokenized_text'] = df['clean_text'].apply(tokenize_vietnamese)
    
    print("Đang lọc bỏ các bình luận quá ngắn (< 2 từ)...")
    # Đếm số từ dựa trên khoảng trắng của chuỗi đã tokenize
    df = df[df['tokenized_text'].apply(lambda x: len(str(x).split()) >= 2)]
    
    final_len = len(df)
    print(f"Số lượng bình luận sau khi lọc: {final_len} (Loại bỏ {initial_len - final_len} dòng không đủ độ dài hoặc bị rỗng)")
    
    # Lưu ra file mới
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Đã lưu kết quả thành công vào: {output_path}")
