"""Bản tiền xử lý GỐC ở commit `afb1c3e`, đóng băng để so sánh A/B.

Đây là bản sao nguyên văn phần tiền xử lý của `src/preprocess.py` tại commit
`afb1c3e` (trước khi hoàn thiện bước làm sạch). Tệp này chỉ tồn tại để
`experiments/ab_preprocess.py` đo được chênh lệch giữa bản gốc và bản đang xuất
xưởng; KHÔNG import tệp này ở bất kỳ đâu trong `src/`.

Giữ nguyên `TEENCODE_DICT`, `normalize_teencode`, `clean_text`,
`tokenize_vietnamese` đúng như bản gốc; chỉ lược bỏ khối `__main__` (đọc ghi
tệp `data/comments.csv`) vì thí nghiệm không cần tới.
"""

import re

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
