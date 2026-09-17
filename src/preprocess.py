"""Tiền xử lý văn bản tiếng Việt cho bình luận YouTube.

Thứ tự xử lý (quan trọng): làm sạch -> tách từ bằng pyvi -> hạ chữ thường.
pyvi phân biệt chữ hoa/chữ thường, nên nếu hạ chữ thường trước khi tách từ thì
tên riêng sẽ không được ghép lại: "Đông Hùng hát" -> "Đông_Hùng hát" (đúng),
còn "đông hùng hát" -> "đông hùng hát" (sai). Vì vậy `clean_text` giữ nguyên
chữ hoa, và `tokenize_vietnamese` mới hạ chữ thường sau khi tách từ.

Hàm dùng chung cho cả huấn luyện lẫn ứng dụng: `preprocess_text`.
"""

import functools
import os
import re
import sys

import pandas as pd
from pyvi import ViTokenizer

# Nguyên tắc duy nhất của từ điển teencode: CHỈ chuẩn hóa các biến thể chính tả
# của cùng một từ (viết tắt, thiếu dấu, viết theo kiểu chat) về dạng chuẩn.
# Tuyệt đối không dịch tiếng lóng hay từ chửi thề thành một từ mang sắc thái
# cảm xúc (ví dụ "vcl" -> "rất", "đkm" -> "tồi"): làm vậy là tự gán nhãn cảm xúc
# cho dữ liệu, gây thiên lệch cho mô hình. Những token đó được giữ nguyên để bộ
# phân loại tự học trọng số từ dữ liệu thật.
# Khóa luôn viết thường; tra cứu theo từ trọn vẹn và không phân biệt hoa/thường.
TEENCODE_DICT = {
    # phủ định
    "k": "không",
    "kh": "không",
    "ko": "không",
    "hok": "không",
    "hong": "không",
    "khong": "không",
    "kb": "không biết",
    # đại từ, xưng hô
    "t": "tôi",
    "mk": "mình",
    "mik": "mình",
    "mn": "mọi người",
    "ae": "anh em",
    "ck": "chồng",
    "vk": "vợ",
    # hư từ, trạng từ
    "đc": "được",
    "dc": "được",
    "cx": "cũng",
    "vs": "với",
    "ms": "mới",
    "trc": "trước",
    "nx": "nữa",
    "r": "rồi",
    "lun": "luôn",
    "z": "vậy",
    "dz": "vậy",
    "zậy": "vậy",
    "qá": "quá",
    "wá": "quá",
    "wa": "quá",
    "nhìu": "nhiều",
    # từ để hỏi
    "j": "gì",
    "ntn": "như thế nào",
    "bh": "bao giờ",
    "bn": "bao nhiêu",
    # từ nội dung viết tắt
    "bt": "bình thường",
    "iu": "yêu",
    "thik": "thích",
    "bít": "biết",
    "dth": "dễ thương",
    # biến thể chính tả của "ok" (giữ nguyên nghĩa, không đổi thành "tốt")
    "oke": "ok",
    "okie": "ok",
    "okê": "ok",
}

# Một mẫu regex duy nhất cho cả từ điển; khóa dài xếp trước để không khớp nhầm
# phần đầu của khóa dài hơn ("kb" phải thắng "k").
_TEENCODE_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(re.escape(key) for key in sorted(TEENCODE_DICT, key=len, reverse=True))
    + r")\b",
    flags=re.IGNORECASE,
)

# Từ 3 ký tự chữ cái giống nhau trở lên rút về một ký tự ("luônnnn" -> "luôn").
# [^\W\d_] là ký tự chữ cái Unicode (đã loại chữ số và dấu gạch dưới) nên "1000"
# giữ nguyên. Khoảng 21% bình luận trong tập dữ liệu có hiện tượng lặp ký tự.
_REPEATED_CHARS_PATTERN = re.compile(r"([^\W\d_])\1{2,}")

_URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)

# Bỏ ký hiệu @ và # nhưng giữ lại chữ đứng sau ("#anhtraisayhi" -> "anhtraisayhi").
_MENTION_HASHTAG_PATTERN = re.compile(r"[@#](\w+)")

_NON_WORD_PATTERN = re.compile(r"[^\w\s]")

_WHITESPACE_PATTERN = re.compile(r"\s+")

# Danh sách từ dừng tiếng Việt, ở dạng đã tách từ (có dấu gạch dưới).
# Đường dẫn tính theo __file__ để chạy được từ streamlit, từ script và từ pytest.
STOPWORDS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "resources", "vietnamese_stopwords.txt"
)


def normalize_repeated_chars(text: str) -> str:
    """Rút gọn chuỗi 3 ký tự chữ cái giống nhau trở lên về một ký tự.

    "luônnnn" -> "luôn", "hayyyy" -> "hay". Chữ số không bị đụng tới nên "1000"
    vẫn là "1000".
    """
    return _REPEATED_CHARS_PATTERN.sub(r"\1", text)


def normalize_teencode(text: str) -> str:
    """Chuẩn hóa teencode theo từ trọn vẹn, không phân biệt hoa/thường.

    Chỉ những từ có trong `TEENCODE_DICT` mới bị thay thế; các từ khác giữ
    nguyên cả nội dung lẫn kiểu viết hoa ("kho" không bị đổi thành "không").
    """
    return _TEENCODE_PATTERN.sub(lambda m: TEENCODE_DICT[m.group(0).lower()], text)


def clean_text(text) -> str:
    """Làm sạch văn bản thô, GIỮ NGUYÊN chữ hoa (pyvi cần chữ hoa để ghép tên riêng).

    Các bước: bỏ URL -> bỏ ký hiệu @ và # (giữ lại chữ) -> thay dấu câu, ký tự
    đặc biệt và emoji bằng khoảng trắng -> rút gọn ký tự lặp -> chuẩn hóa
    teencode -> gộp khoảng trắng thừa. Giá trị không phải chuỗi (NaN, None)
    trả về chuỗi rỗng.
    """
    if not isinstance(text, str):
        return ""

    text = _URL_PATTERN.sub(" ", text)
    text = _MENTION_HASHTAG_PATTERN.sub(r"\1", text)
    # \w của Python 3 đã hỗ trợ Unicode nên chữ tiếng Việt có dấu được giữ lại.
    text = _NON_WORD_PATTERN.sub(" ", text)
    # Rút gọn ký tự lặp trước khi tra teencode để "kooo" -> "ko" -> "không",
    # và bỏ dấu câu trước đó để "k." -> "k" -> "không".
    text = normalize_repeated_chars(text)
    text = normalize_teencode(text)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def tokenize_vietnamese(text: str) -> str:
    """Tách từ bằng pyvi rồi mới hạ chữ thường.

    Thứ tự này giữ được cụm tên riêng: "Đông Hùng hát" -> "đông_hùng hát".
    Đây cũng là định dạng đầu vào mà PhoBERT (vietnamese-sbert) mong đợi.
    """
    if not text:
        return ""
    return ViTokenizer.tokenize(text).lower()


def preprocess_text(text) -> str:
    """Quy trình tiền xử lý đầy đủ cho một bình luận: làm sạch rồi tách từ."""
    return tokenize_vietnamese(clean_text(text))


@functools.lru_cache(maxsize=1)
def _load_stopwords() -> tuple:
    """Đọc và chuẩn hóa file từ dừng (có cache nên chỉ đọc đĩa một lần)."""
    words = set()
    with open(STOPWORDS_PATH, encoding="utf-8") as f:
        for line in f:
            entry = line.strip().lower()
            if not entry or entry.startswith("#"):
                continue
            words.add(entry)
    return tuple(sorted(words))


def get_vietnamese_stopwords() -> list:
    """Trả về danh sách từ dừng tiếng Việt, đã sắp xếp và loại trùng.

    Các mục ở dạng đã tách từ (dấu gạch dưới cho cụm: "như_thế_nào"), viết
    thường, khớp đúng với token do `tokenize_vietnamese` sinh ra nên dùng được
    trực tiếp cho `CountVectorizer` của BERTopic.
    """
    # Trả bản sao để phía gọi có thể sửa danh sách mà không làm hỏng cache.
    return list(_load_stopwords())


if __name__ == "__main__":
    # Sửa lỗi in Unicode trên terminal Windows; có kiểm tra để không vỡ khi
    # stdout bị chuyển hướng hoặc không có thuộc tính encoding.
    if sys.stdout and getattr(sys.stdout, "encoding", None):
        if sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
            try:
                sys.stdout.reconfigure(encoding="utf-8")
            except (AttributeError, ValueError):
                pass

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(project_root, "data", "comments.csv")
    output_path = os.path.join(project_root, "data", "comments_clean.csv")

    if not os.path.exists(input_path):
        print(f"Lỗi: Không tìm thấy file dữ liệu tại {input_path}")
        sys.exit(1)

    print("Đang đọc dữ liệu từ comments.csv...")
    df = pd.read_csv(input_path)

    initial_len = len(df)
    print(f"Số lượng bình luận ban đầu: {initial_len}")

    print("Đang làm sạch văn bản (xóa URL, emoji, dấu câu, chuẩn hóa teencode)...")
    df["clean_text"] = df["text"].apply(clean_text)

    print("Đang tách từ bằng pyvi rồi hạ chữ thường...")
    df["tokenized_text"] = df["text"].apply(preprocess_text)

    print("Đang lọc bỏ các bình luận quá ngắn (< 2 từ)...")
    df = df[df["tokenized_text"].apply(lambda x: len(str(x).split()) >= 2)]

    final_len = len(df)
    print(
        f"Số lượng bình luận sau khi lọc: {final_len} "
        f"(Loại bỏ {initial_len - final_len} dòng không đủ độ dài hoặc bị rỗng)"
    )

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Đã lưu kết quả thành công vào: {output_path}")
