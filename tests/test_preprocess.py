"""Kiểm thử cho module tiền xử lý văn bản tiếng Việt."""

import os
import unicodedata

import pytest

import preprocess
from preprocess import (
    STOPWORDS_PATH,
    TEENCODE_DICT,
    clean_text,
    get_vietnamese_stopwords,
    normalize_repeated_chars,
    normalize_teencode,
    preprocess_text,
    tokenize_vietnamese,
)


# --- rút gọn ký tự lặp ---


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("luônnnn", "luôn"),
        ("hayyy", "hay"),
        ("traiiiiii", "trai"),
        ("đỉnhhhhh", "đỉnh"),
        ("hay", "hay"),          # không đủ 3 ký tự lặp thì giữ nguyên
        ("xoong", "xoong"),      # 2 ký tự lặp là chính tả hợp lệ
    ],
)
def test_normalize_repeated_chars(raw, expected):
    assert normalize_repeated_chars(raw) == expected


def test_normalize_repeated_chars_khong_dung_den_chu_so():
    assert normalize_repeated_chars("1000 nghìn 5555") == "1000 nghìn 5555"


def test_normalize_repeated_chars_giu_dau_gach_duoi():
    assert normalize_repeated_chars("a___b") == "a___b"


@pytest.mark.parametrize("raw", ["kkkk", "hhhh", "zzzz", "KKKK"])
def test_normalize_repeated_chars_bo_han_tu_chi_gom_mot_chu_cai_lap(raw):
    # Tiếng cười hoặc tạp âm: bỏ hẳn, vì rút về "k" sẽ bị teencode đổi thành
    # "không", tức là thêm nghĩa phủ định không có trong bình luận.
    assert normalize_repeated_chars(raw) == ""


def test_normalize_repeated_chars_bo_tieng_cuoi_giua_cau():
    assert normalize_repeated_chars("kkkk hài quá") == "hài quá"
    assert normalize_repeated_chars("hài kkkk quá") == "hài quá"


def test_normalize_repeated_chars_van_rut_gon_trong_tu_dai_hon():
    # Chuỗi lặp nằm trong từ dài hơn vẫn rút về một ký tự như cũ.
    assert normalize_repeated_chars("luônnnn") == "luôn"
    assert normalize_repeated_chars("hayyy") == "hay"
    assert normalize_repeated_chars("traiiiii") == "trai"


# --- teencode ---


def test_normalize_teencode_theo_tu_tron_ven_va_khong_phan_biet_hoa_thuong():
    assert normalize_teencode("K hiểu") == "không hiểu"
    assert normalize_teencode("Ko thích") == "không thích"
    assert normalize_teencode("ĐC không") == "được không"


def test_normalize_teencode_khong_dung_vao_tu_khac():
    # "kho", "kia", "ta" chứa ký tự của khóa nhưng không phải là khóa trọn vẹn
    for word in ["kho", "kia", "khoai", "tao", "top"]:
        assert normalize_teencode(word) == word


def test_normalize_teencode_giu_nguyen_kieu_viet_hoa_cua_tu_khong_doi():
    assert normalize_teencode("Đông Hùng hát") == "Đông Hùng hát"


def test_teencode_khong_doi_tieng_long_thanh_tu_cam_xuc():
    # Nguyên tắc: không dịch tiếng lóng/chửi thề thành từ mang sắc thái cảm xúc.
    for token in ["ok", "sp", "vcl", "vl", "đkm", "vlol"]:
        assert token not in TEENCODE_DICT
        assert normalize_teencode(token) == token


def test_teencode_dict_toan_khoa_chu_thuong():
    assert all(key == key.lower() for key in TEENCODE_DICT)


# --- clean_text ---


def test_clean_text_xoa_url():
    out = clean_text("Xem ở đây https://youtu.be/abc123 nhé")
    assert "http" not in out and "youtu" not in out
    assert out == "Xem ở đây nhé"


def test_clean_text_xoa_emoji_va_dau_cau():
    out = clean_text("Hay quá 😭😭 !!! ???")
    assert out == "Hay quá"


def test_clean_text_giu_nguyen_chu_hoa():
    # pyvi cần chữ hoa để ghép tên riêng, nên clean_text KHÔNG hạ chữ thường.
    assert clean_text("Đông Hùng Hát Rất Hay") == "Đông Hùng Hát Rất Hay"


def test_clean_text_bo_ky_hieu_mention_va_hashtag_giu_lai_chu():
    assert clean_text("@Hieu2ndd hát hay #AnhTraiSayHi") == "Hieu2ndd hát hay AnhTraiSayHi"


def test_clean_text_xu_ly_dau_cau_truoc_teencode():
    # "k." -> "k" -> "không"
    assert clean_text("k. hiểu sao") == "không hiểu sao"


def test_clean_text_gia_tri_khong_phai_chuoi_tra_ve_rong():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""
    assert clean_text(123) == ""


def test_clean_text_gop_khoang_trang_thua():
    assert clean_text("  hay    quá   ") == "hay quá"


def test_clean_text_chuan_hoa_unicode_ve_nfc():
    # Bình luận gõ trên macOS/iOS có thể ở dạng NFD ("ô" là "o" + U+0302).
    nfd = unicodedata.normalize("NFD", "không hiểu luônnnn")
    assert nfd != "không hiểu luônnnn"
    assert clean_text(nfd) == clean_text("không hiểu luônnnn")
    assert "luôn" in clean_text(nfd)


def test_clean_text_bo_dau_gach_duoi_nguoi_dung_go():
    # pyvi tự thêm dấu gạch dưới ở bước tách từ; dấu người dùng gõ phải biến mất.
    assert clean_text("anh_trai say hi") == "anh trai say hi"
    assert clean_text("__init__ hay") == "init hay"


def test_clean_text_bo_tieng_cuoi():
    assert clean_text("kkkk hài quá") == "hài quá"


# --- tokenize_vietnamese và preprocess_text ---


def test_tokenize_vietnamese_ghep_ten_rieng_roi_moi_ha_chu_thuong():
    out = tokenize_vietnamese("Đông Hùng hát")
    assert "đông_hùng" in out
    assert out == out.lower()


def test_tokenize_vietnamese_chuoi_rong():
    assert tokenize_vietnamese("") == ""


def test_preprocess_text_toan_trinh():
    out = preprocess_text("Đông Hùng hát bài này hay quá trời luônnnn 😭 k hiểu sao top 4")
    assert "đông_hùng" in out
    assert "luôn" in out and "luônnnn" not in out
    assert "không hiểu" in out
    assert out == out.lower()


def test_preprocess_text_bo_tieng_cuoi_thay_vi_doi_thanh_phu_dinh():
    assert preprocess_text("kkkk hài quá") == "hài quá"
    out = preprocess_text("hahaha kkk hay quá")
    assert "hay" in out
    assert "không" not in out


def test_preprocess_text_dau_gach_duoi_nguoi_dung_go_khong_vao_token():
    assert preprocess_text("__init__ hay") == "init hay"


def test_preprocess_text_gia_tri_thieu():
    assert preprocess_text(float("nan")) == ""
    assert preprocess_text(None) == ""
    assert preprocess_text("") == ""


# --- danh sách từ dừng ---


def test_stopwords_path_ton_tai_va_tinh_theo_file():
    assert os.path.isabs(STOPWORDS_PATH)
    assert os.path.exists(STOPWORDS_PATH)
    assert STOPWORDS_PATH.startswith(os.path.dirname(os.path.abspath(preprocess.__file__)))


def test_stopwords_dinh_dang_hop_le():
    words = get_vietnamese_stopwords()
    assert len(words) >= 150
    assert words == sorted(words)
    assert len(words) == len(set(words))
    assert all(word == word.lower() for word in words)
    assert all(" " not in word for word in words)
    assert all(word.strip() == word and word for word in words)
    assert not any(word.startswith("#") for word in words)


def test_stopwords_chua_hu_tu_va_cum_da_tach_tu():
    words = get_vietnamese_stopwords()
    for word in ["và", "như_thế_nào", "tuy_nhiên", "có_thể", "bởi_vì", "bao_giờ", "ơi", "nhé"]:
        assert word in words


def test_stopwords_khong_chua_tu_noi_dung():
    words = get_vietnamese_stopwords()
    for word in ["hát", "bài", "nhạc", "đỉnh", "hay", "thích", "buồn"]:
        assert word not in words


def test_stopwords_co_cache_va_tra_ve_ban_sao():
    first = get_vietnamese_stopwords()
    first.append("token_thu_nghiem")
    second = get_vietnamese_stopwords()
    assert "token_thu_nghiem" not in second
