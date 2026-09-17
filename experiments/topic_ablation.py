"""Thí nghiệm gom cụm chủ đề: 5 cấu hình HDBSCAN trên cùng một bộ nhúng câu.

Chạy đúng pipeline đang xuất xưởng (`src/topic_pipeline.py`) nên số liệu ở đây
tái lập được bằng chính mã nguồn nộp bài. Bình luận được lấy từ N dòng ĐẦU TIÊN
của tệp dữ liệu (`--limit`, mặc định 1.500), làm sạch và tách từ bằng
`prepare_documents`, rồi nhúng MỘT lần và dùng lại cho cả 5 cấu hình: khác biệt
giữa các dòng kết quả chỉ đến từ tham số HDBSCAN.

Ngoài ra chạy thêm cấu hình khuyến nghị (15/1) với `use_stopwords=False` để kiểm
tra một khẳng định trong báo cáo: danh sách từ dừng chỉ đi vào `CountVectorizer`
ở bước tính c-TF-IDF, tức là SAU khi nhãn cụm đã cố định, nên nó không thể làm
thay đổi số chủ đề hay tỷ lệ nhiễu.

Kết quả ghi vào `results/topic_ablation.csv` và `results/topic_ablation.txt`.

Chạy:
    python experiments/topic_ablation.py --limit 1500
Thời gian: khoảng 30-60 giây (nhúng vài giây, mỗi lần khớp BERTopic 3-7 giây).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import topic_pipeline as tp  # noqa: E402

# 5 cấu hình: mặc định của BERTopic, cấu hình gây gom cụm gốc, cấu hình khuyến
# nghị của dự án, và hai biến thể 'leaf' để đối chứng.
CONFIGS = [
    ("eom (mặc định BERTopic)", "eom", 10, None),
    ("eom", "eom", 20, 5),
    ("eom (cấu hình khuyến nghị)", "eom", 15, 1),
    ("leaf", "leaf", 10, None),
    ("leaf", "leaf", 20, 5),
]

PRESET_INDEX = 2  # cấu hình 15/1, cũng là cấu hình đem đi kiểm tra từ dừng


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ablation tham số HDBSCAN cho pipeline chủ đề.")
    parser.add_argument("--input", default=os.path.join(PROJECT_ROOT, "data", "dataset_chuan.csv"),
                        help="Tệp CSV đầu vào.")
    parser.add_argument("--text-col", default="text", help="Tên cột văn bản gốc.")
    parser.add_argument("--limit", type=int, default=1500,
                        help="Chỉ lấy N dòng ĐẦU TIÊN của tệp (giống --limit của src/topic_model.py).")
    parser.add_argument("--seed", type=int, default=42, help="Hạt giống cho UMAP.")
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "results"),
                        help="Thư mục ghi kết quả.")
    return parser


def _versions() -> dict:
    import bertopic
    import hdbscan
    import umap

    def version_of(module):
        return getattr(module, "__version__", None) or "không rõ"

    try:  # hdbscan không khai báo __version__ ở mọi bản
        from importlib.metadata import version as dist_version

        hdbscan_version = dist_version("hdbscan")
    except Exception:  # pragma: no cover - chỉ để báo cáo
        hdbscan_version = version_of(hdbscan)
    return {
        "bertopic": version_of(bertopic),
        "umap-learn": version_of(umap),
        "hdbscan": hdbscan_version,
    }


def run_config(model, docs, embeddings, method, min_topic_size, min_samples, seed,
               use_stopwords=True) -> dict:
    cfg = tp.TopicConfig(
        min_topic_size=min_topic_size,
        min_samples=min_samples,
        cluster_selection_method=method,
        random_state=seed,
        use_stopwords=use_stopwords,
    )
    started = time.perf_counter()
    topic_model = tp.build_topic_model(model, cfg)
    topics, topic_model = tp.fit_topics(topic_model, docs, embeddings)
    elapsed = time.perf_counter() - started

    info = topic_model.get_topic_info()
    n_topics = int((info["Topic"] != -1).sum())
    n_noise = int(sum(1 for topic in topics if topic == -1))
    real = info[info["Topic"] != -1].sort_values("Count", ascending=False)
    if len(real):
        largest = real.iloc[0]
        largest_size = int(largest["Count"])
        largest_id = int(largest["Topic"])
        keywords = [str(word) for word in (largest["Representation"] or [])][:8]
    else:  # pragma: no cover - chỉ xảy ra khi mọi tài liệu đều là nhiễu
        largest_size, largest_id, keywords = 0, -1, []
    return {
        "cluster_selection_method": method,
        "min_topic_size": min_topic_size,
        "min_samples": "None" if min_samples is None else min_samples,
        "use_stopwords": use_stopwords,
        "n_topics": n_topics,
        "n_noise": n_noise,
        "noise_share_pct": round(100.0 * n_noise / len(topics), 4),
        "largest_topic_id": largest_id,
        "largest_topic_size": largest_size,
        "largest_topic_share_pct": round(100.0 * largest_size / len(topics), 4),
        "largest_topic_keywords": ", ".join(keywords),
        "fit_seconds": round(elapsed, 2),
        "topics": topics,
    }


def write_report(rows, meta, path) -> str:
    lines = [
        "ABLATION THAM SỐ HDBSCAN CHO PIPELINE CHỦ ĐỀ (mã nguồn đang xuất xưởng)",
        "=" * 92,
        f"Dữ liệu        : {meta['input']}, {meta['limit']} dòng đầu tiên "
        f"-> {meta['n_docs']} bình luận hợp lệ (>= 2 token)",
        f"Nhúng câu      : {meta['embedding_model']} ({meta['embed_seconds']:.1f}s cho toàn bộ, "
        "dùng lại cho mọi cấu hình)",
        f"UMAP           : n_neighbors={meta['n_neighbors']}, n_components={meta['n_components']}, "
        f"metric=cosine, random_state={meta['seed']}",
        f"Từ dừng        : {meta['n_stopwords']} mục, chỉ dùng cho CountVectorizer của c-TF-IDF",
        f"Phiên bản      : BERTopic {meta['versions']['bertopic']}, "
        f"umap-learn {meta['versions']['umap-learn']}, hdbscan {meta['versions']['hdbscan']}",
        "",
        f"{'chọn cụm':<28}{'min_topic_size':>16}{'min_samples':>13}{'chủ đề':>9}"
        f"{'nhiễu':>18}{'cụm lớn nhất':>20}{'thời gian':>11}",
        "-" * 92,
    ]
    for label, row in rows:
        noise = f"{row['n_noise']} ({row['noise_share_pct']:.1f}%)"
        largest = f"{row['largest_topic_size']} ({row['largest_topic_share_pct']:.1f}%)"
        fit = f"{row['fit_seconds']:.1f}s"
        lines.append(
            f"{label:<28}{row['min_topic_size']:>16}{str(row['min_samples']):>13}"
            f"{row['n_topics']:>9}{noise:>18}{largest:>20}{fit:>11}"
        )
    lines += ["", "Từ khóa của cụm lớn nhất từng cấu hình:"]
    for label, row in rows:
        lines.append(
            f"  {label} {row['min_topic_size']}/{row['min_samples']} "
            f"-> [{row['largest_topic_id']}] {row['largest_topic_keywords']}"
        )
    lines += [
        "",
        "KIỂM TRA DANH SÁCH TỪ DỪNG (cấu hình khuyến nghị 15/1)",
        "-" * 92,
        meta["stopword_note"],
        "",
    ]
    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return text


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)
    df = tp.prepare_documents(df, text_col=args.text_col)
    docs = df["tokenized_text"].tolist()
    print(f"{len(docs)} bình luận hợp lệ từ {args.limit} dòng đầu tiên")

    model = tp.load_embedding_model()
    started = time.perf_counter()
    embeddings = tp.embed_documents(model, docs)
    embed_seconds = time.perf_counter() - started
    print(f"Đã nhúng câu trong {embed_seconds:.1f}s")

    rows = []
    for label, method, min_topic_size, min_samples in CONFIGS:
        print(f"Đang chạy {label} {min_topic_size}/{min_samples}...")
        row = run_config(model, docs, embeddings, method, min_topic_size, min_samples, args.seed)
        print(f"   {row['n_topics']} chủ đề, {row['n_noise']} nhiễu "
              f"({row['noise_share_pct']:.1f}%), cụm lớn nhất {row['largest_topic_size']}")
        rows.append((label, row))

    preset_label, preset_method, preset_size, preset_min_samples = CONFIGS[PRESET_INDEX]
    print("Đang chạy lại cấu hình khuyến nghị với use_stopwords=False...")
    no_stop = run_config(model, docs, embeddings, preset_method, preset_size,
                         preset_min_samples, args.seed, use_stopwords=False)
    preset_topics = rows[PRESET_INDEX][1]["topics"]
    identical = list(no_stop["topics"]) == list(preset_topics)
    stopword_note = (
        "Tắt từ dừng (use_stopwords=False) cho cột Topic GIỐNG HỆT bản bật từ dừng "
        f"({no_stop['n_topics']} chủ đề, {no_stop['noise_share_pct']:.1f}% nhiễu); chỉ danh sách "
        "từ khóa đổi. Danh sách từ dừng chỉ đi vào CountVectorizer ở bước c-TF-IDF, tức là sau "
        "khi HDBSCAN đã cố định nhãn cụm, nên nó không thể làm đổi số chủ đề hay tỷ lệ nhiễu."
        if identical else
        "Tắt từ dừng cho cột Topic KHÁC bản bật từ dừng "
        f"({no_stop['n_topics']} chủ đề, {no_stop['noise_share_pct']:.1f}% nhiễu)."
    )
    rows.append((preset_label + " · tắt từ dừng", no_stop))
    print(stopword_note)

    csv_path = os.path.join(args.out_dir, "topic_ablation.csv")
    columns = ["label", "cluster_selection_method", "min_topic_size", "min_samples",
               "use_stopwords", "n_topics", "n_noise", "noise_share_pct", "largest_topic_id",
               "largest_topic_size", "largest_topic_share_pct", "largest_topic_keywords",
               "fit_seconds"]
    records = [{"label": label, **{key: row[key] for key in columns if key != "label"}}
               for label, row in rows]
    pd.DataFrame(records, columns=columns).to_csv(csv_path, index=False, encoding="utf-8-sig")

    meta = {
        "input": os.path.relpath(args.input, PROJECT_ROOT),
        "limit": args.limit,
        "n_docs": len(docs),
        "embedding_model": tp.DEFAULT_EMBEDDING_MODEL,
        "embed_seconds": embed_seconds,
        "seed": args.seed,
        "n_neighbors": tp.TopicConfig().n_neighbors,
        "n_components": tp.TopicConfig().n_components,
        "n_stopwords": len(tp._stopword_list(True) or []),
        "versions": _versions(),
        "stopword_note": stopword_note,
    }
    txt_path = os.path.join(args.out_dir, "topic_ablation.txt")
    print()
    print(write_report(rows[:len(CONFIGS)], meta, txt_path))
    print(f"Đã ghi: {csv_path}")
    print(f"Đã ghi: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
