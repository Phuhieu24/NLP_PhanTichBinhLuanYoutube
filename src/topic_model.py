"""Công cụ dòng lệnh chạy pipeline gom cụm chủ đề trên một file CSV có sẵn.

Dùng khi muốn thử nghiệm ngoài giao diện web (ví dụ chạy trên tập dữ liệu
huấn luyện để lấy số liệu cho báo cáo). Mọi bước tính toán đều gọi cùng một
module `topic_pipeline` mà `app.py` đang dùng, nên kết quả hai bên khớp nhau.

Ví dụ:
    python src/topic_model.py --input data/dataset_chuan.csv --text-col text \
        --limit 1500 --min-topic-size 15 --min-samples 1 --out-dir /tmp/topics
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import topic_pipeline as tp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gom cụm chủ đề bình luận tiếng Việt bằng BERTopic. Khuyến nghị chạy với "
            "--min-topic-size 15 --min-samples 1: với tham số mặc định của BERTopic, "
            "1.496 bình luận mẫu bị gom 91% vào một chủ đề duy nhất."
        )
    )
    parser.add_argument("--input", default=os.path.join("data", "comments.csv"),
                        help="File CSV đầu vào (mặc định: data/comments.csv, tức kết quả "
                             "thô của crawler; bước làm sạch chạy lại từ cột văn bản gốc).")
    parser.add_argument("--text-col", default="text", help="Tên cột chứa văn bản gốc.")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ lấy N dòng đầu tiên.")
    parser.add_argument("--min-topic-size", type=int, default=10,
                        help="Kích thước cụm tối thiểu (mặc định 10 của BERTopic; nên đặt 15).")
    parser.add_argument("--nr-topics", type=int, default=None, help="Ép về N chủ đề sau khi gom cụm.")
    parser.add_argument("--no-stopwords", action="store_true", help="Không loại bỏ từ dừng tiếng Việt.")
    parser.add_argument("--cluster-selection", choices=["eom", "leaf"], default="eom",
                        help="Cách HDBSCAN chọn cụm: 'eom' (ít cụm lớn) hoặc 'leaf' (nhiều cụm nhỏ).")
    parser.add_argument("--min-samples", type=int, default=None,
                        help="Tham số min_samples của HDBSCAN (mặc định bằng "
                             "min_topic_size; nên đặt 1 để bớt nhiễu gom vào một cụm).")
    parser.add_argument("--out-dir", default="results", help="Thư mục ghi kết quả.")
    parser.add_argument("--seed", type=int, default=42, help="Seed cho UMAP.")
    return parser


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def main(argv=None) -> int:
    if sys.stdout is not None and getattr(sys.stdout, "encoding", "") != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = build_parser().parse_args(argv)
    input_path = _resolve(args.input)
    out_dir = _resolve(args.out_dir)

    if not os.path.exists(input_path):
        print(f"Lỗi: không tìm thấy file dữ liệu tại {input_path}")
        return 1
    os.makedirs(out_dir, exist_ok=True)

    print(f"1. Đọc dữ liệu từ {input_path}")
    df = pd.read_csv(input_path)
    if args.text_col not in df.columns:
        print(f"Lỗi: không có cột '{args.text_col}'. Các cột hiện có: {list(df.columns)}")
        return 1
    if args.limit:
        df = df.head(args.limit)

    print("2. Làm sạch và tách từ tiếng Việt")
    df = tp.prepare_documents(df, text_col=args.text_col)
    docs = df["tokenized_text"].tolist()
    print(f"   Còn {len(docs)} bình luận hợp lệ")
    if len(docs) < 10:
        print("Lỗi: cần ít nhất 10 bình luận hợp lệ để gom cụm.")
        return 1

    print(f"3. Nhúng câu bằng {tp.DEFAULT_EMBEDDING_MODEL}")
    model = tp.load_embedding_model()
    embeddings = tp.embed_documents(model, docs)

    cfg = tp.TopicConfig(
        min_topic_size=args.min_topic_size,
        nr_topics=args.nr_topics,
        random_state=args.seed,
        use_stopwords=not args.no_stopwords,
        cluster_selection_method=args.cluster_selection,
        min_samples=args.min_samples,
    )
    print(f"4. Gom cụm bằng BERTopic ({cfg.cluster_selection_method}, "
          f"min_topic_size={cfg.min_topic_size})")
    topic_model = tp.build_topic_model(model, cfg)
    topics, topic_model = tp.fit_topics(topic_model, docs, embeddings)
    df["Topic"] = topics
    topics_df = tp.topic_table(topic_model, df, docs, text_col=args.text_col)
    df["Topic_Name"] = df["Topic"].map(dict(zip(topics_df["Topic"], topics_df["Name"])))

    comments_path = os.path.join(out_dir, "comments_with_topics.csv")
    df.to_csv(comments_path, index=False, encoding="utf-8-sig")
    summary = topics_df.copy()
    summary["Keywords"] = summary["Keywords"].apply(lambda words: ", ".join(words))
    summary["Representative"] = summary["Representative"].apply(lambda items: " | ".join(items[:3]))
    summary_path = os.path.join(out_dir, "topics_summary.csv")
    summary[["Topic", "Count", "Keywords", "Representative"]].to_csv(
        summary_path, index=False, encoding="utf-8-sig"
    )

    print("5. Vẽ biểu đồ HTML")
    charts = (
        ("topic_barchart.html", lambda: topic_model.visualize_barchart(top_n_topics=10)),
        ("intertopic_distance_map.html", lambda: topic_model.visualize_topics()),
        ("topic_hierarchy.html", lambda: topic_model.visualize_hierarchy(top_n_topics=15)),
    )
    for filename, builder in charts:
        try:
            builder().write_html(os.path.join(out_dir, filename))
        except Exception as error:
            print(f"   Bỏ qua {filename}: {error}")

    n_topics = int((topics_df["Topic"] != -1).sum())
    print(f"\n--- Kết quả: {n_topics} chủ đề, "
          f"{tp.outlier_share(topics):.1f}% bình luận là nhiễu (-1) ---")
    for _, row in topics_df[topics_df["Topic"] != -1].head(10).iterrows():
        print(f"  [{int(row['Topic']):>3}] {int(row['Count']):>5} bình luận | "
              f"{', '.join(row['Keywords'][:8])}")
    print(f"\nĐã ghi: {comments_path}")
    print(f"Đã ghi: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
