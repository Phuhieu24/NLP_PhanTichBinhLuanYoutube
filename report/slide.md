---
marp: true
theme: default
paginate: true
size: 16:9
---

# Phân tích chủ đề và cảm xúc bình luận YouTube tiếng Việt

Đồ án môn Xử lý ngôn ngữ tự nhiên

Nhóm:
- Lê Phú Hiếu | 26410038 | LT.K2026.1.TTNT	 
- Nguyễn Thanh Duy | 26410030 | LT.K2026.1.TTNT	 
- Nguyễn Thanh Phong | 26410090 | LT.K2026.1.TTNT	 
- Nguyễn Thị Mai Thi | 26410117 | LT.K2026.1.TTNT	 
- Hồ Viết Trịnh | 26410140 | LT.K2026.1.TTNT	 
- Nguyễn Ngọc Bích | 25730012 | CN1.K2025.1.TTNT	 
- Nguyễn Anh Tài | 25730063 | CN1.K2025.1.TTNT

<!-- Ghi chú: Thưa thầy và các bạn, nhóm em xin trình bày đồ án phân tích bình luận YouTube tiếng Việt. Hệ thống nhận bình luận dưới một video và trả về hai thứ: người xem đang bàn chủ đề gì, và họ khen hay chê. Bài nói khoảng 10 phút theo đúng năm bước của môn, sau đó nhóm em chạy ứng dụng trực tiếp 3 phút. Nếu chỉ nhớ một câu, xin thầy nhớ câu này: mọi con số trong bài đều đọc lại được từ thư mục results của mã nguồn. -->

---

## Bài toán: hàng nghìn bình luận, hai câu hỏi

- Đầu vào: toàn bộ bình luận dưới một video YouTube tiếng Việt
- Đầu ra 1: các chủ đề người xem bàn tới, mỗi chủ đề là một nhóm từ khóa và vài bình luận tiêu biểu
- Đầu ra 2: phân bố cảm xúc theo ba lớp tiêu cực, trung tính, tích cực
- Vì sao khó: teencode (`ko`, `đc`, `j`), không dấu, emoji trong 34.5% bình luận, câu ngắn (trung vị 52 ký tự), châm biếm và vừa khen vừa chê
- Tiếng Việt không có khoảng trắng giữa các từ, nên tách từ quyết định chất lượng cả hai đầu ra

<!-- Ghi chú: Bài toán xuất phát từ nhu cầu thật: một video có vài nghìn bình luận, không ai đọc nổi từng dòng. Nhóm em muốn trả lời hai câu: người xem đang nói về cái gì, và họ thấy thế nào. Dữ liệu này khó hơn văn bản báo chí. Một phần ba bình luận có emoji. Trung vị chỉ 52 ký tự, tức là một câu ngắn, ít ngữ cảnh. Và rất nhiều câu kiểu "hay mà tiếc", vừa khen vừa chê. Thêm cái khó riêng của tiếng Việt: từ ghép không có ranh giới, nên tách từ sai thì từ khóa chủ đề sai và đặc trưng phân loại cũng sai. -->

---

## Pipeline năm bước theo bài giảng, mỗi bước một module

![h:490](../docs/diagrams/pipeline.png)

Thu thập (`src/crawler.py`, ba nguồn dữ liệu), tiền xử lý (`src/preprocess.py`), biểu diễn (TF-IDF và vector câu), thuật toán (LinearSVC, BERTopic trong `src/topic_pipeline.py`), đánh giá (`results/`, `experiments/`)

<!-- Ghi chú: Nhóm em xếp đồ án theo đúng quy trình năm bước của bài 5 trong môn: thu thập và phân tích dữ liệu, tiền xử lý, biểu diễn, thuật toán, rồi đánh giá và phân tích lỗi. Sơ đồ này cho thấy mỗi bước nằm ở module nào trong mã nguồn. Điểm khác so với bài phân loại thuần: từ bước biểu diễn, pipeline tách làm hai nhánh. Nhánh cảm xúc dùng TF-IDF và LinearSVC. Nhánh chủ đề dùng vector câu và BERTopic. Hai nhánh gặp lại nhau ở ứng dụng, khi cảm xúc được hiển thị theo từng chủ đề. Phần còn lại của bài đi lần lượt qua năm bước này. -->

---

## Dữ liệu: 20,000 bình luận có nhãn về một chương trình

<style scoped>table { font-size: 0.8em; } ul { font-size: 0.86em; }</style>

| Chỉ số | Giá trị |
|---|---|
| Số dòng | 20,000 |
| Tiêu cực (0) | 7,000 (35.0%) |
| Trung tính (1) | 3,409 (17.0%) |
| Tích cực (2) | 9,591 (48.0%) |
| Dòng trùng nội dung | 0 |
| Dòng rỗng sau tiền xử lý | 3 (còn 19,997) |

- Nguồn: gói ATSH-NLP-20k, trích từ dự án ATSH-ABSA (Phạm Xuân Vĩnh Hà, UIT); bình luận về Anh Trai Say Hi mùa 1, tập 1 đến 14
- Nhãn silver: mô hình ngôn ngữ lớn gán theo đối tượng và khía cạnh rồi gộp; chưa kiểm tay, trung tính nhiễu nhất
- Tác giả lấy mẫu lại từ dữ liệu gốc khoảng 88% tích cực: giữ hết trung tính, lấy 7,000 tiêu cực, bù tích cực; tỉ lệ không phải tỉ lệ thật trên YouTube
- Một chương trình duy nhất, nên kết quả chỉ có giá trị trong miền đó
- Điều kiện của tác giả: chỉ dùng cho học tập, không công bố lại

<!-- Ghi chú: Tập dữ liệu có 20,000 bình luận, hai cột: văn bản và nhãn. Đây là hai cột text và label của gói ATSH-NLP-20k, trích từ dự án ATSH-ABSA của Phạm Xuân Vĩnh Hà ở UIT; nhóm em chỉ dùng cho học tập theo đúng điều kiện của tác giả. Ba lớp lệch nhau: tích cực gần một nửa, trung tính chỉ 17%. Con số 17% này sẽ quay lại ở phần kết quả, vì trung tính là lớp yếu nhất. Ba điều nhóm em nói thẳng. Thứ nhất, mọi bình luận đều về một chương trình, nên mô hình học cả tên thí sinh làm tín hiệu. Thứ hai, tỉ lệ ba lớp là do tác giả lấy mẫu lại: dữ liệu gốc khoảng 88% tích cực, tác giả giữ hết trung tính, lấy đúng 7,000 tiêu cực rồi bù tích cực cho đủ 20,000; vì vậy số 7,000 tròn, và tỉ lệ này không phải tỉ lệ thật trên YouTube. Thứ ba, nhãn là nhãn silver do mô hình ngôn ngữ lớn gán theo từng đối tượng và khía cạnh rồi gộp lại, chưa có người kiểm từng dòng; chính tác giả ghi trung tính là lớp nhiễu nhất. Phần đọc tay 150 bình luận ở cuối bài cũng là để kiểm chất lượng nhãn. -->

---

## Tiền xử lý: tách từ trước, hạ chữ thường sau

- Thứ tự trong mã: chuẩn hóa NFC, bỏ URL, bỏ ký hiệu `@` và `#`, thay dấu câu và emoji bằng khoảng trắng, rút chữ lặp (`luônnnn` thành `luôn`), chuẩn hóa teencode, rồi tách từ bằng pyvi, cuối cùng mới hạ chữ thường
- Vì sao thứ tự đó: pyvi phân biệt chữ hoa khi ghép tên riêng. `Đông Hùng hát` cho `Đông_Hùng hát`; đã hạ chữ thường thì thành ba token rời
- Từ điển teencode chỉ sửa chính tả (`ko`, `hok`, `khong` thành `không`), không dịch tiếng lóng sang từ cảm xúc. Bản cũ có `ok` thành `tốt`, `vcl` thành `rất`: đó là gán nhãn trước khi mô hình học
- 236 từ dừng chỉ dùng cho từ khóa chủ đề, vì `không` nằm trong danh sách và là đặc trưng mạnh nhất của lớp tiêu cực (+2.93)

<!-- Ghi chú: Bước tiền xử lý có ba quyết định. Một, tách từ trước rồi mới hạ chữ thường. Nhóm em kiểm tra trực tiếp: đưa "Đông Hùng hát" vào pyvi thì được tên riêng một token; đưa bản đã hạ chữ thường thì tên bị tách làm hai. Phiên bản đầu của đồ án làm ngược thứ tự, nên tên riêng chưa bao giờ được ghép. Hai, từ điển teencode. Bản cũ dịch "vcl" thành "rất", "ok" thành "tốt". Nghe tiện, nhưng thực ra là mình gán sắc thái cho dữ liệu trước khi mô hình được học. Nhóm em bỏ hết, chỉ giữ sửa chính tả, để bộ phân loại tự học từ 20,000 mẫu. Ba, từ dừng. Danh sách có chữ "không". Nếu loại từ dừng khỏi bộ phân loại cảm xúc thì mất luôn tín hiệu phủ định, nên danh sách này chỉ đi vào bước chọn từ khóa chủ đề. -->

---

## Đổi tiền xử lý không đổi điểm cảm xúc, nhưng đổi từ khóa chủ đề

| Tiền xử lý | C | Tỉ lệ dự đoán đúng (accuracy) | Macro-F1 | F1 trung tính |
|---|---|---|---|---|
| Cũ | 0.3 | 77.58% ± 0.47 | 0.7199 ± 0.0052 | 0.5218 |
| Mới | 0.3 | 77.58% ± 0.50 | 0.7200 ± 0.0041 | 0.5222 |
| Cũ | 1.0 | 76.61% ± 0.40 | 0.7079 ± 0.0052 | 0.5004 |
| Mới | 1.0 | 76.57% ± 0.43 | 0.7075 ± 0.0053 | 0.5007 |

- Cùng TF-IDF và LinearSVC, năm hạt giống chia tập (0 đến 4), trên bình luận của tập dữ liệu
- Mọi chênh lệch nằm trong một độ lệch chuẩn: với bộ phân loại, thay đổi là trung tính
- Giữ bản mới vì tên riêng thành một token trong từ khóa chủ đề và từ điển không áp sắc thái lên dữ liệu

<!-- Ghi chú: Bài 5 của môn có cặp thí nghiệm có tách từ và không tách từ để xem tiền xử lý đổi kết quả thế nào. Nhóm em làm cặp thí nghiệm tương tự cho hai phiên bản tiền xử lý của mình. Kết quả là bảng này. Mời thầy nhìn hai dòng đầu: 77.58% và 77.58%, giống nhau đến hai chữ số. Nói thẳng: với bộ phân loại cảm xúc, đổi tiền xử lý không tăng điểm. Nhóm em vẫn giữ bản mới, nhưng vì lý do khác: từ khóa chủ đề có tên riêng đúng, và từ điển không còn tự gán cảm xúc. Điểm tăng thật sự ở đồ án đến từ chỗ khác, em sẽ nói ở phần mô hình. -->

---

## Biểu diễn: TF-IDF cho cảm xúc, vector câu 768 chiều cho chủ đề

- Cảm xúc: TF-IDF unigram và bigram, 15,000 đặc trưng, `sublinear_tf`
- Bigram là bắt buộc vì phủ định đứng trước từ bị phủ định: `không hay`, `không thích` là đặc trưng của lớp tiêu cực; `hay mà`, `hay nhưng` là đặc trưng của lớp trung tính
- Chủ đề: mỗi bình luận đã tách từ thành một vector 768 chiều từ `keepitreal/vietnamese-sbert` (Sentence-BERT tiếng Việt; theo `config.json` trên Hugging Face, mô hình được tinh chỉnh từ PhoBERT-base, nên cần tách từ trước)
- Từ khóa của mỗi cụm chọn bằng c-TF-IDF: từ nào nhiều trong cụm này, ít ở cụm khác thì lên đầu (ví dụ số ở phụ lục)

<!-- Ghi chú: Hai bài toán con cần hai cách biểu diễn. Với cảm xúc, nhóm em dùng TF-IDF có bigram. Nếu chỉ dùng unigram, chữ "hay" sẽ kéo cả "không hay" lẫn "hay nhưng" về lớp tích cực. Bigram giữ được cặp phủ định. Với chủ đề, mỗi bình luận thành một vector 768 chiều từ một mô hình Sentence-BERT tiếng Việt. Thẻ mô hình không ghi mô hình gốc, nhóm em đọc file config trên Hugging Face và suy ra nó tinh chỉnh từ PhoBERT-base. Điều đó có hệ quả thực tế: PhoBERT học trên văn bản đã tách từ, nên bước pyvi phía trước vừa phục vụ từ khóa, vừa là định dạng đầu vào mà mô hình nhúng mong đợi. -->

---

## Quy trình đánh giá: chọn trên tập huấn luyện, chấm một lần trên tập kiểm tra

![h:520](../docs/diagrams/evaluation.png)

<!-- Ghi chú: Trước khi xem con số, em nói cách chấm. Dữ liệu chia phân tầng 80 trên 20 với hạt giống 42. Mọi việc chọn lựa, so bốn mô hình nền và dò tham số C, chỉ chạy bằng cross-validation năm phần trên 15,997 dòng huấn luyện. Tập kiểm tra 4,000 dòng để dành, chấm đúng một lần với mô hình cuối. Sau đó nhóm em chia lại với năm hạt giống khác để xem con số có ổn định không. Quy trình này là lý do nhóm em tin các số ở hai slide sau. -->

---

## Mô hình cảm xúc: bốn mô hình nền, dò C, chỉ trên tập huấn luyện

<style scoped>table { font-size: 0.78em; } ul { font-size: 0.85em; }</style>

| Mô hình | Tỉ lệ dự đoán đúng (accuracy), CV | Macro-F1, CV |
|---|---|---|
| MostFrequent (luôn đoán lớp đa số) | 47.95% | 0.2161 |
| MultinomialNB | 73.82% | 0.5620 |
| LogisticRegression | 76.25% | 0.7170 ± 0.0054 |
| LinearSVC, C = 1 | 76.56% | 0.7054 ± 0.0095 |
| LinearSVC, C = 0.3 (đã dò) | 77.49% | 0.7180 ± 0.0066 |

- Cross-validation 5-fold phân tầng trên 15,997 dòng huấn luyện, `class_weight="balanced"`
- Lưới C theo macro-F1: 0.1 cho 0.7124, 0.3 cho 0.7180, 1.0 cho 0.7054, 3.0 cho 0.6875
- LinearSVC hơn LogisticRegression 0.0010, nhỏ hơn độ lệch chuẩn giữa các fold: hai mô hình ngang nhau

<!-- Ghi chú: Nhóm em so bốn mô hình nền bằng cross-validation, chỉ trên tập huấn luyện, tập kiểm tra để dành. Dòng đầu là mốc sàn: đoán toàn lớp đa số được 47.95% nhưng macro-F1 chỉ 0.22, nên nhóm em không dùng accuracy làm chỉ tiêu chính. Hai mô hình tuyến tính vượt Naive Bayes rõ. Rồi nhóm em dò C cho LinearSVC trên bốn giá trị: từ C bằng 1 xuống 0.3, macro-F1 tăng từ 0.7054 lên 0.7180. Đây là phần tăng đo được của đồ án, không phải tiền xử lý. Còn LinearSVC so với hồi quy logistic: chênh một phần nghìn, nhỏ hơn nhiễu giữa các fold. Nhóm em giữ LinearSVC theo thiết kế ban đầu, không phải vì nó thắng. -->

---

## Tập kiểm tra: macro-F1 0.7161, lớp trung tính là điểm yếu

![bg right:42% fit](../results/confusion_matrix_normalized.png)

<style scoped>table { font-size: 0.8em; } ul { font-size: 0.9em; }</style>

| Lớp | Độ chính xác | Độ phủ | F1 |
|---|---|---|---|
| Tiêu cực (1,400) | 0.7458 | 0.7564 | 0.7511 |
| Trung tính (682) | 0.5270 | 0.5440 | 0.5354 |
| Tích cực (1,918) | 0.8715 | 0.8525 | 0.8619 |

- Tỉ lệ dự đoán đúng (accuracy) 76.62%, macro-F1 0.7161 trên 4,000 dòng, chấm một lần; độ chính xác là precision, độ phủ là recall
- Năm hạt giống 0 đến 4: 77.58% ± 0.44, macro-F1 0.7200 ± 0.0037
- Trung tính: đúng 371/682; 203 bị gán tiêu cực, 108 bị gán tích cực

<!-- Ghi chú: Đây là kết quả trên tập kiểm tra, chấm đúng một lần với mô hình đã chọn. Tỉ lệ dự đoán đúng 76.62%, macro-F1 0.7161. Chạy lại với năm hạt giống chia tập khác thì được 77.58%, nên con số 76.62% là ước lượng thận trọng. Mời thầy nhìn hàng giữa của ma trận nhầm lẫn bên phải: lớp trung tính chỉ nhận ra 54%, còn lại chia đều về hai phía, 203 sang tiêu cực, 108 sang tích cực. Hai lớp tiêu cực và tích cực hiếm khi nhầm sang nhau, chỉ 133 và 158 trên gần 3,300 mẫu. Nói gọn: mô hình phân biệt khen với chê tốt, nhưng không chắc đâu là "không khen không chê". -->

---

## Gom cụm chủ đề: tham số HDBSCAN quyết định hơn cả từ dừng

<style scoped>table { font-size: 0.78em; } ul { font-size: 0.9em; }</style>

| Chọn cụm | Cụm tối thiểu | min_samples | Chủ đề | Nhiễu (-1) | Cụm lớn nhất |
|---|---|---|---|---|---|
| eom (mặc định BERTopic) | 10 | mặc định | 23 | 735 (49.2%) | 81 (5.4%) |
| eom | 20 | 5 | 3 | 0 (0%) | 1,359 (91.0%) |
| eom (chọn dùng) | 15 | 1 | 26 | 497 (33.3%) | 117 (7.8%) |
| leaf | 10 | mặc định | 27 | 793 (53.1%) | 80 (5.4%) |
| leaf | 20 | 5 | 19 | 620 (41.5%) | 94 (6.3%) |

- Khảo sát trên 1,493 bình luận của tập dữ liệu (1,500 dòng đầu), không phải bình luận của một video thật
- Mặc định đẩy gần nửa vào nhiễu; `min_samples` 5 gom 91% vào một cụm; 15/1 cân bằng nhất và là mặc định của ứng dụng
- Bật hay tắt từ dừng cho phép gán chủ đề giống hệt từng dòng, chỉ từ khóa đổi

<!-- Ghi chú: Sang nhánh chủ đề. BERTopic gồm UMAP giảm 768 chiều xuống 5, HDBSCAN gom cụm theo mật độ, rồi c-TF-IDF chọn từ khóa. Tham số mặc định không dùng được cho bình luận cùng một chương trình: 49% bình luận bị xếp vào nhiễu. Đổi min_samples lên 5 thì ngược lại, 91% dồn vào một cụm, không có nhiễu, nhưng vô nghĩa. Cấu hình 15 và 1 nằm giữa: 26 chủ đề, một phần ba là nhiễu, và các bình luận nhiễu vẫn được giữ trong bảng dưới tên nhóm -1, không bị loại. Một phát hiện nhóm em thấy đáng nói: tắt từ dừng thì số chủ đề và tỉ lệ nhiễu không đổi một dòng nào, vì từ dừng chỉ đi vào bước chọn từ khóa, sau khi HDBSCAN đã gom xong. Xin nhấn mạnh là bảng này chạy trên bình luận trong tập dữ liệu; nó có chuyển sang một video thật hay không, nhóm em chưa đo. -->

---

## Ứng dụng: năm tab, ba nguồn dữ liệu, không cần API key để chấm

![bg right:48%](../docs/screenshots/03_tab_tong_quan.png)

<style scoped>ul { font-size: 0.9em; }</style>

- Ba nguồn: Link YouTube (cần API key), Tệp CSV, Dữ liệu mẫu
- Năm tab: Tổng quan, Chủ đề, Cảm xúc, Dữ liệu, Mô hình
- Bình luận và vector nhúng được cache; đổi tab hay lọc bảng không chạy lại
- Ảnh: 1,000 dòng mẫu, 23 chủ đề, 20.6% nhiễu, 47.4% tích cực (tập huấn luyện, nên tỉ lệ cảm xúc lạc quan hơn thực tế)

<!-- Ghi chú: Toàn bộ pipeline đóng thành một ứng dụng Streamlit. Bây giờ nhóm em chạy trực tiếp khoảng 3 phút. (mở ứng dụng, ở thanh bên chọn Tệp CSV với bình luận đã cào trước của [video demo]; nếu mạng hoặc tệp có vấn đề thì chọn Dữ liệu mẫu, 1,000 dòng; bấm Bắt đầu phân tích) Trong lúc chạy em nói qua sáu bước của ứng dụng: lấy dữ liệu, làm sạch, nhúng câu, gom cụm, phân loại, tóm tắt tùy chọn. (khi xong, mở tab Tổng quan) Đây là số chủ đề, tỉ lệ nhiễu và tỉ lệ tích cực. (mở tab Chủ đề, chỉ vào một cụm) Mỗi chủ đề có từ khóa và bình luận tiêu biểu; bản đồ khoảng cách cho thấy cụm nào gần nhau. (mở tab Cảm xúc) Cảm xúc theo từng chủ đề: chủ đề nào bị chê nhiều nhất. (mở tab Dữ liệu, lọc một từ khóa) Bảng có lọc theo chủ đề, cảm xúc, từ khóa và tải CSV. Tab Mô hình chỉ đọc lại thẻ mô hình và các bảng em vừa trình bày, em không mở để tiết kiệm thời gian. -->

---

## Phân tích lỗi: mô hình học cả chủ đề và tên riêng làm tín hiệu cảm xúc

<style scoped>ul { font-size: 0.86em; }</style>

- Lớp tiêu cực dựa vào `không` (+2.93), `không hay` (+2.29), `không thích` (+2.10), và cả `quảng_cáo`, `khán_giả`: hai từ sau là chủ đề, không phải cảm xúc
- Lớp trung tính dựa vào `nhưng` (+2.95), `tiếc`, `hay mà`, `hay nhưng`: đúng định nghĩa lớp, nhưng cho thấy ranh giới mờ ngay trong nhãn
- Lớp tích cực dựa vào `đỉnh` (+3.36), `mê` (+3.20), và tên riêng `atus`, `negav`: đúng trong miền, không chuyển được sang video khác
- Việc nhóm sẽ hoàn thành: đọc tay 150 bình luận sai trên tập kiểm tra, xếp theo hiện tượng ngôn ngữ (phủ định, teencode ngoài từ điển, không dấu, châm biếm, vừa khen vừa chê, chỉ còn emoji hoặc tên riêng, nhãn gốc đáng ngờ)
- Hạn chế: emoji bị xóa, bình luận không dấu tự tạo cụm riêng, một miền dữ liệu, tóm tắt LLM chưa đánh giá, nhãn silver do LLM gán, chưa kiểm tay

<!-- Ghi chú: Nhóm em đọc trọng số của LinearSVC để hiểu mô hình dựa vào gì. Phần tốt: chữ "không" và các bigram phủ định là đặc trưng mạnh nhất của lớp tiêu cực, đúng như thiết kế bigram. Phần đáng lo: "quảng cáo" và "khán giả" cũng là đặc trưng tiêu cực, tức là mô hình học được rằng ai nhắc quảng cáo thì thường chê. Và tên hai thí sinh là đặc trưng tích cực. Tín hiệu đó đúng trong chương trình này, nhưng đem sang video khác sẽ sai. Phần phân tích định tính nhóm em chưa xong: sẽ đọc tay 150 bình luận sai và đếm theo từng hiện tượng, không ước lượng. Hạn chế lớn nhất theo nhóm em là emoji: có trong 34.5% bình luận, mang cảm xúc, mà bước làm sạch xóa mất. -->

---

## Kết luận: pipeline chạy được, tái lập được, và biết mình yếu ở đâu

- Đủ năm bước của môn, mọi con số tái lập từ `results/` bằng lệnh trong phụ lục báo cáo
- Macro-F1 0.72 trên tập kiểm tra và năm hạt giống; lớp trung tính (F1 0.5354) là giới hạn của cách tiếp cận TF-IDF tuyến tính
- Với chủ đề, tham số HDBSCAN quyết định kết quả nhiều hơn danh sách từ dừng
- Tiếp theo: giữ emoji làm token và đo lại theo cùng quy trình năm hạt giống; dùng vector câu cho hồi quy logistic hoặc tinh chỉnh PhoBERT ba lớp; tính coherence `c_v` để chọn cấu hình chủ đề theo số đo thay vì cảm nhận

<!-- Ghi chú: Ba điều nhóm em rút ra. Một, pipeline đi đủ năm bước và ai cũng chạy lại được: một lệnh huấn luyện 12 giây sinh lại toàn bộ kết quả. Hai, macro-F1 dừng ở 0.72, và cái kéo nó xuống là lớp trung tính; TF-IDF tuyến tính không đủ để hiểu "hay mà tiếc" là không khen không chê. Ba, ở phần chủ đề, thứ đáng dò là tham số gom cụm, không phải danh sách từ dừng. Ba việc tiếp theo xếp theo chi phí tăng dần: giữ emoji, thử vector câu cho phân loại, và đo coherence. Câu nhóm em muốn để lại: số đẹp thì dễ, số đọc lại được mới khó, và nhóm em chọn cái khó. Nhóm em xin cảm ơn thầy và các bạn. -->

---

## Câu hỏi

Nhóm em xin cảm ơn thầy và các bạn.

Mã nguồn, 146 kiểm thử và toàn bộ tệp kết quả nằm trong repo nộp kèm báo cáo. Ba slide phụ lục phía sau: tài liệu tham khảo, ví dụ số c-TF-IDF, và một bẫy kỹ thuật của BERTopic.

<!-- Ghi chú: Nhóm em xin dừng ở đây và sẵn sàng trả lời câu hỏi. (nếu được hỏi về c-TF-IDF hoặc về BERTopic với tiếng Việt, chuyển sang slide phụ lục tương ứng) -->

---

## Phụ lục B1. Tài liệu tham khảo

1. Slide môn CS221 Xử lý ngôn ngữ tự nhiên, NCS.ThS Đặng Văn Thìn, UIT, bài 3 (tiền xử lý), bài 4 (biểu diễn văn bản), bài 5 (phân tích cảm xúc), bài 6 (độ đo đánh giá)
2. Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794. Kiểm tra lại trước khi nộp.
3. McInnes, L., Healy, J., Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426. Kiểm tra lại trước khi nộp.
4. Campello, R. J. G. B., Moulavi, D., Sander, J. (2013). Density-Based Clustering Based on Hierarchical Density Estimates. PAKDD 2013. Kiểm tra lại trước khi nộp.
5. Nguyen, D. Q., Nguyen, A. T. (2020). PhoBERT: Pre-trained language models for Vietnamese. Findings of EMNLP 2020. Kiểm tra lại trước khi nộp.
6. Thẻ mô hình `keepitreal/vietnamese-sbert`, Hugging Face Hub. Kiểm tra lại trước khi nộp.
7. pyvi: Python Vietnamese toolkit. Pedregosa, F. và cộng sự (2011). Scikit-learn: Machine Learning in Python. JMLR 12. Tài liệu Streamlit. Tài liệu YouTube Data API v3. Kiểm tra lại phiên bản trước khi nộp.
8. Dữ liệu được cung cấp bởi dự án ATSH-ABSA (Phạm Xuân Vĩnh Hà, UIT), chỉ dùng cho mục đích học tập. Gói ATSH-NLP-20k; README của tác giả tại `data/README_ATSH_NLP_20k_goc.md`.

<!-- Ghi chú: Danh sách tài liệu tham khảo, trùng với mục 10.2 của báo cáo. Các mục ghi "kiểm tra lại trước khi nộp" nhóm em sẽ đối chiếu bản gốc về năm và nơi công bố trước ngày nộp. -->

---

## Phụ lục B2. c-TF-IDF: vì sao `hay` không bao giờ đứng đầu từ khóa

W(t, c) = tf(t, c) × log(1 + A / f(t)), với A là số từ trung bình của một cụm

| Từ | Cụm 1 | Cụm 2 | Cụm 3 | f(t) | log(1 + 10/f(t)) |
|---|---|---|---|---|---|
| hát | 6 | 0 | 0 | 6 | log(2.667) = 0.981 |
| hạng | 0 | 6 | 0 | 6 | 0.981 |
| quảng_cáo | 1 | 1 | 6 | 8 | log(2.25) = 0.811 |
| hay | 3 | 3 | 4 | 10 | log(2) = 0.693 |

- Ba cụm, mỗi cụm 10 từ, nên A = 10; logarit tự nhiên
- Cụm 1: `hát` 6 × 0.981 = 5.88; `hay` 3 × 0.693 = 2.08; `quảng_cáo` 0.81. Cụm 3: `quảng_cáo` 4.87; `hay` 2.77
- `hay` có mặt ở cả ba cụm nên thành phần logarit thấp nhất, luôn đứng sau từ đặc trưng riêng. Hư từ như `là`, `của`, `mà` có tần suất thô quá cao nên vẫn cần danh sách từ dừng

<!-- Ghi chú: Ví dụ số cho c-TF-IDF. BERTopic nối mọi bình luận của một cụm thành một văn bản rồi chấm từng từ: tần suất trong cụm nhân với logarit của một cộng A chia f. Từ "hát" chỉ có ở cụm 1 nên điểm 5.88, đứng đầu. Từ "hay" rải khắp ba cụm, thành phần logarit chỉ 0.693, nên dù xuất hiện nhiều vẫn đứng sau. Nhưng hư từ như "là", "của" thì tần suất thô lớn đến mức vẫn chiếm đầu bảng, nên danh sách từ dừng ở bước này vẫn cần. -->

---

## Phụ lục B3. Bẫy BERTopic với tiếng Việt và cách tái lập kết quả

- `BERTopic()` mặc định `language="english"`; nếu không truyền `embedding_model`, bước làm sạch nội bộ xóa mọi ký tự ngoài `[A-Za-z0-9 ]` trước c-TF-IDF: `không` thành `khng`, `chương_trình` thành `chngtrnh`
- Pipeline của nhóm luôn truyền mô hình nhúng vào `BERTopic`, kể cả khi vector đã tính sẵn; một kiểm thử xác nhận `language` của mô hình là `None`
- Tái lập: `python src/train_sentiment.py` (khoảng 12 giây, sinh lại `results/` và `models/`); `python experiments/dataset_stats.py`, `ab_preprocess.py`, `topic_ablation.py` cho ba bảng thực nghiệm; `pytest` chạy 146 kiểm thử trong khoảng 16 giây
- Lệnh gom cụm dòng lệnh trên 1,500 dòng đầu cho 1,493 bình luận hợp lệ, 26 chủ đề, 33.3% nhiễu; UMAP có hạt giống cố định nên cùng máy cho cùng kết quả

<!-- Ghi chú: Một bài học kỹ thuật nhóm em xác minh trong mã nguồn thư viện. BERTopic khởi tạo với ngôn ngữ mặc định là tiếng Anh, và nếu mình không truyền mô hình nhúng, nó âm thầm xóa hết dấu tiếng Việt trước khi tính từ khóa. Chữ "không" thành "khng". Nhóm em vá bằng cách luôn truyền mô hình nhúng và viết một kiểm thử canh chỗ đó. Phần tái lập: một lệnh huấn luyện, ba script thực nghiệm, và pytest. Không lệnh nào cần API key hay mạng, trừ lần đầu tải mô hình nhúng về cache. -->
