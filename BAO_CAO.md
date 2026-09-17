# BÁO CÁO ĐỒ ÁN: PHÂN TÍCH VÀ GOM CỤM CHỦ ĐỀ BÌNH LUẬN YOUTUBE (TOPIC MODELING)

## 1. Giới thiệu Bài toán
Trong kỷ nguyên số, nền tảng YouTube chứa đựng một lượng khổng lồ các phản hồi (bình luận) từ người xem. Việc đọc thủ công hàng ngàn bình luận để hiểu phản ứng của khán giả là điều gần như bất khả thi.
Đồ án này ra đời nhằm mục đích ứng dụng các kỹ thuật Xử lý Ngôn ngữ Tự nhiên (NLP) tiên tiến nhất để tự động phân tích hàng ngàn bình luận, sau đó gom nhóm chúng thành các "Chủ đề" (Topic) cốt lõi, đồng thời phân loại "Cảm xúc" (Sentiment) của khán giả. Từ đó, phần mềm giúp nhà phân tích nhanh chóng nắm bắt được "cộng đồng mạng đang thực sự bàn tán về điều gì" và "họ đang cảm thấy thế nào" đằng sau một video.

---

## 2. Quy trình Thực hiện (Pipeline)
Hệ thống được thiết kế theo một luồng xử lý (pipeline) tự động hóa chặt chẽ bao gồm 4 giai đoạn chính:

### 2.1. Thu thập dữ liệu (Data Collection)
Sử dụng **YouTube Data API v3** để tự động hóa việc kết nối và lấy dữ liệu. Thuật toán cào dữ liệu được lập trình để xử lý phân trang (pagination), cho phép lấy được số lượng lớn bình luận (vượt qua giới hạn 100 bình luận/lần của YouTube API). Dữ liệu thô thu về bao gồm: tên người dùng, thời gian, số lượt thích và nội dung bình luận.

### 2.2. Tiền xử lý dữ liệu Tiếng Việt (Preprocessing)
Ngôn ngữ mạng xã hội thường chứa nhiều "rác". Dữ liệu thô phải trải qua các bước làm sạch nghiêm ngặt:
- **Làm sạch văn bản (Cleaning):** Sử dụng Biểu thức chính quy (Regex) để xóa bỏ các đường link URL, biểu tượng cảm xúc (Emoji), các ký tự đặc biệt, dấu câu và đưa toàn bộ về chữ in thường.
- **Chuẩn hóa Teencode (Từ lóng):** Trước khi tách từ, văn bản đi qua một bộ từ điển teencode để phiên dịch các từ lóng phổ biến trên mạng (ví dụ: `ko`, `k` -> `không`, `đc` -> `được`, `sp` -> `sản phẩm`). Bước này giúp thuật toán ở các bước sau nhận diện chính xác ngữ nghĩa của câu.
- **Tách từ (Word Tokenization):** Khác với tiếng Anh, tiếng Việt có đặc thù từ ghép (ví dụ: "đông hùng" là 1 từ chỉ tên người, không phải 2 từ độc lập "đông" và "hùng"). Hệ thống sử dụng thư viện **`pyvi`** (Python Vietnamese Toolkit) để tự động nhận diện và nối các từ ghép bằng dấu gạch dưới (VD: `đông_hùng`, `mỹ_mãn`, `xuân_thì`). Bước này là bắt buộc để các mô hình học máy đằng sau hiểu đúng ngữ nghĩa của tiếng Việt.

### 2.3. Khám phá và Gom cụm Chủ đề (Thuật toán BERTopic)
Đây là "trái tim" của hệ thống. Để máy tính có thể tự động hiểu và gom cụm các bình luận thành từng nhóm từ khóa riêng biệt, chúng ta sử dụng thư viện **BERTopic**. Quá trình này diễn ra qua 4 bước toán học cực kỳ phức tạp bên dưới mảng ngầm:

#### Bước 1: Nhúng ngữ nghĩa (Embedding) bằng Sentence-BERT (vietnamese-sbert)
Máy tính không hiểu chữ viết, nó chỉ hiểu các con số. Do đó, hệ thống truyền các bình luận qua mạng nơ-ron **Sentence-BERT** (sử dụng pre-trained model `keepitreal/vietnamese-sbert`). Về bản chất, đây là kiến trúc mạng nơ-ron sinh đôi (Siamese Neural Network) được tinh chỉnh (fine-tune) trên nền tảng mô hình ngôn ngữ **PhoBERT-base**. Mô hình sẽ "đọc hiểu" toàn bộ ngữ cảnh và mã hóa mỗi câu bình luận thành một Vector nhúng (Sentence Embedding) đại diện cho ý nghĩa của câu trong không gian 768 chiều. 
> *Nguyên lý:* Các bình luận có ngữ nghĩa giống nhau (dù dùng từ vựng khác nhau, ví dụ: "tuyệt vời" và "hay quá") sẽ có các điểm vector nằm rất sát nhau trong không gian 768 chiều này.

#### Bước 2: Giảm chiều không gian (Dimensionality Reduction) bằng UMAP
Không gian 768 chiều là một không gian quá lớn (Curse of Dimensionality), khiến thuật toán rất khó để khoanh vùng và chạy rất chậm. Thuật toán **UMAP** được áp dụng để "ép" không gian này xuống chỉ còn 5 chiều. 
> *Ưu điểm:* UMAP có khả năng giữ lại cấu trúc lân cận cục bộ (local structure), nghĩa là các bình luận vốn có ý nghĩa gần nhau ở 768 chiều thì khi bị ép xuống 5 chiều, chúng vẫn đứng sát cạnh nhau.

#### Bước 3: Phân cụm (Clustering) bằng HDBSCAN
Sau khi đã rải các bình luận lên không gian 5 chiều, thuật toán **HDBSCAN** bắt đầu quét qua toàn bộ dữ liệu. Nó đi tìm những khu vực có **mật độ điểm dữ liệu tụ tập dày đặc** và gom chúng lại thành 1 Cụm (Cluster).
> *Sự thông minh của HDBSCAN:* Những bình luận nói về cùng một bài hát hoặc một ca sĩ cụ thể sẽ tụ tập lại thành một đám đông. Ngược lại, những bình luận spam, vô nghĩa sẽ bị đẩy ra xa và đứng bơ vơ. HDBSCAN tự động nhận diện những kẻ đứng bơ vơ này, gán cho chúng nhãn `-1` (Nhiễu / Outlier) và loại bỏ hoàn toàn khỏi phân tích.

#### Bước 4: Bóc tách Từ khóa (Topic Representation) bằng c-TF-IDF
Sau khi đã gom được thành các Cụm (Nhóm), làm sao máy tính biết Cụm đó thực chất đang nói về cái gì? BERTopic sẽ gộp tất cả bình luận trong 1 Cụm lại thành 1 văn bản khổng lồ. Sau đó nó dùng công thức **c-TF-IDF** (Class-based Term Frequency - Inverse Document Frequency).
> *Giải thích tại sao nó lại nhặt ra các từ khóa riêng biệt:* Công thức c-TF-IDF sẽ đi dò tìm những từ vựng **xuất hiện cực kỳ nhiều ở Cụm này**, nhưng lại **vắng bóng ở tất cả các Cụm khác**. 
> Ví dụ: Cụm từ `đông_hùng` và `hát` xuất hiện liên tục ở Cụm số 1, nhưng lại gần như không xuất hiện ở Cụm 2, Cụm 3. Do đó, c-TF-IDF sẽ chấm điểm số rất cao cho 2 từ này và lập tức trích xuất chúng làm Nhãn (Keywords) đại diện cho Cụm số 1. Đó chính là lý do bạn thấy biểu đồ cột (Barchart) tách ra được các từ khóa rất chuẩn xác.

### 2.4. Giải nghĩa bằng AI Sinh tạo (Local LLM - Ollama)
Điểm yếu của c-TF-IDF là nó chỉ cung cấp các từ khóa rời rạc, đòi hỏi người xem biểu đồ phải tự tư duy để ghép thành câu. Để nâng tầm đồ án, hệ thống đã tích hợp thêm **Ollama** (Một trí tuệ nhân tạo chạy hoàn toàn offline ngay trên máy tính của bạn) với mô hình `qwen2`.
Hệ thống sẽ nhặt ra 10 bình luận tiêu biểu nhất của mỗi Cụm, gửi cho Ollama với tư cách là một "Chuyên gia tóm tắt". Ollama sẽ đọc hiểu 10 bình luận này và viết ra một câu văn tiếng Việt có chủ ngữ, vị ngữ hoàn chỉnh (VD: *"Nhiều khán giả bày tỏ sự xúc động và khen ngợi phần biểu diễn của anh tài Đông Hùng"*). Việc này biến một biểu đồ khô khan thành một báo cáo có tính con người và cực kỳ dễ tiếp cận.

### 2.5. Phân tích Cảm xúc Đa lớp bằng Machine Learning (Multiclass Sentiment Analysis)
Bên cạnh việc dùng mô hình học sâu có sẵn (PhoBERT), đồ án này còn trình diễn một chu trình huấn luyện Học máy (Machine Learning Pipeline) truyền thống từ con số không, nhằm phân loại các bình luận thành 3 sắc thái (Multiclass): Tích cực (Positive), Trung tính (Neutral) hoặc Tiêu cực (Negative).
Quy trình tự huấn luyện (Tự Train) diễn ra theo các bước sau:
1. **Thu thập tập dữ liệu:** Hệ thống sử dụng tập dữ liệu chuẩn mực gồm 20.000 bình luận mạng xã hội tiếng Việt đã được gán nhãn thủ công (2: Tích cực, 1: Trung tính, 0: Tiêu cực).
2. **Trích xuất đặc trưng (N-grams TF-IDF):** Văn bản sau khi tách từ sẽ đi qua thuật toán `TfidfVectorizer` với cấu hình N-grams (Bigram). Không chỉ học từ đơn, thuật toán học cả cụm 2 từ (như "không_tốt", "rất_tuyệt") để bắt trọn ngữ nghĩa phủ định/khẳng định. TF-IDF sẽ biến đổi chúng thành một véc-tơ không gian thưa lên tới 15.000 chiều.
3. **Huấn luyện bằng LinearSVC:** Khác với Hồi quy Logistic thông thường, mô hình Máy Véc-tơ hỗ trợ (Support Vector Classification - LinearSVC) được sử dụng để tối đa hóa ranh giới quyết định (decision boundaries) giữa 3 miền cảm xúc. Đặc biệt, tham số `class_weight='balanced'` được kích hoạt để khắc phục tình trạng mất cân bằng dữ liệu, ép thuật toán phải chú ý phân tích sâu vào các bình luận mang nhãn Trung tính.
4. **Lưu trữ và Ứng dụng:** Sau khi Train và đạt độ chính xác ổn định trên tập Test, mô hình được lưu lại thành file `.pkl`. Khi người dùng nhập một video YouTube mới, Web Streamlit sẽ sử dụng file `.pkl` này để dự đoán cảm xúc của từng bình luận và vẽ biểu đồ Tròn (Pie Chart) với 3 dải màu trực quan hóa thái độ chung của khán giả.

---

## 3. Trực quan hóa và Triển khai (Deployment)
Thay vì bắt người dùng chạy code trên môi trường Console/Terminal khó hiểu, toàn bộ quy trình trên đã được đóng gói bằng thư viện **Streamlit** thành một Trang Web (Web Application) giao diện Dark Mode chuyên nghiệp. 
Người dùng chỉ việc dán link YouTube và click chuột. Các kết quả dữ liệu sẽ được trực quan hóa (Data Visualization) thành dạng biểu đồ tương tác thông qua thư viện `Plotly`, giúp việc báo cáo trở nên trực quan và thuyết phục nhất.
