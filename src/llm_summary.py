"""Tóm tắt chủ đề bằng mô hình ngôn ngữ chạy nội bộ qua Ollama.

c-TF-IDF chỉ cho ra danh sách từ khóa rời rạc. Lớp `OllamaSummarizer` gửi từ
khóa kèm vài bình luận tiêu biểu (văn bản GỐC, không phải chuỗi đã tách từ)
sang Ollama và nhận lại đúng một câu tiếng Việt mô tả chủ đề.

Ollama phơi một API tương thích OpenAI nên ở đây dùng thư viện `openai` với
`base_url` trỏ về máy cá nhân. Không có khóa thật nào được gửi đi: Ollama chỉ
đòi một chuỗi bất kỳ ở vị trí `api_key`.
"""

from __future__ import annotations

DEFAULT_MODEL = "qwen2"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_TIMEOUT = 60.0

PROMPT_TEMPLATE = (
    "Bạn là chuyên gia phân tích bình luận mạng xã hội tiếng Việt.\n"
    "Dưới đây là các từ khóa và một số bình luận thuộc cùng một chủ đề của một video YouTube.\n\n"
    "Từ khóa: {keywords}\n\n"
    "Các bình luận:\n{comments}\n\n"
    "Hãy viết ĐÚNG MỘT CÂU tiếng Việt (tối đa 40 từ) nêu người xem đang bàn về điều gì. "
    "Chỉ trả về câu đó, không thêm tiêu đề, không gạch đầu dòng, không giải thích."
)


class OllamaSummarizer:
    """Bọc một lần gọi Ollama; có thể tiêm client giả khi kiểm thử."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        client=None,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._client = client

    @property
    def client(self):
        """Tạo client muộn để việc import module không cần tới mạng."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key="ollama", timeout=self.timeout)
        return self._client

    def available(self) -> tuple[bool, str]:
        """Thử gọi `models.list()` một lần để biết Ollama có đang chạy không."""
        try:
            self.client.models.list()
        except Exception as error:
            return (
                False,
                f"Ollama chưa chạy hoặc không truy cập được tại {self.base_url} ({error}).",
            )
        return True, f"Đã kết nối Ollama tại {self.base_url}, dùng mô hình '{self.model}'."

    def build_prompt(self, keywords: list[str], comments: list[str], n_comments: int = 10) -> str:
        """Ghép câu nhắc tiếng Việt từ từ khóa và tối đa `n_comments` bình luận."""
        keyword_text = ", ".join(str(word) for word in keywords if str(word).strip()) or "(không có)"
        selected = [str(item).strip() for item in comments if str(item).strip()][:n_comments]
        comment_text = "\n".join(f"- {item}" for item in selected) or "- (không có bình luận)"
        return PROMPT_TEMPLATE.format(keywords=keyword_text, comments=comment_text)

    def summarize_topic(self, keywords: list[str], comments: list[str], n_comments: int = 10) -> str:
        """Trả về một câu tóm tắt; lỗi gọi API được đẩy lên cho phía gọi xử lý."""
        prompt = self.build_prompt(keywords, comments, n_comments=n_comments)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.3,
        )
        content = response.choices[0].message.content
        return (content or "").strip()
