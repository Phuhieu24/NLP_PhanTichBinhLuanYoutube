"""Kiểm thử OllamaSummarizer bằng client giả, không gọi mạng."""

import pytest

from llm_summary import DEFAULT_BASE_URL, OllamaSummarizer


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, content="  Khán giả khen phần trình diễn của Đông Hùng.  "):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self.content)


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeModels:
    def __init__(self, error=None):
        self.error = error
        self.call_count = 0

    def list(self):
        self.call_count += 1
        if self.error is not None:
            raise self.error
        return ["qwen2"]


class FakeClient:
    def __init__(self, content="  Khán giả khen phần trình diễn của Đông Hùng.  ", error=None):
        self.completions = FakeCompletions(content)
        self.chat = FakeChat(self.completions)
        self.models = FakeModels(error)


KEYWORDS = ["đông_hùng", "hát", "xúc_động"]
COMMENTS = [
    "Đông Hùng hát bài này quá xúc động!",
    "Nghe mà rơi nước mắt luôn",
    "Giọng anh ấm thật sự",
]


def test_available_true_probes_models_once():
    client = FakeClient()
    summarizer = OllamaSummarizer(client=client)
    ready, message = summarizer.available()
    assert ready is True
    assert client.models.call_count == 1
    assert DEFAULT_BASE_URL in message


def test_available_false_on_exception():
    client = FakeClient(error=ConnectionError("connection refused"))
    summarizer = OllamaSummarizer(base_url="http://localhost:9999/v1", client=client)
    ready, message = summarizer.available()
    assert ready is False
    assert "Ollama chưa chạy hoặc không truy cập được tại http://localhost:9999/v1" in message
    assert "connection refused" in message


def test_summarize_topic_builds_vietnamese_prompt():
    client = FakeClient()
    summarizer = OllamaSummarizer(model="qwen2.5", client=client)
    summary = summarizer.summarize_topic(KEYWORDS, COMMENTS)

    assert summary == "Khán giả khen phần trình diễn của Đông Hùng."
    assert len(client.completions.calls) == 1
    call = client.completions.calls[0]
    assert call["model"] == "qwen2.5"
    assert call["max_tokens"] == 150
    assert call["temperature"] == pytest.approx(0.3)

    prompt = call["messages"][0]["content"]
    assert call["messages"][0]["role"] == "user"
    for comment in COMMENTS:
        assert comment in prompt
    for keyword in KEYWORDS:
        assert keyword in prompt
    assert "ĐÚNG MỘT CÂU" in prompt
    assert "tiếng Việt" in prompt


def test_summarize_topic_limits_number_of_comments():
    client = FakeClient()
    summarizer = OllamaSummarizer(client=client)
    summarizer.summarize_topic(KEYWORDS, COMMENTS, n_comments=2)

    prompt = client.completions.calls[0]["messages"][0]["content"]
    assert COMMENTS[0] in prompt
    assert COMMENTS[1] in prompt
    assert COMMENTS[2] not in prompt


def test_build_prompt_handles_empty_input():
    summarizer = OllamaSummarizer(client=FakeClient())
    prompt = summarizer.build_prompt([], ["", "   "])
    assert "(không có)" in prompt
    assert "(không có bình luận)" in prompt


def test_summarize_topic_returns_empty_string_when_content_is_none():
    summarizer = OllamaSummarizer(client=FakeClient(content=None))
    assert summarizer.summarize_topic(KEYWORDS, COMMENTS) == ""


def test_api_error_propagates_to_caller():
    class BoomCompletions(FakeCompletions):
        def create(self, **kwargs):
            raise RuntimeError("model 'qwen2' not found")

    client = FakeClient()
    client.chat.completions = BoomCompletions()
    summarizer = OllamaSummarizer(client=client)
    with pytest.raises(RuntimeError):
        summarizer.summarize_topic(KEYWORDS, COMMENTS)
