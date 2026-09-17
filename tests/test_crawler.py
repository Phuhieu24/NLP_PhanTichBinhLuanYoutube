"""Kiểm thử cho src/crawler.py.

Toàn bộ test dùng một client YouTube giả (không có mạng, không tốn quota).
Lỗi API được dựng bằng `googleapiclient.errors.HttpError` thật để phần ánh xạ
lỗi được chạy đúng như khi gọi API thật.
"""

import json

import httplib2
import pytest
from googleapiclient.errors import HttpError

import crawler


# --------------------------------------------------------------------------
# Hạ tầng giả lập
# --------------------------------------------------------------------------

class _FakeRequest:
    """Bắt chước một request của googleapiclient: chỉ có `execute()`."""

    def __init__(self, result):
        self._result = result

    def execute(self):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _FakeEndpoint:
    def __init__(self, handler, calls):
        self._handler = handler
        self._calls = calls

    def list(self, **kwargs):
        self._calls.append(kwargs)
        return _FakeRequest(self._handler(kwargs))


class FakeYouTube:
    """Client giả: trả về các trang dữ liệu đã chuẩn bị sẵn và ghi lại lời gọi."""

    def __init__(self, thread_pages=None, reply_pages=None, video_response=None):
        self.thread_pages = thread_pages or []
        self.reply_pages = reply_pages or {}
        self.video_response = video_response
        self.thread_calls = []
        self.reply_calls = []
        self.video_calls = []

    def commentThreads(self):
        return _FakeEndpoint(self._threads, self.thread_calls)

    def comments(self):
        return _FakeEndpoint(self._replies, self.reply_calls)

    def videos(self):
        return _FakeEndpoint(self._videos, self.video_calls)

    def _threads(self, kwargs):
        return self.thread_pages[int(kwargs.get('pageToken') or 0)]

    def _replies(self, kwargs):
        pages = self.reply_pages[kwargs['parentId']]
        return pages[int(kwargs.get('pageToken') or 0)]

    def _videos(self, kwargs):
        return self.video_response


def page(items, next_index=None):
    """Dựng một trang kết quả; `pageToken` chính là chỉ số trang kế tiếp."""
    result = {'items': list(items)}
    if next_index is not None:
        result['nextPageToken'] = str(next_index)
    return result


def thread_item(thread_id, top_id, total_replies=0, inline=(), snippet=None):
    item = {
        'id': thread_id,
        'snippet': {
            'totalReplyCount': total_replies,
            'topLevelComment': {
                'id': top_id,
                'snippet': snippet or comment_snippet('gốc ' + top_id),
            },
        },
    }
    if inline:
        item['replies'] = {'comments': list(inline)}
    return item


def reply_item(comment_id, snippet=None):
    return {'id': comment_id, 'snippet': snippet or comment_snippet('trả lời ' + comment_id)}


def comment_snippet(text, author='Người xem', like_count=3):
    return {
        'textOriginal': text,
        'authorDisplayName': author,
        'publishedAt': '2026-01-01T00:00:00Z',
        'likeCount': like_count,
    }


def http_error(status, reason, message='loi'):
    content = json.dumps(
        {'error': {'code': status, 'message': message,
                   'errors': [{'reason': reason, 'message': message}]}}
    ).encode('utf-8')
    return HttpError(resp=httplib2.Response({'status': status}), content=content)


# --------------------------------------------------------------------------
# extract_video_id
# --------------------------------------------------------------------------

@pytest.mark.parametrize('url', [
    'https://www.youtube.com/watch?v=exR2qh0zFCA',
    'https://www.youtube.com/watch?t=30&v=exR2qh0zFCA',
    'https://youtu.be/exR2qh0zFCA?si=abcd',
    'https://www.youtube.com/embed/exR2qh0zFCA',
    'https://www.youtube.com/shorts/exR2qh0zFCA',
    'https://www.youtube.com/live/exR2qh0zFCA',
    'exR2qh0zFCA',
])
def test_extract_video_id_supported_shapes(url):
    assert crawler.extract_video_id(url) == 'exR2qh0zFCA'


@pytest.mark.parametrize('value', ['https://www.youtube.com/watch?v=short', 'khong phai link', ''])
def test_extract_video_id_invalid(value):
    with pytest.raises(ValueError):
        crawler.extract_video_id(value)


# --------------------------------------------------------------------------
# Bình luận gốc và phản hồi
# --------------------------------------------------------------------------

def test_inline_replies_consumed_without_extra_call():
    youtube = FakeYouTube(thread_pages=[page([
        thread_item('thread-1', 'top-1', total_replies=2,
                    inline=[reply_item('rep-1'), reply_item('rep-2')]),
    ])])

    df = crawler.get_video_comments('key', 'vid', max_results=10, youtube=youtube)

    assert list(df.columns) == crawler.COLUMNS
    assert list(df['comment_id']) == ['top-1', 'rep-1', 'rep-2']
    assert list(df['is_reply']) == [False, True, True]
    assert list(df['parent_id']) == [None, 'top-1', 'top-1']
    assert list(df['reply_count']) == [2, 0, 0]
    # Phản hồi đã có sẵn trong `part=snippet,replies` nên không tốn thêm quota.
    assert youtube.reply_calls == []
    assert youtube.thread_calls[0]['part'] == 'snippet,replies'


def test_extra_reply_call_uses_top_level_comment_id_and_dedupes():
    youtube = FakeYouTube(
        thread_pages=[page([
            thread_item('thread-1', 'top-1', total_replies=3, inline=[reply_item('rep-1')]),
        ])],
        # API trả lại cả phản hồi đã có sẵn: bản ghi trùng phải bị loại.
        reply_pages={'top-1': [page([reply_item('rep-1'), reply_item('rep-2'), reply_item('rep-3')])]},
    )

    df = crawler.get_video_comments('key', 'vid', max_results=10, youtube=youtube)

    assert len(youtube.reply_calls) == 1
    call = youtube.reply_calls[0]
    assert call['parentId'] == 'top-1'  # không phải 'thread-1'
    assert call['maxResults'] == 100
    assert call['textFormat'] == 'plainText'
    assert list(df['comment_id']) == ['top-1', 'rep-1', 'rep-2', 'rep-3']


def test_no_extra_reply_call_when_inline_covers_all():
    youtube = FakeYouTube(thread_pages=[page([
        thread_item('thread-1', 'top-1', total_replies=1, inline=[reply_item('rep-1')]),
        thread_item('thread-2', 'top-2', total_replies=0),
    ])])

    df = crawler.get_video_comments('key', 'vid', max_results=10, youtube=youtube)

    assert youtube.reply_calls == []
    assert len(df) == 3


def test_include_replies_false_skips_replies():
    youtube = FakeYouTube(
        thread_pages=[page([
            thread_item('thread-1', 'top-1', total_replies=5, inline=[reply_item('rep-1')]),
        ])],
        reply_pages={'top-1': [page([reply_item('rep-2')])]},
    )

    df = crawler.get_video_comments('key', 'vid', max_results=10,
                                    include_replies=False, youtube=youtube)

    assert list(df['comment_id']) == ['top-1']
    assert youtube.reply_calls == []


def test_reply_pagination_follows_next_page_token():
    youtube = FakeYouTube(
        thread_pages=[page([thread_item('thread-1', 'top-1', total_replies=3)])],
        reply_pages={'top-1': [
            page([reply_item('rep-1'), reply_item('rep-2')], next_index=1),
            page([reply_item('rep-3')]),
        ]},
    )

    df = crawler.get_video_comments('key', 'vid', max_results=10, youtube=youtube)

    assert len(youtube.reply_calls) == 2
    assert youtube.reply_calls[1]['pageToken'] == '1'
    assert list(df['comment_id']) == ['top-1', 'rep-1', 'rep-2', 'rep-3']


# --------------------------------------------------------------------------
# Giới hạn số lượng và khử trùng lặp
# --------------------------------------------------------------------------

def test_pagination_stops_at_max_results():
    youtube = FakeYouTube(thread_pages=[
        page([thread_item('t1', 'top-1'), thread_item('t2', 'top-2')], next_index=1),
        page([thread_item('t3', 'top-3')]),
    ])

    df = crawler.get_video_comments('key', 'vid', max_results=2, youtube=youtube)

    assert list(df['comment_id']) == ['top-1', 'top-2']
    # Đã đủ số lượng nên không xin thêm trang nào.
    assert len(youtube.thread_calls) == 1


def test_reply_fetch_stops_at_max_results():
    youtube = FakeYouTube(
        thread_pages=[page([thread_item('thread-1', 'top-1', total_replies=5)])],
        reply_pages={'top-1': [
            page([reply_item('rep-1'), reply_item('rep-2')], next_index=1),
            page([reply_item('rep-3')]),
        ]},
    )

    df = crawler.get_video_comments('key', 'vid', max_results=2, youtube=youtube)

    assert list(df['comment_id']) == ['top-1', 'rep-1']
    assert len(youtube.reply_calls) == 1


def test_duplicate_comment_id_keeps_first():
    youtube = FakeYouTube(thread_pages=[
        page([thread_item('thread-1', 'top-1',
                          snippet=comment_snippet('bản đầu tiên'))], next_index=1),
        page([thread_item('thread-1', 'top-1',
                          snippet=comment_snippet('bản trùng lặp'))]),
    ])

    df = crawler.get_video_comments('key', 'vid', max_results=10, youtube=youtube)

    assert list(df['comment_id']) == ['top-1']
    assert df.loc[0, 'text'] == 'bản đầu tiên'


def test_empty_video_returns_empty_dataframe_with_columns():
    youtube = FakeYouTube(thread_pages=[page([])])

    df = crawler.get_video_comments('key', 'vid', max_results=10, youtube=youtube)

    assert df.empty
    assert list(df.columns) == crawler.COLUMNS


# --------------------------------------------------------------------------
# Tham số và nội dung bản ghi
# --------------------------------------------------------------------------

def test_order_and_text_format_passed_through():
    youtube = FakeYouTube(thread_pages=[page([thread_item('t1', 'top-1')])])
    crawler.get_video_comments('key', 'vid', max_results=5, order='time', youtube=youtube)
    assert youtube.thread_calls[0]['order'] == 'time'
    assert youtube.thread_calls[0]['textFormat'] == 'plainText'

    youtube = FakeYouTube(thread_pages=[page([thread_item('t1', 'top-1')])])
    crawler.get_video_comments('key', 'vid', max_results=5, youtube=youtube)
    assert youtube.thread_calls[0]['order'] == 'relevance'


def test_invalid_order_raises_value_error():
    with pytest.raises(ValueError):
        crawler.get_video_comments('key', 'vid', order='popular',
                                   youtube=FakeYouTube(thread_pages=[page([])]))


def test_text_falls_back_to_text_display():
    snippet = {'textDisplay': 'nội dung <b>hiển thị</b>', 'authorDisplayName': 'A',
               'publishedAt': '2026-01-01T00:00:00Z', 'likeCount': 0}
    youtube = FakeYouTube(thread_pages=[page([thread_item('t1', 'top-1', snippet=snippet)])])

    df = crawler.get_video_comments('key', 'vid', max_results=5, youtube=youtube)

    assert df.loc[0, 'text'] == 'nội dung <b>hiển thị</b>'
    assert df.loc[0, 'author'] == 'A'


def test_on_progress_receives_counts_and_messages():
    youtube = FakeYouTube(thread_pages=[page([thread_item('t1', 'top-1')])])
    events = []

    crawler.get_video_comments('key', 'vid', max_results=5, youtube=youtube,
                               on_progress=lambda n, message: events.append((n, message)))

    assert events
    assert events[-1][0] == 1
    assert all(isinstance(message, str) for _, message in events)


# --------------------------------------------------------------------------
# Ánh xạ lỗi
# --------------------------------------------------------------------------

@pytest.mark.parametrize('status, reason, expected', [
    (403, 'commentsDisabled', 'commentsDisabled'),
    (403, 'quotaExceeded', 'quotaExceeded'),
    (404, 'videoNotFound', 'videoNotFound'),
    (403, 'forbidden', 'forbidden'),
    (400, 'keyInvalid', 'invalidKey'),
    (400, 'badRequest', 'invalidKey'),
    (500, 'backendError', 'other'),
])
def test_http_error_mapped_to_crawl_error(status, reason, expected):
    youtube = FakeYouTube(thread_pages=[http_error(status, reason)])

    with pytest.raises(crawler.CrawlError) as excinfo:
        crawler.get_video_comments('key', 'vid', max_results=5, youtube=youtube)

    assert excinfo.value.reason == expected
    assert str(excinfo.value).strip()


def test_comments_disabled_message_is_vietnamese_and_actionable():
    youtube = FakeYouTube(thread_pages=[http_error(403, 'commentsDisabled')])

    with pytest.raises(crawler.CrawlError) as excinfo:
        crawler.get_video_comments('key', 'vid', max_results=5, youtube=youtube)

    assert 'tắt bình luận' in str(excinfo.value)


def test_quota_exceeded_message_mentions_quota():
    youtube = FakeYouTube(thread_pages=[http_error(403, 'quotaExceeded')])

    with pytest.raises(crawler.CrawlError) as excinfo:
        crawler.get_video_comments('key', 'vid', max_results=5, youtube=youtube)

    assert 'hạn mức' in str(excinfo.value)


def test_non_http_exception_propagates():
    youtube = FakeYouTube(thread_pages=[RuntimeError('lỗi mạng')])

    with pytest.raises(RuntimeError):
        crawler.get_video_comments('key', 'vid', max_results=5, youtube=youtube)


def test_reply_error_is_reported_and_top_level_comments_kept():
    youtube = FakeYouTube(
        thread_pages=[page([
            thread_item('thread-1', 'top-1', total_replies=2),
            thread_item('thread-2', 'top-2'),
        ])],
        reply_pages={'top-1': [http_error(403, 'forbidden')]},
    )
    events = []

    df = crawler.get_video_comments('key', 'vid', max_results=10, youtube=youtube,
                                    on_progress=lambda n, m: events.append(m))

    assert list(df['comment_id']) == ['top-1', 'top-2']
    assert any('phản hồi' in message and 'top-1' in message for message in events)
    assert any('không lấy đủ phản hồi' in message for message in events)


# --------------------------------------------------------------------------
# Thông tin video
# --------------------------------------------------------------------------

def test_get_video_metadata_returns_fields():
    youtube = FakeYouTube(video_response={'items': [{
        'snippet': {'title': 'Anh Trai Say Hi', 'channelTitle': 'Kênh ABC',
                    'publishedAt': '2026-01-01T00:00:00Z',
                    'thumbnails': {'high': {'url': 'https://i.ytimg.com/high.jpg'},
                                   'default': {'url': 'https://i.ytimg.com/default.jpg'}}},
        'statistics': {'viewCount': '1234', 'likeCount': '56', 'commentCount': '78'},
    }]})

    meta = crawler.get_video_metadata('key', 'vid', youtube=youtube)

    assert meta == {
        'title': 'Anh Trai Say Hi',
        'channel_title': 'Kênh ABC',
        'published_at': '2026-01-01T00:00:00Z',
        'view_count': 1234,
        'like_count': 56,
        'comment_count': 78,
        'thumbnail_url': 'https://i.ytimg.com/high.jpg',
    }
    assert youtube.video_calls[0]['part'] == 'snippet,statistics'


def test_get_video_metadata_returns_none_on_error_or_empty():
    assert crawler.get_video_metadata(
        'key', 'vid', youtube=FakeYouTube(video_response={'items': []})) is None

    class Broken(FakeYouTube):
        def _videos(self, kwargs):
            return http_error(404, 'videoNotFound')

    assert crawler.get_video_metadata('key', 'vid', youtube=Broken()) is None
