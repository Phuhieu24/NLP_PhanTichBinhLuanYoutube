"""Thu thập bình luận YouTube qua YouTube Data API v3.

Module này cố gắng tiết kiệm quota: mỗi lần gọi `commentThreads.list`,
`comments.list` hay `videos.list` đều tốn 1 đơn vị quota (mặc định 10.000
đơn vị/ngày). Vì vậy ta xin sẵn `part="snippet,replies"` để dùng các phản hồi
được trả về kèm theo, và chỉ gọi thêm `comments.list` khi một bình luận gốc
còn phản hồi chưa lấy được.
"""

import argparse
import json
import os
import re
import sys

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Thứ tự cột của DataFrame trả về (cố định theo hợp đồng giao diện).
COLUMNS = [
    'comment_id',
    'parent_id',
    'is_reply',
    'author',
    'published_at',
    'like_count',
    'reply_count',
    'text',
]

# Các giá trị hợp lệ của tham số sắp xếp bình luận gốc.
VALID_ORDERS = ('relevance', 'time')

# Số phần tử tối đa YouTube API cho phép lấy trong một trang.
_MAX_PAGE_SIZE = 100


class CrawlError(Exception):
    """Lỗi khi gọi YouTube Data API, kèm thông điệp tiếng Việt cho người dùng.

    Thuộc tính `reason` nhận một trong các giá trị: 'commentsDisabled',
    'quotaExceeded', 'videoNotFound', 'forbidden', 'invalidKey', 'other'.
    """

    def __init__(self, message, reason='other'):
        super().__init__(message)
        self.reason = reason


# Ánh xạ mã lỗi của Google (viết thường) sang `reason` trong hợp đồng.
_REASON_ALIASES = {
    'commentsdisabled': 'commentsDisabled',
    'quotaexceeded': 'quotaExceeded',
    'dailylimitexceeded': 'quotaExceeded',
    'ratelimitexceeded': 'quotaExceeded',
    'userratelimitexceeded': 'quotaExceeded',
    'videonotfound': 'videoNotFound',
    'commentnotfound': 'videoNotFound',
    'notfound': 'videoNotFound',
    'forbidden': 'forbidden',
    'keyinvalid': 'invalidKey',
    'keyexpired': 'invalidKey',
    'badrequest': 'invalidKey',
}

# Ánh xạ dự phòng khi payload lỗi không có trường `reason`.
_STATUS_REASONS = {
    400: 'invalidKey',
    401: 'invalidKey',
    403: 'forbidden',
    404: 'videoNotFound',
}

# Thông điệp tiếng Việt: nói rõ người dùng cần làm gì tiếp theo.
_REASON_MESSAGES = {
    'commentsDisabled': (
        'Video này đã tắt bình luận nên không thu thập được dữ liệu. '
        'Hãy chọn một video khác.'
    ),
    'quotaExceeded': (
        'Đã hết hạn mức YouTube Data API cho hôm nay (mặc định 10.000 đơn vị/ngày). '
        'Hãy đợi hạn mức đặt lại vào nửa đêm giờ Thái Bình Dương, giảm số bình luận '
        'cần lấy, hoặc dùng API key của một project khác.'
    ),
    'videoNotFound': (
        'Không tìm thấy video. Hãy kiểm tra lại đường dẫn hoặc ID video.'
    ),
    'forbidden': (
        'YouTube từ chối yêu cầu (403). Hãy kiểm tra quyền của API key và xem video '
        'có đang ở chế độ riêng tư hay bị giới hạn khu vực không.'
    ),
    'invalidKey': (
        'API key không hợp lệ hoặc chưa bật YouTube Data API v3. Hãy kiểm tra '
        'YOUTUBE_API_KEY trong file .env và bật API trong Google Cloud Console.'
    ),
}


def _error_payload(error):
    """Đọc phần JSON trong `HttpError.content`; trả về dict rỗng nếu không đọc được."""
    content = getattr(error, 'content', None)
    if not content:
        return {}
    if isinstance(content, bytes):
        try:
            content = content.decode('utf-8', errors='replace')
        except Exception:
            return {}
    try:
        payload = json.loads(content)
    except (ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _http_error_to_crawl_error(error):
    """Chuyển `HttpError` của googleapiclient thành `CrawlError` dễ hiểu."""
    payload = _error_payload(error).get('error', {})
    if not isinstance(payload, dict):
        payload = {}

    raw_reason = ''
    errors = payload.get('errors')
    if isinstance(errors, list) and errors and isinstance(errors[0], dict):
        raw_reason = str(errors[0].get('reason', '') or '')

    status = getattr(getattr(error, 'resp', None), 'status', None)
    try:
        status = int(status)
    except (TypeError, ValueError):
        status = None

    reason = _REASON_ALIASES.get(raw_reason.lower())
    if reason is None:
        reason = _STATUS_REASONS.get(status, 'other')

    message = _REASON_MESSAGES.get(reason)
    if message is None:
        detail = payload.get('message') or raw_reason or str(error)
        message = 'Lỗi khi gọi YouTube Data API (mã {}): {}'.format(
            status if status is not None else 'không rõ', detail
        )
    return CrawlError(message, reason)


def _notify(on_progress, n_collected, message):
    """Gọi callback tiến độ nếu có; lỗi trong callback không làm hỏng việc thu thập."""
    if on_progress is None:
        return
    try:
        on_progress(n_collected, message)
    except Exception:
        pass


def _to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_client(api_key, youtube=None):
    """Trả về client được truyền vào, hoặc dựng client thật từ API key."""
    if youtube is not None:
        return youtube
    return build('youtube', 'v3', developerKey=api_key, cache_discovery=False)


# --------------------------------------------------------------------------
# Tách ID video
# --------------------------------------------------------------------------

_ID_PATTERN = r'[A-Za-z0-9_-]{11}'

# Các dạng đường dẫn YouTube được hỗ trợ.
_URL_PATTERNS = (
    r'(?:[?&]|^)v=(' + _ID_PATTERN + r')',
    r'youtu\.be/(' + _ID_PATTERN + r')',
    r'/embed/(' + _ID_PATTERN + r')',
    r'/shorts/(' + _ID_PATTERN + r')',
    r'/live/(' + _ID_PATTERN + r')',
    r'/v/(' + _ID_PATTERN + r')',
)


def extract_video_id(url_or_id):
    """Tách ID video (11 ký tự) từ đường dẫn YouTube hoặc từ chính ID đó.

    Hỗ trợ `watch?v=`, `youtu.be/`, `/embed/`, `/shorts/`, `/live/`, `/v/`
    và ID trần. Ném `ValueError` kèm thông điệp tiếng Việt nếu không hợp lệ.
    """
    if not isinstance(url_or_id, str) or not url_or_id.strip():
        raise ValueError('Chưa nhập đường dẫn hoặc ID video YouTube.')

    value = url_or_id.strip()
    if re.fullmatch(_ID_PATTERN, value):
        return value

    for pattern in _URL_PATTERNS:
        match = re.search(pattern, value)
        if match:
            return match.group(1)

    raise ValueError(
        'Đường dẫn YouTube không hợp lệ. Hãy dán link dạng '
        'https://www.youtube.com/watch?v=..., https://youtu.be/..., '
        'https://www.youtube.com/shorts/... hoặc chính ID video (11 ký tự).'
    )


# --------------------------------------------------------------------------
# Thông tin video
# --------------------------------------------------------------------------

def _pick_thumbnail(thumbnails):
    """Chọn ảnh đại diện có độ phân giải cao nhất còn khả dụng."""
    if not isinstance(thumbnails, dict):
        return ''
    for key in ('maxres', 'standard', 'high', 'medium', 'default'):
        item = thumbnails.get(key)
        if isinstance(item, dict) and item.get('url'):
            return item['url']
    return ''


def get_video_metadata(api_key, video_id, youtube=None):
    """Lấy thông tin mô tả của video (1 đơn vị quota).

    Trả về `None` nếu có bất kỳ lỗi nào: giao diện chỉ dùng dữ liệu này để
    hiển thị thêm, không nên vì nó mà làm hỏng cả phiên phân tích.
    """
    try:
        client = _build_client(api_key, youtube)
        response = client.videos().list(
            part='snippet,statistics',
            id=video_id,
        ).execute()
        items = response.get('items') or []
        if not items:
            return None
        snippet = items[0].get('snippet', {}) or {}
        stats = items[0].get('statistics', {}) or {}
        return {
            'title': snippet.get('title', ''),
            'channel_title': snippet.get('channelTitle', ''),
            'published_at': snippet.get('publishedAt', ''),
            'view_count': _to_int(stats.get('viewCount')),
            'like_count': _to_int(stats.get('likeCount')),
            'comment_count': _to_int(stats.get('commentCount')),
            'thumbnail_url': _pick_thumbnail(snippet.get('thumbnails')),
        }
    except Exception:
        return None


# --------------------------------------------------------------------------
# Thu thập bình luận
# --------------------------------------------------------------------------

def _parse_comment(snippet, comment_id, parent_id, reply_count):
    """Dựng một bản ghi bình luận từ phần `snippet` mà API trả về."""
    snippet = snippet or {}
    text = snippet.get('textOriginal')
    if not text:
        text = snippet.get('textDisplay', '')
    return {
        'comment_id': comment_id,
        'parent_id': parent_id,
        'is_reply': parent_id is not None,
        'author': snippet.get('authorDisplayName') or 'Ẩn danh',
        'published_at': snippet.get('publishedAt', ''),
        'like_count': _to_int(snippet.get('likeCount')),
        'reply_count': _to_int(reply_count),
        'text': text or '',
    }


def _append_record(records, seen, record):
    """Thêm bản ghi nếu ID chưa xuất hiện (khử trùng lặp, giữ bản đầu tiên)."""
    comment_id = record.get('comment_id')
    if comment_id in seen:
        return False
    seen.add(comment_id)
    records.append(record)
    return True


def _fetch_remaining_replies(youtube, parent_id, records, seen, max_results):
    """Lấy nốt phản hồi của một bình luận gốc qua `comments.list`.

    Trả về thông điệp lỗi (chuỗi) nếu phải dừng giữa chừng, ngược lại `None`.
    Luôn truyền ID của bình luận gốc (`snippet.topLevelComment.id`) chứ không
    phải ID của chuỗi hội thoại: đó mới là `parentId` đúng theo tài liệu.
    """
    request = youtube.comments().list(
        part='snippet',
        parentId=parent_id,
        maxResults=_MAX_PAGE_SIZE,
        textFormat='plainText',
    )
    while request is not None and len(records) < max_results:
        try:
            response = request.execute()
        except HttpError as error:
            # Không nuốt lỗi: dừng lấy phản hồi của bình luận này rồi báo ra ngoài;
            # các bình luận đã thu được vẫn được giữ lại.
            return str(_http_error_to_crawl_error(error))

        for item in response.get('items', []):
            if len(records) >= max_results:
                break
            _append_record(
                records,
                seen,
                _parse_comment(item.get('snippet'), item.get('id'), parent_id, 0),
            )

        token = response.get('nextPageToken')
        if token and len(records) < max_results:
            request = youtube.comments().list(
                part='snippet',
                parentId=parent_id,
                maxResults=_MAX_PAGE_SIZE,
                pageToken=token,
                textFormat='plainText',
            )
        else:
            request = None
    return None


def _to_dataframe(records, max_results):
    """Chuyển danh sách bản ghi thành DataFrame đúng thứ tự cột và kiểu dữ liệu."""
    df = pd.DataFrame(records, columns=COLUMNS)
    df = df.drop_duplicates(subset='comment_id', keep='first')
    df = df.head(max_results).reset_index(drop=True)
    if not df.empty:
        df['is_reply'] = df['is_reply'].astype(bool)
        df['like_count'] = df['like_count'].astype(int)
        df['reply_count'] = df['reply_count'].astype(int)
    return df


def get_video_comments(api_key, video_id, max_results=500, include_replies=True,
                       order='relevance', on_progress=None, youtube=None):
    """Thu thập tối đa `max_results` bình luận (kèm phản hồi nếu bật).

    `on_progress(n_collected, message)` được gọi sau mỗi trang để cập nhật
    giao diện; `youtube` cho phép truyền client giả khi kiểm thử. Lỗi API được
    chuyển thành `CrawlError`, các lỗi khác được ném nguyên trạng.
    """
    if order not in VALID_ORDERS:
        raise ValueError(
            "Tham số order chỉ nhận 'relevance' (phổ biến) hoặc 'time' (mới nhất)."
        )
    max_results = max(1, _to_int(max_results, 1))

    client = _build_client(api_key, youtube)
    records = []
    seen = set()
    reply_failures = 0

    _notify(on_progress, 0,
            'Bắt đầu thu thập bình luận của video {}...'.format(video_id))

    request = client.commentThreads().list(
        part='snippet,replies',
        videoId=video_id,
        maxResults=min(_MAX_PAGE_SIZE, max_results),
        order=order,
        textFormat='plainText',
    )

    while request is not None and len(records) < max_results:
        try:
            response = request.execute()
        except HttpError as error:
            raise _http_error_to_crawl_error(error) from error

        for item in response.get('items', []):
            if len(records) >= max_results:
                break
            snippet = item.get('snippet', {}) or {}
            top_level = snippet.get('topLevelComment', {}) or {}
            # `parentId` đúng là ID của bình luận gốc, không phải ID chuỗi hội thoại.
            top_id = top_level.get('id') or item.get('id')
            total_replies = _to_int(snippet.get('totalReplyCount'))

            _append_record(
                records,
                seen,
                _parse_comment(top_level.get('snippet'), top_id, None, total_replies),
            )

            if not include_replies or total_replies <= 0 or len(records) >= max_results:
                continue

            # Ưu tiên dùng phản hồi trả về kèm theo (không tốn thêm quota).
            inline_replies = (item.get('replies', {}) or {}).get('comments', []) or []
            for reply in inline_replies:
                if len(records) >= max_results:
                    break
                _append_record(
                    records,
                    seen,
                    _parse_comment(reply.get('snippet'), reply.get('id'), top_id, 0),
                )

            # Chỉ gọi thêm `comments.list` khi vẫn còn phản hồi chưa lấy được.
            if len(inline_replies) >= total_replies or len(records) >= max_results:
                continue

            failure = _fetch_remaining_replies(
                client, top_id, records, seen, max_results
            )
            if failure:
                reply_failures += 1
                _notify(
                    on_progress,
                    len(records),
                    'Không lấy được đầy đủ phản hồi của bình luận {}: {}'.format(
                        top_id, failure
                    ),
                )

        _notify(on_progress, len(records),
                'Đã thu thập {} bình luận.'.format(len(records)))

        token = response.get('nextPageToken')
        if token and len(records) < max_results:
            request = client.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                maxResults=min(_MAX_PAGE_SIZE, max_results - len(records)),
                order=order,
                pageToken=token,
                textFormat='plainText',
            )
        else:
            request = None

    df = _to_dataframe(records, max_results)
    summary = 'Hoàn tất: thu được {} bình luận.'.format(len(df))
    if reply_failures:
        summary += ' Có {} bình luận không lấy đủ phản hồi.'.format(reply_failures)
    _notify(on_progress, len(df), summary)
    return df


# --------------------------------------------------------------------------
# Chạy từ dòng lệnh
# --------------------------------------------------------------------------

def _fix_stdout_encoding():
    """Sửa lỗi font tiếng Việt trên terminal Windows, bỏ qua nếu không cần."""
    stream = sys.stdout
    if stream is None:
        return
    encoding = (getattr(stream, 'encoding', '') or '').lower()
    if encoding in ('utf-8', 'utf8'):
        return
    try:
        stream.reconfigure(encoding='utf-8')
    except Exception:
        pass


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Thu thập bình luận YouTube và lưu ra file CSV.'
    )
    parser.add_argument('--url', required=True,
                        help='Đường dẫn video YouTube hoặc ID video.')
    parser.add_argument('--max', type=int, default=1000, dest='max_results',
                        help='Số bình luận tối đa cần lấy (mặc định 1000).')
    parser.add_argument('--no-replies', action='store_true',
                        help='Chỉ lấy bình luận gốc, bỏ qua phản hồi.')
    parser.add_argument('--order', choices=list(VALID_ORDERS), default='relevance',
                        help="Thứ tự bình luận gốc: 'relevance' hoặc 'time'.")
    parser.add_argument('--out', default=None,
                        help='Đường dẫn file CSV đầu ra (mặc định data/comments.csv).')
    return parser.parse_args(argv)


def main(argv=None):
    _fix_stdout_encoding()
    args = _parse_args(argv)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(project_root, '.env'))
    except ImportError:
        pass

    api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        print('Lỗi: không tìm thấy YOUTUBE_API_KEY trong file .env')
        return 1

    try:
        video_id = extract_video_id(args.url)
    except ValueError as error:
        print('Lỗi: {}'.format(error))
        return 1

    try:
        df = get_video_comments(
            api_key,
            video_id,
            max_results=args.max_results,
            include_replies=not args.no_replies,
            order=args.order,
            on_progress=lambda n, message: print(message),
        )
    except CrawlError as error:
        print('Lỗi ({}): {}'.format(error.reason, error))
        return 1

    output_path = args.out or os.path.join(project_root, 'data', 'comments.csv')
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    # utf-8-sig để Excel mở file không bị lỗi font tiếng Việt.
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print('Đã lưu {} bình luận vào: {}'.format(len(df), output_path))
    return 0


if __name__ == '__main__':
    sys.exit(main())
