import os
import sys
import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv
import re

def get_video_comments(api_key, video_id, max_results=500):
    # Khởi tạo API
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    comments = []
    
    # Request lấy danh sách comment cha (top-level)
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )
    
    print(f"Bắt đầu thu thập bình luận từ Video ID: {video_id}...")
    
    while request and len(comments) < max_results:
        try:
            response = request.execute()
            
            for item in response['items']:
                top_comment_snippet = item['snippet']['topLevelComment']['snippet']
                comments.append([
                    top_comment_snippet.get('authorDisplayName', 'Ẩn danh'),
                    top_comment_snippet.get('publishedAt', ''),
                    top_comment_snippet.get('likeCount', 0),
                    top_comment_snippet.get('textDisplay', '')
                ])
                
                # Kiểm tra và lấy các bình luận con (replies)
                if item['snippet']['totalReplyCount'] > 0:
                    try:
                        replies_request = youtube.comments().list(
                            part="snippet",
                            parentId=item['id'],
                            maxResults=100,
                            textFormat="plainText"
                        )
                        while replies_request and len(comments) < max_results:
                            replies_response = replies_request.execute()
                            for reply_item in replies_response.get('items', []):
                                reply_snippet = reply_item['snippet']
                                comments.append([
                                    reply_snippet.get('authorDisplayName', 'Ẩn danh'),
                                    reply_snippet.get('publishedAt', ''),
                                    reply_snippet.get('likeCount', 0),
                                    reply_snippet.get('textDisplay', '')
                                ])
                            if 'nextPageToken' in replies_response:
                                replies_request = youtube.comments().list(
                                    part="snippet",
                                    parentId=item['id'],
                                    maxResults=100,
                                    pageToken=replies_response['nextPageToken'],
                                    textFormat="plainText"
                                )
                            else:
                                break
                    except Exception as e:
                        print(f"Lỗi khi cào replies: {e}")
            
            # Kiểm tra xem còn trang tiếp theo không
            if 'nextPageToken' in response:
                print(f"Đã lấy {len(comments)} bình luận. Đang sang trang tiếp theo...")
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    pageToken=response['nextPageToken'],
                    maxResults=100,
                    textFormat="plainText"
                )
            else:
                break
                
        except Exception as e:
            raise Exception(f"Lỗi gọi API YouTube: {e}")
            
    df = pd.DataFrame(comments, columns=['author', 'published_at', 'like_count', 'text'])
    df = df.head(max_results)
    print(f"Đã lấy thành công tổng cộng {len(df)} bình luận.")
    return df

def extract_video_id(url):
    match = re.search(r'(?:v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})', url)
    if match:
        return match.group(1)
    else:
        raise ValueError("URL YouTube không hợp lệ hoặc không tìm thấy Video ID (yêu cầu 11 ký tự).")

if __name__ == "__main__":
    # Fix unicode printing on Windows terminal
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
        
    # Đọc biến môi trường từ file .env
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
    api_key = os.getenv("YOUTUBE_API_KEY")
    
    if not api_key:
        print("Lỗi: Không tìm thấy YOUTUBE_API_KEY trong file .env")
        sys.exit(1)
        
    video_url = "https://www.youtube.com/watch?v=exR2qh0zFCA"
    video_id = extract_video_id(video_url)
    
    # Tạo thư mục data nếu chưa có ở thư mục gốc dự án
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Thu thập tối đa 1000 bình luận
    df = get_video_comments(api_key, video_id, max_results=1000)
    
    # Lưu ra file CSV (dùng utf-8-sig để Excel đọc được tiếng Việt không bị lỗi font)
    output_path = os.path.join(data_dir, 'comments.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Đã lưu kết quả thành công vào: {output_path}")
