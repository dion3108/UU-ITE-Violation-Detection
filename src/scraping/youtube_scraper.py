from googleapiclient.discovery import build
import csv
import os
import time
import sys
from datetime import datetime

# Load API Key from environment variable (NEVER hardcode keys in source code)
# Set it in your terminal: export YOUTUBE_API_KEY="your_key_here"
# Or create a .env file and load with python-dotenv
api_key = os.environ.get('YOUTUBE_API_KEY')
if not api_key:
    raise ValueError("YOUTUBE_API_KEY environment variable is not set. "
                     "Run: export YOUTUBE_API_KEY='your_api_key_here'")

# Daftar URL video yang akan discrape (tambahkan atau ubah sesuai kebutuhan)
VIDEO_URLS = [
    # 'https://www.youtube.com/watch?v=jNQXAC9IVRw',  # Hello World - First YouTube video
    # 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',  # Rick Astley - Never Gonna Give You Up
    # 'https://www.youtube.com/watch?v=E-hweFwSC_s', 
    # 'https://www.youtube.com/watch?v=OwwE5rCfk5k',
    # 'https://www.youtube.com/watch?v=DmogrGm8zWU'   
    # Tambahkan URL video YouTube lainnya di sini
#    "https://youtu.be/XelqyYvWYe0?si=FvFDErUUiFx1K4Pg",
#    'https://youtu.be/MMtpAOGpO8A?si=jNFp5Garqlj4vs0i',
#    'https://youtu.be/gsiV0J1B2n4?si=hSIw7CzcGPmdIopS',
#    'https://youtu.be/p4mTiGyed0k?si=JzcDCiQzqmECIHaU',
#    'https://youtu.be/x46NSqOQerk?si=iPESzFJovcYyZRqA',
#    'https://youtu.be/vwejrT9XHWI?si=XWxLDIZCQerz76Ie',
#    'https://youtu.be/VZ-oD3AEODA?si=wKeeXXfgnKtRwLBt',
#    'https://youtu.be/eZ6fPQY95qY?si=kgOTR3y25iULZC25',
#    'https://youtu.be/qqGhtqIUke4?si=U6yO8b4M7sZYRU_m',
#    'https://youtu.be/3GIuRlKBHKc?si=x2ri2XYhOqh2QKpn',
#    'https://youtu.be/D8tOI1VGyis?si=2-qNjG6jbqWB9RRd',
#    'https://youtu.be/zXpMNClRxQQ?si=1iDNYKgROcfR_mgt',
#    'https://youtu.be/dKbfT0eCki8?si=nF4A0XpetFOHpcG8',
#    'https://youtu.be/ki9_6jDnJyI?si=FZ6ocQJ4K6tlFc9b',
#    'https://youtu.be/JzfnXaPkF_c?si=ODLT1qfuEgbPu0kk',
#    'https://youtu.be/j-TACmv_0vI?si=sfjKiIAdsr0jsJdb',
#    'https://youtu.be/-1Lz2oHN8YU?si=1dDXOSAlrAFTWyKt',
#    'https://youtu.be/HTTCSz2SJqM?si=PfFLJXmLG7Rh1V-0',
#    'https://youtu.be/-MnlcCTEbl0?si=ZXcR-ey3dlMboYUz',
#    'https://youtu.be/nb2nkwzsZNU?si=MbqiPFrAa3LryU5C',
#    'https://youtu.be/DDMiZ6AZZvI?si=ZP1g9GZmihkG23do',
#    'https://youtu.be/MK8qwclLnls?si=ydGREuA7xDZDfL0j',
#    'https://youtu.be/1OELu8oi_KE?si=fdO8T1V32WdLTzhb',
#    'https://youtu.be/sTrB_fTUoZU?si=Zj1b83FX7x3CBjHE',
#    'https://youtu.be/EI17botAjZ4?si=_GONRZSxszVG5Oqg',
#    'https://youtu.be/7af05ZCU_oo?si=7YEcKq-CPIQ3M0EW',
    'https://www.youtube.com/watch?v=DPqvYfF5R6Q'
]

# Setup log file dengan timestamp
log_file = f"youtube_scraper_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def log_message(message, level="INFO"):
    """Write log message to console and log file with color coding for terminal"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Color codes for terminal
    colors = {
        "INFO": "\033[0m",       # Default/White
        "SUCCESS": "\033[92m",   # Green
        "WARNING": "\033[93m",   # Yellow
        "ERROR": "\033[91m",     # Red
        "PROGRESS": "\033[94m",  # Blue
    }
    reset = "\033[0m"
    
    # Format log entry
    log_entry = f"[{timestamp}] [{level}] {message}"
    
    # Print to console with color if available
    if sys.stdout.isatty():  # Check if running in terminal that supports colors
        colored_level = f"{colors.get(level, '')}{level}{reset}"
        print(f"[{timestamp}] [{colored_level}] {message}")
    else:
        print(log_entry)
    
    # Write to log file (without color codes)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry + '\n')

def progress_bar(current, total, bar_length=50):
    """Display a progress bar in the terminal"""
    percent = min(float(current) / total, 1.0)
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    return f"[{bar}] {percent:.1%} ({current}/{total})"

def get_video_details(video_id):
    """Get video title and channel name for better logging"""
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        response = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()
        
        if response['items']:
            video = response['items'][0]['snippet']
            return {
                'title': video['title'],
                'channel': video['channelTitle']
            }
        return {'title': 'Unknown', 'channel': 'Unknown'}
    except Exception as e:
        log_message(f"Tidak dapat mengambil detail video: {e}", "WARNING")
        return {'title': 'Unknown', 'channel': 'Unknown'}

def get_comments_and_replies(video_id, video_url, max_comments):
    """Mengambil komentar dan balasan dari video berdasarkan video_id."""
    all_comments = []
    total_comments_collected = 0
    page_count = 0
    
    video_details = get_video_details(video_id)
    log_message(f"Mulai mengumpulkan komentar untuk video: '{video_details['title']}' - {video_id}", "INFO")
    
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        request = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText" 
        )

        while request and total_comments_collected < max_comments:
            start_time = time.time()
            
            try:
                response = request.execute()
                page_count += 1
                page_items = 0
                total_items_on_page = len(response.get('items', []))
                
                log_message(f"Halaman {page_count}: Memproses {total_items_on_page} komentar utama", "PROGRESS")
                
                for i, item in enumerate(response.get('items', [])):
                    # Process top-level comment
                    comment_snippet = item['snippet']['topLevelComment']['snippet']
                    text_original = comment_snippet.get('textOriginal', '')
                    
                    if text_original:
                        comment_info = {
                            'komentar_id': item['id'],
                            'Username': comment_snippet.get('authorDisplayName', 'Anonymous'),
                            'T_komentar': text_original,
                            'link': video_url
                        }
                        all_comments.append(comment_info)
                        total_comments_collected += 1
                        page_items += 1
                    
                    # Process replies if available
                    if 'replies' in item:
                        reply_count = len(item['replies'].get('comments', []))
                        if reply_count > 0:
                            log_message(f"  Komentar {i+1}/{total_items_on_page} memiliki {reply_count} balasan", "PROGRESS")
                            
                            for reply_item in item['replies'].get('comments', []):
                                reply_snippet = reply_item.get('snippet', {})
                                reply_text = reply_snippet.get('textOriginal', '')
                                
                                if reply_text:
                                    reply_info = {
                                        'komentar_id': reply_item.get('id', ''),
                                        'Username': reply_snippet.get('authorDisplayName', 'Anonymous'),
                                        'T_komentar': reply_text,
                                        'link': video_url
                                    }
                                    all_comments.append(reply_info)
                                    total_comments_collected += 1
                                    page_items += 1
                    
                    # Show progress
                    if (i+1) % 10 == 0 or i+1 == total_items_on_page:
                        progress = progress_bar(i+1, total_items_on_page)
                        log_message(f"  {progress}", "PROGRESS")
                    
                    if total_comments_collected >= max_comments:
                        log_message(f"Mencapai batas maksimum {max_comments} komentar", "INFO")
                        break
                
                # Calculate processing time and rate
                elapsed_time = time.time() - start_time
                processing_rate = page_items / elapsed_time if elapsed_time > 0 else 0
                
                log_message(
                    f"Halaman {page_count} selesai: {page_items} item ({processing_rate:.1f} item/detik) - "
                    f"Total: {total_comments_collected}", 
                    "SUCCESS"
                )
                
                # Check if there are more comments to fetch
                if ('nextPageToken' in response and 
                    total_comments_collected < max_comments):
                    request = youtube.commentThreads().list(
                        part="snippet,replies",
                        videoId=video_id,
                        maxResults=100,
                        pageToken=response['nextPageToken'],
                        textFormat="plainText" 
                    )
                    delay = min(1.0, 100/processing_rate if processing_rate > 0 else 1.0)
                    log_message(f"Menunggu {delay:.1f} detik sebelum halaman berikutnya...", "INFO")
                    time.sleep(delay)  # Dynamic delay based on processing rate
                else:
                    log_message("Tidak ada halaman komentar lebih lanjut", "INFO")
                    break
                    
            except Exception as e:
                log_message(f"Error saat memproses halaman {page_count}: {str(e)}", "ERROR")
                break
                
        log_message(f"Selesai mengumpulkan {total_comments_collected} item untuk video {video_id}", "SUCCESS")
        return all_comments
        
    except Exception as e:
        log_message(f"Error saat mengakses YouTube API untuk video {video_id}: {str(e)}", "ERROR")
        return []

def save_to_csv(comments_data, filename):
    """Save all comments to CSV file"""
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['komentar_id', 'Username', 'T_komentar', 'link']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            total_rows = len(comments_data)
            log_message(f"Menyimpan {total_rows} item ke CSV...", "INFO")
            
            for i, comment in enumerate(comments_data):
                writer.writerow(comment)
                
                # Show progress every 5% or for every 1000 items
                if total_rows > 20 and (i+1) % max(int(total_rows/20), 1000) == 0:
                    progress = progress_bar(i+1, total_rows)
                    log_message(f"Menyimpan: {progress}", "PROGRESS")
        
        log_message(f"Berhasil menyimpan {len(comments_data)} item ke {filename}", "SUCCESS")
        return True
    except Exception as e:
        log_message(f"Error saat menyimpan ke CSV: {str(e)}", "ERROR")
        return False

def extract_video_id(url):
    """Extract YouTube video ID from various URL formats"""
    try:
        if "v=" in url:
            return url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            return url.split("be/")[1].split("?")[0]
        elif "embed/" in url:
            return url.split("embed/")[1].split("?")[0]
        else:
            log_message(f"Format URL tidak didukung: {url}", "ERROR")
            return None
    except Exception as e:
        log_message(f"Error saat mengekstrak ID video: {e}", "ERROR")
        return None

def scrape_multiple_videos(urls, output_csv='all_youtube_comments(1).csv', max_comments_per_video=1000):
    """Scrape comments from multiple YouTube videos"""
    log_message(f"MEMULAI SCRAPING KOMENTAR YOUTUBE", "INFO")
    log_message(f"Target: {len(urls)} video, Max {max_comments_per_video} komentar per video", "INFO")
    log_message(f"Output: {output_csv}", "INFO")
    log_message("-" * 70, "INFO")
    
    start_time_total = time.time()
    all_comments = []
    success_count = 0
    fail_count = 0
    
    for i, url in enumerate(urls):
        try:
            log_message(f"\n{'='*70}", "INFO")
            log_message(f"Video {i+1}/{len(urls)}: {url}", "INFO")
            
            video_id = extract_video_id(url)
            if not video_id:
                log_message(f"Gagal mengekstrak ID video dari URL: {url}", "ERROR")
                fail_count += 1
                continue
            
            start_time_video = time.time()
            
            # Get comments for this video
            video_comments = get_comments_and_replies(video_id, url, max_comments_per_video)
            
            # Calculate processing time
            elapsed_time = time.time() - start_time_video
            minutes, seconds = divmod(elapsed_time, 60)
            
            if video_comments:
                all_comments.extend(video_comments)
                log_message(
                    f"Video {i+1}/{len(urls)}: Berhasil mengumpulkan {len(video_comments)} "
                    f"item dalam {int(minutes)}m {int(seconds)}s", 
                    "SUCCESS"
                )
                success_count += 1
            else:
                log_message(f"Video {i+1}/{len(urls)}: Tidak ada komentar dikumpulkan", "WARNING")
                fail_count += 1
                
            # Add delay between videos to prevent rate limiting
            if i < len(urls) - 1:
                sleep_time = min(5, max(2, int(elapsed_time / 10)))  # Dynamic delay based on processing time
                log_message(f"Menunggu {sleep_time} detik sebelum memproses video berikutnya...", "INFO")
                time.sleep(sleep_time)
                
        except Exception as e:
            log_message(f"Error saat memproses video {url}: {str(e)}", "ERROR")
            fail_count += 1
    
    # Save all comments to CSV
    if all_comments:
        save_to_csv(all_comments, output_csv)
        
    # Print summary
    total_time = time.time() - start_time_total
    minutes, seconds = divmod(total_time, 60)
    hours, minutes = divmod(minutes, 60)
    
    log_message("\n" + "="*70, "INFO")
    log_message("RINGKASAN PENGUMPULAN KOMENTAR YOUTUBE", "INFO")
    log_message("="*70, "INFO")
    log_message(f"Total video yang diproses: {len(urls)}", "INFO")
    log_message(f"Video berhasil: {success_count}", "SUCCESS" if success_count > 0 else "INFO")
    log_message(f"Video gagal: {fail_count}", "ERROR" if fail_count > 0 else "INFO")
    log_message(f"Total komentar yang dikumpulkan: {len(all_comments)}", "INFO")
    
    if hours > 0:
        log_message(f"Total waktu eksekusi: {int(hours)}h {int(minutes)}m {int(seconds)}s", "INFO")
    else:
        log_message(f"Total waktu eksekusi: {int(minutes)}m {int(seconds)}s", "INFO")
    
    log_message(f"Hasil disimpan ke: {output_csv}", "INFO")
    log_message(f"Log disimpan ke: {log_file}", "INFO")
    
    return all_comments

def main():
    try:
        print("\n" + "="*70)
        print("YOUTUBE COMMENT SCRAPER - CSV EDITION")
        print("="*70)
        
        # Get output filename
        output_file = "youtube_comments(3).csv"
        
        # Maximum comments per video
        max_comments = 1000
        
        # Run the scraper with the URLs defined at the top
        scrape_multiple_videos(VIDEO_URLS, output_file, max_comments)
        
    except KeyboardInterrupt:
        log_message("\nProses dibatalkan oleh pengguna.", "WARNING")
    except Exception as e:
        log_message(f"Error tak terduga: {str(e)}", "ERROR")

if __name__ == "__main__":
    main()