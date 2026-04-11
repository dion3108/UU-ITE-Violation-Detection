import requests
import json
import time
import random
import csv
import browser_cookie3
import concurrent.futures
from datetime import datetime
import os
import signal
import sys
import threading
import atexit
import gc
from tqdm import tqdm
import datetime

# Global variables for tracking
active_urls = []
completed_urls = []
failed_urls = []
all_comments = []
last_save_time = time.time()
save_lock = threading.Lock()
exit_event = threading.Event()  # For clean shutdown
scrape_start_time = None
last_comments_count = 0
last_update_time = None
last_full_status = None

def get_tiktok_cookies():
    """Get TikTok cookies from the browser."""
    browsers = [browser_cookie3.chrome, browser_cookie3.firefox]
    for browser in browsers:
        try:
            return browser(domain_name='.tiktok.com')
        except:
            continue
    print("Warning: Could not get TikTok cookies from browsers.")
    return None

def create_session():
    """Create and configure a requests session with connection pooling."""
    session = requests.Session()
    session.cookies = get_tiktok_cookies()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36',
        'Referer': 'https://www.tiktok.com/',
        'Origin': 'https://www.tiktok.com',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept': 'application/json, text/plain, */*'
    })
    return session

def req(post_id, curs, session):
    """API request with retries and connection pooling."""
    url = f'https://www.tiktok.com/api/comment/list/?aid=1988&aweme_id={post_id}&count=20&cursor={curs}'
    
    for attempt in range(3):
        try:
            response = session.get(url, timeout=15)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print("Rate limited, waiting...")
                time.sleep(10)
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            if attempt == 2:
                print(f"Final request failure: {str(e)[:100]}")
                return {"comments": [], "has_more": 0}
            time.sleep(1 + attempt*2)
    return {"comments": [], "has_more": 0}

def fetch_replies(comment_id, post_id, reply_cursor, session):
    """Fetch replies with session reuse."""
    url = f'https://www.tiktok.com/api/comment/list/reply/?aid=1988&comment_id={comment_id}&count=20&cursor={reply_cursor}&item_id={post_id}'
    try:
        return session.get(url, timeout=15).json()
    except:
        return {"comments": [], "has_more": 0}

def get_all_replies(comment_id, post_id, session):
    """Fetch replies with rate limiting."""
    reply_cursor = 0
    replies = []
    for _ in range(5):  # Max 5 pages
        if exit_event.is_set():
            return []
        data = fetch_replies(comment_id, post_id, reply_cursor, session)
        replies.extend(data.get('comments', []))
        if not data.get('has_more', 0):
            break
        reply_cursor += 20
        time.sleep(0.3)
    return replies[:200]  # Safety limit

def process_comment(comment, post_url, session):
    """Process a single comment and its replies."""
    comment_data = {
        'komentar_id': comment.get('cid', ''),
        'Username': comment.get('user', {}).get('unique_id', ''),
        'T_komentar': comment.get('text', ''),
        'link': post_url
    }
    
    replies = []
    if comment.get('reply_comment_total', 0) > 0:
        replies = get_all_replies(comment['cid'], comment['aweme_id'], session)
    
    return [comment_data] + [
        {
            'komentar_id': r.get('cid', ''),
            'Username': r.get('user', {}).get('unique_id', ''),
            'T_komentar': r.get('text', ''),
            'link': post_url
        } for r in replies
    ]

def scrape_single_tiktok(url, session):
    """Scrape a single TikTok post with improved cursor handling and retry logic."""
    global active_urls, completed_urls, failed_urls

    try:
        with save_lock:
            active_urls.append(url)
            update_progress_display()

        post_id = url.split('/')[-1].split('?')[0]
        comments = []
        curs = 0
        consecutive_empty = 0

        for page in range(200):  # Increased max pages
            if exit_event.is_set():
                return []

            data = req(post_id, curs, session)

            batch_comments = data.get('comments', [])

            if not batch_comments:
                consecutive_empty += 1
                if consecutive_empty >= 5:
                    print("⚠️ Too many empty comment pages, stopping early.")
                    break
                time.sleep(1)
                continue
            else:
                consecutive_empty = 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_comment, c, url, session) for c in batch_comments]
                for future in concurrent.futures.as_completed(futures):
                    comments.extend(future.result())

            print(f"\nProcessed {len(batch_comments)} comments on page {page+1} for {url.split('/')[-2]}")

            # Use cursor from response if available
            if data.get('has_more', 0):
                curs = data.get('cursor', curs + 20)
                if not curs:
                    print("⚠️ No cursor provided in response; stopping.")
                    break
            else:
                break

            time.sleep(0.5 + random.random())

        with save_lock:
            all_comments.extend(comments)
            active_urls.remove(url)
            completed_urls.append(url)
            update_progress_display()

        print(f"✅ Done scraping {url}. Total collected: {len(comments)}")
        return comments

    except Exception as e:
        with save_lock:
            if url in active_urls:
                active_urls.remove(url)
            failed_urls.append(url)
            update_progress_display()
        print(f"❌ Error processing {url}: {str(e)[:200]}")
        return []

def save_to_csv(comments, filename):
    """Thread-safe CSV saving with batch writing."""
    with save_lock:
        try:
            file_exists = os.path.isfile(filename)
            with open(filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['komentar_id', 'Username', 'T_komentar', 'link'])
                if not file_exists:
                    writer.writeheader()
                writer.writerows(comments)
            return len(comments)
        except Exception as e:
            print(f"Save error: {str(e)[:200]}")
            return 0

def update_progress_display():
    """Enhanced progress display with detailed statistics."""
    global scrape_start_time, last_comments_count, last_update_time, last_full_status
    
    current_time = time.time()
    elapsed_time = current_time - scrape_start_time
    current_comments = len(all_comments)
    
    # Calculate speed (comments per second)
    if current_time - last_update_time >= 5:  # Update stats every 5 seconds
        delta_comments = current_comments - last_comments_count
        delta_time = current_time - last_update_time
        comments_per_second = delta_comments / delta_time if delta_time > 0 else 0
        
        # Save current values for next calculation
        last_comments_count = current_comments
        last_update_time = current_time
    else:
        comments_per_second = current_comments / elapsed_time if elapsed_time > 0 else 0
    
    # Calculate estimated time remaining
    total_urls = len(active_urls) + len(completed_urls) + len(failed_urls)
    if completed_urls:
        avg_comments_per_url = current_comments / len(completed_urls)
        remaining_urls = len(active_urls)
        estimated_remaining_comments = remaining_urls * avg_comments_per_url
        
        if comments_per_second > 0:
            est_seconds_remaining = estimated_remaining_comments / comments_per_second
            eta = str(datetime.timedelta(seconds=int(est_seconds_remaining)))
        else:
            eta = "Unknown"
    else:
        eta = "Calculating..."
        avg_comments_per_url = 0
    
    # Format elapsed time
    elapsed = str(datetime.timedelta(seconds=int(elapsed_time)))
    
    # Calculate percentage complete
    if total_urls > 0:
        percent_complete = (len(completed_urls) + len(failed_urls)) / total_urls * 100
    else:
        percent_complete = 0
        
    # Progress bar (width of 30 characters)
    bar_width = 30
    filled_length = int(bar_width * percent_complete / 100)
    bar = '█' * filled_length + '░' * (bar_width - filled_length)
    
    # Clear previous output
    terminal_width = os.get_terminal_size().columns
    sys.stdout.write('\r' + ' ' * terminal_width)
    
    # Detailed status line
    status = f"\r[{bar}] {percent_complete:.1f}% | "
    status += f"URLs: {len(completed_urls)}/{total_urls} | "
    status += f"Active: {len(active_urls)} | "
    status += f"Comments: {current_comments} | "
    status += f"Speed: {comments_per_second:.1f} comments/sec | "
    status += f"Elapsed: {elapsed} | ETA: {eta}"
    
    # Ensure it fits in the terminal
    if len(status) > terminal_width:
        status = status[:terminal_width-3] + "..."
        
    sys.stdout.write(status)
    sys.stdout.flush()
    
    # Every minute, print a full status update on a new line
    if last_full_status is None or current_time - last_full_status >= 60:
        sys.stdout.write('\n')
        print(f"\n{'='*50}")
        print(f"📊 DETAILED STATUS REPORT - {datetime.datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*50}")
        print(f"✅ Completed URLs: {len(completed_urls)}/{total_urls} ({percent_complete:.1f}%)")
        print(f"⏳ Active URLs: {len(active_urls)}")
        print(f"❌ Failed URLs: {len(failed_urls)}")
        print(f"💬 Total comments: {current_comments}")
        print(f"📈 Average comments per URL: {avg_comments_per_url:.1f}")
        print(f"⏱️ Elapsed time: {elapsed}")
        print(f"🚀 Processing speed: {comments_per_second:.2f} comments/sec")
        print(f"🔮 Estimated completion: {eta}")
        if active_urls:
            print(f"🔄 Currently processing: {', '.join(url.split('/')[-2] for url in active_urls[:3])}" + 
                  (f" and {len(active_urls)-3} more..." if len(active_urls) > 3 else ""))
        print(f"{'='*50}\n")
        
        last_full_status = current_time

def periodic_save():
    """Daemonized periodic saving."""
    while not exit_event.is_set():
        time.sleep(300)  # Save every 5 minutes
        if all_comments:
            backup_file = f"tiktok_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            num_saved = save_to_csv(all_comments, backup_file)
            print(f"\n💾 Auto-saved {num_saved} comments to {backup_file}")
            update_progress_display()  # Refresh progress display after message

def signal_handler(sig, frame):
    """Unified signal handling."""
    print("\nExiting gracefully...")
    exit_event.set()
    save_to_csv(all_comments, 'final_save.csv')
    sys.exit(0)

def scrape_multiple_tiktoks(urls, max_workers=5):
    """Main scraping function with resource management."""
    global scrape_start_time, last_comments_count, last_update_time, last_full_status
    scrape_start_time = time.time()
    last_comments_count = 0
    last_update_time = scrape_start_time
    last_full_status = None
    
    print(f"🚀 Starting TikTok scraper at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Processing {len(urls)} URLs with {max_workers} workers")
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start periodic save daemon
    saver = threading.Thread(target=periodic_save, daemon=True)
    saver.start()

    with create_session() as session:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(scrape_single_tiktok, url, session): url for url in urls}
            
            for future in concurrent.futures.as_completed(futures):
                url = futures[future]
                try:
                    future.result(timeout=600)
                except Exception as e:
                    print(f"Task failed: {str(e)[:200]}")

    # Calculate final statistics
    elapsed_time = time.time() - scrape_start_time
    elapsed = str(datetime.timedelta(seconds=int(elapsed_time)))
    
    print("\n\n" + "="*60)
    print("🏁 SCRAPING COMPLETED")
    print("="*60)
    print(f"📊 Total URLs processed: {len(urls)}")
    print(f"✅ Successfully scraped: {len(completed_urls)}")
    print(f"❌ Failed: {len(failed_urls)}")
    print(f"💬 Total comments collected: {len(all_comments)}")
    print(f"⏱️ Total time: {elapsed}")
    print(f"🚀 Average speed: {len(all_comments)/elapsed_time:.2f} comments/sec")
    
    if completed_urls:
        print(f"📈 Average comments per URL: {len(all_comments)/len(completed_urls):.1f}")
    
    print(f"💾 Results saved to: tes(6).csv")
    print("="*60)
    
    save_to_csv(all_comments, 'tes(6).csv')
    
    # Ensure clean exit to prevent hanging
    os._exit(0)

if __name__ == "__main__":
    tiktok_urls = [
        # Your list of URLs here
        # 'https://www.tiktok.com/@deddyyevrisitorus/video/7513889467913030920',
        # 'https://www.tiktok.com/@ailifeindo/video/7509779259787873543?',
        # 'https://www.tiktok.com/@user552499223/video/7511585891912125701',
        # 'https://www.tiktok.com/@loveyu715/photo/7508044584828538129',
        # 'https://www.tiktok.com/@maimun.zammi3/video/7515477073335520518',
        # 'https://www.tiktok.com/@annabell_19808/video/7513151926486322439',
        # 'https://www.tiktok.com/@idealis92/video/7514518109160492344',
        # 'https://www.tiktok.com/@rararuru887/video/7515013985830243602',
        # Add more URLs as needed
        # 'https://www.tiktok.com/@suaramulut26/video/7515403728158199045',
        # 'https://www.tiktok.com/@dndaatasyaa/photo/7517769805189876999',
        # 'https://www.tiktok.com/@dcrmamad/video/7519549792096226568',
        # 'https://www.tiktok.com/@cliprandom739/video/7521360932669230343',
        # 'https://www.tiktok.com/@kucingpintarid/video/7521978709851606279',
        # 'https://www.tiktok.com/@tejusapelrasamanggis2/video/7521711034827099398',
        # 'https://www.tiktok.com/@penyu_laut87/video/7513153992776617223',
        # 'https://www.tiktok.com/@medanmediainfo/video/7524266043569605895',
        # 'https://www.tiktok.com/@sofyan.tok/video/7522090917256334598',
        # 'https://www.tiktok.com/@perempatan.udin/video/7519451201256148242',
        # 'https://www.tiktok.com/@ummu_gahania17/video/7521597684604570887',
        # 'https://www.tiktok.com/@wahyuj55/video/7505970746926042423',
        # 'https://www.tiktok.com/@raaciil/video/7314278813209316614',
        # 'https://www.tiktok.com/@raaciil/video/7149495954495016218',
        # 'https://www.tiktok.com/@nrlllfdlhh/video/7468923249213066502',
        # 'https://www.tiktok.com/@ehanke.2/photo/7522365212272446776',
        # 'https://www.tiktok.com/@sabarglowup/video/7475217381074291974',
        # 'https://www.tiktok.com/@eyesfrkaptrina/photo/7521299846456708358',
        # 'https://www.tiktok.com/@udzrieaulia/video/7467920197777706246',
        # 'https://www.tiktok.com/@shasaurus/video/7488314355650546952',
        # 'https://www.tiktok.com/@shez.aaa/video/7519843380457786632',
        # 'https://www.tiktok.com/@namasayagwen/video/7477534377413578002',
        'https://www.tiktok.com/@udzrieaulia/video/7467920197777706246'
    ]
    
    try:
        scrape_multiple_tiktoks(tiktok_urls, max_workers=7)
    except KeyboardInterrupt:
        print("\nUser interrupted. Saving data and exiting...")
        save_to_csv(all_comments, 'interrupted_save.csv')
        os._exit(0)
