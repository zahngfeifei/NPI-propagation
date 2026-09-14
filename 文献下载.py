import csv
import os
import time
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup

# ====== 配置 ======
CSV_PATH = r"F:\NPI-4-code\讨论\第三次审核\参考文献doi\citation-placeholder-doi-map-dedup.csv"
OUTPUT_DIR = r"F:\NPI-4-code\讨论\第三次审核\参考文献"
FAILED_LOG = "failed_dois.txt"
REQUEST_TIMEOUT = 45
RETRY_TIMES = 2

# 全局速率限制（每秒最多请求数，可根据实际调整）
MAX_REQUESTS_PER_SECOND = 5
# 并发线程数（建议不超过 5~8，避免本地资源耗尽）
MAX_WORKERS = 5

# 镜像站列表（按优先级排序，可自行增删）
MIRRORS = [
    "https://www.sci-hub.shop",
    "https://www.sci-hub.ee",
    "https://www.sci-hub.vg",
    "https://sci-hub.st",
    "https://sci-hub.ru",
    "https://sci-hub.se",
    "https://sci-hub.ee",
    "https://sci-hub.la",
    "https://sci-hub.tw",
    "https://sci-hub.ren",
    "https://sci-hub.mk",
    "https://sci-hub.wf",
    "https://sci-hub.ec",
    "https://sci-hub.yt",
    "https://sci-hub.hk",
    "https://sci-hub.li",
    "https://sci-hub.nz",
]
# ==================

# ---------- 速率限制器 ----------
class RateLimiter:
    def __init__(self, max_per_second):
        self.max_per_second = max_per_second
        self.interval = 1.0 / max_per_second
        self.last_time = 0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_time
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_time = time.time()

# 全局速率限制实例
rate_limiter = RateLimiter(MAX_REQUESTS_PER_SECOND)

# ---------- 辅助函数 ----------
def sanitize_filename(name):
    """清理Windows文件名中的非法字符"""
    illegal_chars = r'[\\/:*?"<>|]'
    return re.sub(illegal_chars, '_', name)

def get_doi_list_with_filenames(csv_path):
    """
    返回任务列表，每个任务为 (doi, placeholder, filename)
    其中 filename 已包含 .pdf 后缀（不含路径），且对重复 placeholder 自动编号
    """
    # 第一步：读取所有 (doi, placeholder) 对
    pairs = []
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                placeholder = row.get('citationPlaceholder', '').strip()
                if not placeholder:
                    continue
                field = row.get('doiOrIdentifier', '').strip()
                if not field or 'no DOI' in field.lower():
                    continue
                dois = [d.strip() for d in field.split(';') if d.strip()]
                for doi in dois:
                    pairs.append((doi, placeholder))
    except Exception as e:
        print(f"读取CSV出错: {e}")
        return []

    # 第二步：统计每个 placeholder 出现的次数，分配序号
    placeholder_count = {}
    result = []
    for doi, placeholder in pairs:
        placeholder_count[placeholder] = placeholder_count.get(placeholder, 0) + 1
        serial = placeholder_count[placeholder]
        if serial == 1:
            base_name = placeholder
        else:
            base_name = f"{placeholder}_{serial}"
        filename = sanitize_filename(base_name) + '.pdf'
        result.append((doi, placeholder, filename))
    return result

def extract_pdf_link(html, base_url):
    """从HTML中提取PDF链接（同原逻辑）"""
    soup = BeautifulSoup(html, 'html.parser')
    candidates = []
    for iframe in soup.find_all('iframe'):
        src = iframe.get('src')
        if src:
            candidates.append(src)
    for embed in soup.find_all('embed'):
        src = embed.get('src')
        if src:
            candidates.append(src)
    for obj in soup.find_all('object'):
        data = obj.get('data')
        if data:
            candidates.append(data)
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.endswith('.pdf') or 'download=true' in href:
            candidates.append(href)
    for meta in soup.find_all('meta'):
        if meta.get('http-equiv', '').lower() == 'refresh':
            content = meta.get('content', '')
            match = re.search(r'url=(.+)', content, re.I)
            if match:
                candidates.append(match.group(1).strip())
    for script in soup.find_all('script'):
        if script.string:
            text = script.string
            matches = re.findall(r"(?:location\.href|window\.open)\s*=\s*['\"]([^'\"]+\.pdf[^'\"]*)['\"]", text)
            candidates.extend(matches)

    for link in candidates:
        if not link:
            continue
        if link.startswith('//'):
            link = 'https:' + link
        elif link.startswith('/'):
            link = base_url.rstrip('/') + link
        elif not link.startswith('http'):
            link = base_url.rstrip('/') + '/' + link.lstrip('/')
        if '.pdf' in link.lower() or 'download=true' in link:
            return link
    return None

def download_one(doi, placeholder, filename, output_dir, mirror_list, retries=2):
    """
    下载单个DOI（线程安全），受全局速率限制器控制
    返回 (成功标志, 错误信息, 使用的镜像, doi)
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    filepath = os.path.join(output_dir, filename)

    # 如果文件已存在，直接跳过（无需请求）
    if os.path.exists(filepath):
        return True, "文件已存在", None, doi

    # 使用 Session 复用连接
    session = requests.Session()
    session.headers.update(headers)

    for mirror in mirror_list:
        url = f"{mirror}/{doi}"
        for attempt in range(1, retries + 1):
            # 每次真实请求前都等待速率限制
            rate_limiter.wait()
            try:
                resp = session.get(url, timeout=REQUEST_TIMEOUT)
                if resp.status_code != 200:
                    continue

                # 直接返回PDF
                content_type = resp.headers.get('Content-Type', '').lower()
                if content_type.startswith('application/pdf') or resp.content.startswith(b'%PDF'):
                    with open(filepath, 'wb') as f:
                        f.write(resp.content)
                    return True, None, mirror, doi

                # 解析HTML获取PDF链接
                pdf_link = extract_pdf_link(resp.text, mirror)
                if pdf_link:
                    rate_limiter.wait()  # 获取PDF链接也需要请求
                    pdf_resp = session.get(pdf_link, timeout=REQUEST_TIMEOUT)
                    if pdf_resp.status_code == 200 and (pdf_resp.headers.get('Content-Type', '').lower().startswith('application/pdf') or pdf_resp.content.startswith(b'%PDF')):
                        with open(filepath, 'wb') as f:
                            f.write(pdf_resp.content)
                        return True, None, mirror, doi
                    else:
                        continue
                else:
                    continue
            except Exception:
                continue
            time.sleep(1)  # 重试前短暂等待

    return False, "所有镜像均无法获取PDF", None, doi

def main():
    print("=" * 60)
    print("Sci-Hub 批量下载（多线程 + 速率限制）")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 获取任务列表 (doi, placeholder, filename)
    tasks = get_doi_list_with_filenames(CSV_PATH)
    total = len(tasks)
    if total == 0:
        print("没有找到有效的DOI。")
        return

    print(f"共找到 {total} 个DOI条目。")
    print(f"并发线程数: {MAX_WORKERS}")
    print(f"每秒最大请求数: {MAX_REQUESTS_PER_SECOND}")
    print(f"镜像列表: {MIRRORS}\n")

    success_count = 0
    skipped_count = 0
    failed = []

    # 使用线程池并发下载
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(
                download_one,
                doi, placeholder, filename,
                OUTPUT_DIR, MIRRORS, RETRY_TIMES
            ): (doi, placeholder, filename)
            for doi, placeholder, filename in tasks
        }

        # 按完成顺序处理结果
        for idx, future in enumerate(as_completed(future_to_task), 1):
            doi, placeholder, filename = future_to_task[future]
            try:
                ok, err, used_mirror, _ = future.result()
                if ok:
                    if err == "文件已存在":
                        print(f"[{idx}/{total}] 文件已存在，跳过: {filename}")
                        skipped_count += 1
                    else:
                        print(f"[{idx}/{total}] ✓ 成功: {filename} (镜像: {used_mirror})")
                        success_count += 1
                else:
                    print(f"[{idx}/{total}] ✗ 失败: {filename} - {err}")
                    failed.append((doi, placeholder))
            except Exception as e:
                print(f"[{idx}/{total}] ✗ 异常: {filename} - {str(e)}")
                failed.append((doi, placeholder))

    print("\n" + "=" * 60)
    print(f"下载完成。成功: {success_count}, 跳过(已存在): {skipped_count}, 失败: {len(failed)}")
    if failed:
        with open(FAILED_LOG, 'w', encoding='utf-8') as f:
            for doi, placeholder in failed:
                f.write(f"{doi}\t{placeholder}\n")
        print(f"失败的DOI及对应占位符已保存至: {FAILED_LOG}")

if __name__ == "__main__":
    main()