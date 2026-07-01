#!/usr/bin/env python3
"""
Website image crawler/downloader.

Crawls one or more websites (using each site's sitemap.xml for page
discovery), finds all <img> and srcset images on each page, and downloads
them to a local folder for use as an image-generation training dataset.

Usage:
    python scrape_images.py

All key parameters are configurable in the CONFIGURATION section below.
Add one entry per site to SITES to crawl multiple sites in one run - each
site gets its own output subfolder, log file, and resume state.
"""

import hashlib
import io
import logging
import os
import sys
import time
from datetime import datetime
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# =============================================================================
# CONFIGURATION - edit these values as needed
# =============================================================================

# Sites to crawl - one entry per site. Each site gets its own output subfolder
# (named after "name") under BASE_OUTPUT_DIR, with its own log and resume state,
# so multiple sites (e.g. stc + its subsidiaries) can be crawled in one run
# without their images/logs mixing together.
#
# "sitemap_url" is optional - if omitted, it's auto-discovered from the site's
# robots.txt ("Sitemap:" line), falling back to "<base_url>/sitemap.xml" if
# robots.txt has none. Set it explicitly if a site uses a non-standard path
# (like stc's AEM-style sitemap below).
SITES = [
    {
        "name": "stc",
        "base_url": "https://www.stc.com.sa",
        "sitemap_url": "https://www.stc.com.sa/content/stc/sa.sitemap.xml",
    },
    {"name": "channels", "base_url": "https://channels.com.sa"},
    {"name": "solutions", "base_url": "https://solutions.com.sa"},
    {"name": "center3", "base_url": "https://center3.com"},
    {"name": "stcbank", "base_url": "https://stcbank.com.sa"},
    {"name": "sccc", "base_url": "https://sccc.sa"},
    {"name": "stcsc", "base_url": "https://www.stcsc.sa"},
    {"name": "sirar", "base_url": "https://www.sirar.com.sa"},
    {"name": "iotsquared", "base_url": "https://iotsquared.com.sa"},
    {"name": "aqalat", "base_url": "https://aqalat.com.sa"},
]

# Output
BASE_OUTPUT_DIR = r"C:\Users\jood1\Downloads\stc-images"
LOG_FILENAME = "scrape_log.txt"
PROCESSED_PAGES_FILENAME = "processed_pages.txt"  # resume support: pages fully handled in a prior run

# Image filtering thresholds
MIN_WIDTH = 200          # skip images narrower than this (pixels)
MIN_HEIGHT = 200         # skip images shorter than this (pixels)
MIN_FILE_SIZE_KB = 10    # skip images smaller than this (kilobytes)

# Timing / politeness
PAGE_DELAY_SECONDS = 1.0        # delay between page fetches
IMAGE_DELAY_SECONDS = 0.5       # delay between image downloads
REQUEST_TIMEOUT_SECONDS = 20    # timeout for any single HTTP request

# Retry logic
MAX_RETRIES = 3
RETRY_BACKOFF_BASE_SECONDS = 2  # 2s, 4s, 8s ...

# HTTP headers - identify as a normal browser
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "ar,en-US;q=0.9,en;q=0.8",
}

# Limit how many pages to process (None = no limit, crawl everything in the sitemap)
MAX_PAGES = None

# File extensions considered "images" when resolving <img src> / srcset
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".avif", ".svg")


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, LOG_FILENAME)

    logger = logging.getLogger("image_scraper")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger


# =============================================================================
# HTTP HELPERS (with retry/backoff)
# =============================================================================

def fetch_with_retries(session, url, logger, stream=False):
    """GET a URL with retry/backoff. Returns a Response or None on failure."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS, stream=stream)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (403, 404):
                logger.warning("HTTP %s for %s (not retrying)", resp.status_code, url)
                return None
            logger.warning("HTTP %s for %s (attempt %d/%d)", resp.status_code, url, attempt, MAX_RETRIES)
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("Request error for %s (attempt %d/%d): %s", url, attempt, MAX_RETRIES, exc)

        if attempt < MAX_RETRIES:
            backoff = RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            time.sleep(backoff)

    logger.error("Giving up on %s after %d attempts (%s)", url, MAX_RETRIES, last_exc)
    return None


# =============================================================================
# SITEMAP PARSING (handles sitemap index files with nested sitemaps)
# =============================================================================

SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def discover_sitemap_url(session, base_url, logger):
    """Look up a site's sitemap via robots.txt ("Sitemap:" line), falling
    back to "<base_url>/sitemap.xml" if robots.txt has none / is missing."""
    robots_url = urljoin(base_url, "/robots.txt")
    resp = fetch_with_retries(session, robots_url, logger)
    if resp is not None:
        for line in resp.text.splitlines():
            line = line.strip()
            if line.lower().startswith("sitemap:"):
                candidate = line.split(":", 1)[1].strip()
                if candidate:
                    logger.info("Discovered sitemap via robots.txt: %s", candidate)
                    return candidate

    fallback = urljoin(base_url, "/sitemap.xml")
    logger.info("No sitemap listed in robots.txt - falling back to %s", fallback)
    return fallback


def parse_sitemap(session, sitemap_url, logger, seen_sitemaps=None):
    """Recursively parse a sitemap or sitemap-index and return a list of page URLs."""
    if seen_sitemaps is None:
        seen_sitemaps = set()
    if sitemap_url in seen_sitemaps:
        return []
    seen_sitemaps.add(sitemap_url)

    logger.info("Fetching sitemap: %s", sitemap_url)
    resp = fetch_with_retries(session, sitemap_url, logger)
    if resp is None:
        logger.error("Failed to fetch sitemap: %s", sitemap_url)
        return []

    try:
        root = ElementTree.fromstring(resp.content)
    except ElementTree.ParseError as exc:
        logger.error("Failed to parse sitemap XML at %s: %s", sitemap_url, exc)
        return []

    tag = root.tag.lower()
    urls = []

    if tag.endswith("sitemapindex"):
        nested = [el.text.strip() for el in root.findall(".//sm:sitemap/sm:loc", SITEMAP_NS) if el.text]
        logger.info("Sitemap index with %d nested sitemap(s)", len(nested))
        for nested_url in nested:
            urls.extend(parse_sitemap(session, nested_url, logger, seen_sitemaps))
    elif tag.endswith("urlset"):
        page_urls = [el.text.strip() for el in root.findall(".//sm:url/sm:loc", SITEMAP_NS) if el.text]
        logger.info("Found %d page URL(s) in %s", len(page_urls), sitemap_url)
        urls.extend(page_urls)
    else:
        logger.warning("Unrecognized sitemap root tag '%s' at %s", root.tag, sitemap_url)

    return urls


# =============================================================================
# IMAGE DISCOVERY / URL RESOLUTION
# =============================================================================

def pick_best_srcset_url(srcset_value):
    """Given a srcset attribute value, return the URL with the highest resolution/width."""
    best_url = None
    best_score = -1.0
    for candidate in srcset_value.split(","):
        candidate = candidate.strip()
        if not candidate:
            continue
        parts = candidate.split()
        url = parts[0]
        score = 0.0
        if len(parts) > 1:
            descriptor = parts[1].strip().lower()
            try:
                if descriptor.endswith("w"):
                    score = float(descriptor[:-1])
                elif descriptor.endswith("x"):
                    score = float(descriptor[:-1]) * 1000  # normalize density vs width roughly
            except ValueError:
                score = 0.0
        if score >= best_score:
            best_score = score
            best_url = url
    return best_url


def looks_like_image_url(url):
    path = urlparse(url).path.lower()
    return path.endswith(IMAGE_EXTENSIONS) or "/image" in path or True  # keep permissive; size-filtered later


def extract_image_urls(html, page_url):
    """Parse HTML and return a set of absolute image URLs (best srcset candidate used)."""
    soup = BeautifulSoup(html, "html.parser")
    found = set()

    for img in soup.find_all("img"):
        candidates = []

        srcset = img.get("srcset") or img.get("data-srcset")
        if srcset:
            best = pick_best_srcset_url(srcset)
            if best:
                candidates.append(best)

        src = img.get("src") or img.get("data-src")
        if src:
            candidates.append(src)

        for c in candidates:
            if not c or c.startswith("data:"):
                continue
            absolute = urljoin(page_url, c.strip())
            found.add(absolute)

    # Also catch <source srcset="..."> inside <picture> elements
    for source in soup.find_all("source"):
        srcset = source.get("srcset")
        if srcset:
            best = pick_best_srcset_url(srcset)
            if best and not best.startswith("data:"):
                found.add(urljoin(page_url, best.strip()))

    return found


# =============================================================================
# IMAGE DOWNLOAD / FILTERING
# =============================================================================

def url_hash_filename(url, content_type=None):
    """Build a stable filename from the sha256 hash of the URL."""
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    ext = os.path.splitext(urlparse(url).path)[1].lower()
    if not ext or len(ext) > 5:
        ext_map = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/bmp": ".bmp",
            "image/svg+xml": ".svg",
            "image/avif": ".avif",
        }
        ext = ext_map.get((content_type or "").split(";")[0].strip().lower(), ".jpg")
    return f"{digest}{ext}"


def passes_size_filters(content_bytes, logger, url):
    """Return True if the image meets the minimum file-size and dimension thresholds."""
    size_kb = len(content_bytes) / 1024.0
    if size_kb < MIN_FILE_SIZE_KB:
        logger.info("SKIP (too small, %.1fKB < %dKB): %s", size_kb, MIN_FILE_SIZE_KB, url)
        return False

    if PIL_AVAILABLE:
        try:
            with Image.open(io.BytesIO(content_bytes)) as im:
                width, height = im.size
            if width < MIN_WIDTH or height < MIN_HEIGHT:
                logger.info("SKIP (too small dims, %dx%d): %s", width, height, url)
                return False
        except Exception:
            # Not a decodable raster image (e.g. SVG) - allow it through on size alone
            pass

    return True


def load_existing_filenames(output_dir):
    """Resume support: collect filenames already present in the output dir."""
    if not os.path.isdir(output_dir):
        return set()
    return set(os.listdir(output_dir))


def load_processed_pages(output_dir):
    """Resume support: pages that were fully processed (fetched + all images handled) in a prior run."""
    path = os.path.join(output_dir, PROCESSED_PAGES_FILENAME)
    if not os.path.isfile(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def mark_page_processed(output_dir, page_url):
    path = os.path.join(output_dir, PROCESSED_PAGES_FILENAME)
    with open(path, "a", encoding="utf-8") as f:
        f.write(page_url + "\n")


def download_image(session, image_url, output_dir, downloaded_urls, existing_filenames, logger):
    if image_url in downloaded_urls:
        return "duplicate"

    filename_guess = url_hash_filename(image_url)
    if filename_guess in existing_filenames:
        logger.info("SKIP (already on disk): %s", image_url)
        downloaded_urls.add(image_url)
        return "already_exists"

    resp = fetch_with_retries(session, image_url, logger, stream=True)
    if resp is None:
        logger.error("FAILED to download: %s", image_url)
        return "failed"

    try:
        content = resp.content
    except requests.RequestException as exc:
        logger.error("FAILED reading content for %s: %s", image_url, exc)
        return "failed"

    if not passes_size_filters(content, logger, image_url):
        downloaded_urls.add(image_url)
        return "skipped_small"

    content_type = resp.headers.get("Content-Type", "")
    filename = url_hash_filename(image_url, content_type)
    filepath = os.path.join(output_dir, filename)

    try:
        with open(filepath, "wb") as f:
            f.write(content)
    except OSError as exc:
        logger.error("FAILED writing file for %s: %s", image_url, exc)
        return "failed"

    existing_filenames.add(filename)
    downloaded_urls.add(image_url)
    logger.info("DOWNLOADED (%.1fKB): %s -> %s", len(content) / 1024.0, image_url, filename)
    return "downloaded"


# =============================================================================
# MAIN CRAWL LOOP
# =============================================================================

def crawl_site(session, site_name, base_url, sitemap_url, output_dir):
    """Crawl a single site end-to-end. Returns the stats dict for this site."""
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)

    logger.info("=== Starting crawl of '%s' at %s ===", site_name, datetime.now().isoformat())
    logger.info("Output dir: %s", output_dir)
    if not PIL_AVAILABLE:
        logger.warning("Pillow not installed - dimension filtering disabled (only file-size filter applied)")

    if not sitemap_url:
        sitemap_url = discover_sitemap_url(session, base_url, logger)
    logger.info("Sitemap: %s", sitemap_url)

    page_urls = parse_sitemap(session, sitemap_url, logger)
    page_urls = sorted(set(page_urls))
    if MAX_PAGES:
        page_urls = page_urls[:MAX_PAGES]

    logger.info("Total pages to crawl: %d", len(page_urls))

    existing_filenames = load_existing_filenames(output_dir)
    processed_pages = load_processed_pages(output_dir)
    downloaded_urls = set()

    if processed_pages:
        logger.info("Resume: %d page(s) already fully processed in a prior run - will be skipped", len(processed_pages))

    stats = {
        "pages_ok": 0,
        "pages_failed": 0,
        "pages_skipped": 0,
        "images_found": 0,
        "images_downloaded": 0,
        "images_skipped_small": 0,
        "images_skipped_duplicate": 0,
        "images_already_existing": 0,
        "images_failed": 0,
    }

    # Results that required no network request - don't count toward image download delay
    NO_REQUEST_RESULTS = ("duplicate", "already_exists")

    for i, page_url in enumerate(page_urls, start=1):
        if page_url in processed_pages:
            stats["pages_skipped"] += 1
            continue

        logger.info("[%d/%d] Processing page: %s", i, len(page_urls), page_url)

        resp = fetch_with_retries(session, page_url, logger)
        if resp is None:
            stats["pages_failed"] += 1
            time.sleep(PAGE_DELAY_SECONDS)
            continue

        stats["pages_ok"] += 1
        image_urls = extract_image_urls(resp.text, page_url)
        logger.info("Found %d image(s) on page", len(image_urls))
        stats["images_found"] += len(image_urls)

        for image_url in sorted(image_urls):
            result = download_image(session, image_url, output_dir, downloaded_urls, existing_filenames, logger)
            if result == "downloaded":
                stats["images_downloaded"] += 1
            elif result == "skipped_small":
                stats["images_skipped_small"] += 1
            elif result == "duplicate":
                stats["images_skipped_duplicate"] += 1
            elif result == "already_exists":
                stats["images_already_existing"] += 1
            elif result == "failed":
                stats["images_failed"] += 1
            if result not in NO_REQUEST_RESULTS:
                time.sleep(IMAGE_DELAY_SECONDS)

        mark_page_processed(output_dir, page_url)
        processed_pages.add(page_url)
        time.sleep(PAGE_DELAY_SECONDS)

    logger.info("=== Crawl of '%s' finished at %s ===", site_name, datetime.now().isoformat())
    logger.info(
        "Pages OK: %d | Pages failed: %d | Pages skipped (resumed): %d | Images found: %d | Downloaded: %d | "
        "Skipped(small): %d | Skipped(dup): %d | Already on disk: %d | Failed: %d",
        stats["pages_ok"], stats["pages_failed"], stats["pages_skipped"], stats["images_found"],
        stats["images_downloaded"], stats["images_skipped_small"],
        stats["images_skipped_duplicate"], stats["images_already_existing"],
        stats["images_failed"],
    )
    return stats


def main():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    session = requests.Session()

    for site in SITES:
        output_dir = os.path.join(BASE_OUTPUT_DIR, site["name"])
        crawl_site(session, site["name"], site["base_url"], site.get("sitemap_url"), output_dir)


if __name__ == "__main__":
    main()
