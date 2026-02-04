"""
Bandcamp Creative Commons Electronic Music Auto-Downloader

Automatically finds and downloads CC-licensed electronic music from Bandcamp.

Requirements:
    pip install requests beautifulsoup4 selenium webdriver-manager

Usage:
    # Auto-find and download CC + electronic albums
    python scripts/download_bandcamp_cc.py --auto --limit 50

    # Specific tags combination
    python scripts/download_bandcamp_cc.py --auto --tags "creative-commons,electronic" --limit 100

    # Just list albums without downloading
    python scripts/download_bandcamp_cc.py --auto --tags "creative-commons,ambient" --list-only

    # Manual: single album URL
    python scripts/download_bandcamp_cc.py --url "https://artist.bandcamp.com/album/name"
"""

import os
import sys
import argparse
import json
import time
import re
import html
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict, field
from urllib.parse import urljoin, urlparse

sys.path.append(str(Path(__file__).parent.parent))

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Required packages not found. Install with:")
    print("  pip install requests beautifulsoup4")
    sys.exit(1)


@dataclass
class TrackInfo:
    """Track metadata"""
    title: str
    url: str
    duration: float = 0
    track_num: int = 0


@dataclass
class AlbumInfo:
    """Album metadata"""
    url: str
    title: str = ""
    artist: str = ""
    tags: List[str] = field(default_factory=list)
    is_free: bool = False
    license: str = ""
    tracks: List[TrackInfo] = field(default_factory=list)


class BandcampCrawler:
    """
    Crawls Bandcamp tag pages using Selenium to find albums.
    """

    def __init__(self, headless: bool = True):
        self.headless = headless
        self.driver = None

    def _init_driver(self):
        """Initialize Selenium WebDriver"""
        if self.driver is not None:
            return

        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.chrome.options import Options
            from webdriver_manager.chrome import ChromeDriverManager

            options = Options()
            if self.headless:
                options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")

            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
            print("  WebDriver initialized")

        except Exception as e:
            print(f"  Failed to initialize WebDriver: {e}")
            print("  Make sure Chrome is installed")
            raise

    def close(self):
        """Close WebDriver"""
        if self.driver:
            self.driver.quit()
            self.driver = None

    def search_tags(
        self,
        tags: List[str],
        max_albums: int = 50,
        scroll_pause: float = 2.0,
    ) -> List[str]:
        """
        Search Bandcamp for albums with specific tags.

        Args:
            tags: List of tags (e.g., ["creative-commons", "electronic"])
            max_albums: Maximum number of album URLs to collect
            scroll_pause: Pause between scrolls (seconds)

        Returns:
            List of album URLs
        """
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        self._init_driver()

        # Bandcamp supports multiple tags with + separator
        # e.g., https://bandcamp.com/tag/electronic+creative-commons
        combined_tags = "+".join(tags)
        url = f"https://bandcamp.com/tag/{combined_tags}"

        print(f"  Opening: {url}")
        self.driver.get(url)
        time.sleep(3)

        album_urls = set()
        last_count = 0
        no_change_count = 0

        print(f"  Scrolling to load albums (target: {max_albums})...")

        while len(album_urls) < max_albums:
            # Find album links - use simple approach: all <a> tags with /album/ or /track/
            try:
                all_links = self.driver.find_elements(By.TAG_NAME, "a")

                for link in all_links:
                    href = link.get_attribute("href") or ""
                    if "/album/" in href or "/track/" in href:
                        # Filter: must be bandcamp.com domain
                        if "bandcamp.com" in href:
                            # Remove query params for cleaner URLs
                            clean_url = href.split("?")[0]
                            album_urls.add(clean_url)

            except Exception as e:
                print(f"    Error finding items: {e}")

            print(f"    Found {len(album_urls)} albums...", end="\r")

            # Check if we found enough
            if len(album_urls) >= max_albums:
                break

            # Check if count changed
            if len(album_urls) == last_count:
                no_change_count += 1
                if no_change_count >= 5:
                    print(f"\n    No more albums loading, stopping at {len(album_urls)}")
                    break
            else:
                no_change_count = 0
                last_count = len(album_urls)

            # Scroll down
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_pause)

            # Try clicking "load more" if exists
            try:
                load_more = self.driver.find_element(By.CSS_SELECTOR, "button.load-more, .show-more")
                if load_more.is_displayed():
                    load_more.click()
                    time.sleep(scroll_pause)
            except:
                pass

        print(f"\n  Collected {len(album_urls)} album URLs")

        return list(album_urls)[:max_albums]


class BandcampDownloader:
    """
    Downloads free music from Bandcamp album pages.
    """

    CC_PATTERNS = [
        r'creativecommons\.org/licenses',
        r'creative\s*commons',
        r'CC[\s-]BY',
        r'CC[\s-]0',
        r'CC0',
        r'public\s*domain',
    ]

    def __init__(self, output_dir: str, delay: float = 0.5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/120.0.0.0 Safari/537.36'
        })

    def parse_album_page(self, album_url: str) -> Optional[AlbumInfo]:
        """Parse album page for metadata and track info."""
        try:
            time.sleep(self.delay)
            resp = self.session.get(album_url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            page_text = resp.text

            info = AlbumInfo(url=album_url)

            # Get title
            title_tag = soup.find('h2', class_='trackTitle')
            if title_tag:
                info.title = title_tag.get_text(strip=True)
            else:
                og_title = soup.find('meta', property='og:title')
                if og_title:
                    info.title = og_title.get('content', '')

            # Get artist
            artist_tag = soup.find('span', itemprop='byArtist')
            if artist_tag:
                a_tag = artist_tag.find('a')
                if a_tag:
                    info.artist = a_tag.get_text(strip=True)
                else:
                    info.artist = artist_tag.get_text(strip=True)

            # Try from URL
            if not info.artist:
                parsed = urlparse(album_url)
                if parsed.netloc.endswith('.bandcamp.com'):
                    info.artist = parsed.netloc.replace('.bandcamp.com', '')

            # Get tags
            tag_elements = soup.find_all('a', class_='tag')
            info.tags = [t.get_text(strip=True).lower() for t in tag_elements]

            # Check for CC license
            for pattern in self.CC_PATTERNS:
                if re.search(pattern, page_text, re.IGNORECASE):
                    info.license = "creative-commons"
                    break

            # Check license div
            license_div = soup.find('div', id='license')
            if license_div:
                license_text = license_div.get_text(strip=True)
                info.license = license_text[:100]
                if 'creative commons' in license_text.lower():
                    info.license = "creative-commons"

            # Check if free
            for script in soup.find_all('script'):
                script_text = script.string or ''
                if 'TralbumData' in script_text or 'trackinfo' in script_text:
                    if re.search(r'"minimum_price"\s*:\s*0[,\}]', script_text):
                        info.is_free = True
                    if re.search(r'"minimum_price_nonzero"\s*:\s*false', script_text):
                        info.is_free = True

            # Extract tracks from HTML-decoded content
            decoded_html = html.unescape(page_text)
            match = re.search(r'trackinfo["\s]*:\s*(\[.*?\])\s*[,}]', decoded_html, re.DOTALL)
            if match:
                try:
                    tracks_data = json.loads(match.group(1))
                    for i, track in enumerate(tracks_data):
                        mp3_url = None
                        if track.get('file'):
                            mp3_url = track['file'].get('mp3-128')

                        if mp3_url:
                            mp3_url = html.unescape(mp3_url)
                            info.tracks.append(TrackInfo(
                                title=track.get('title', f'Track {i+1}'),
                                url=mp3_url,
                                duration=track.get('duration', 0),
                                track_num=i + 1,
                            ))
                except json.JSONDecodeError:
                    pass

            if 'free download' in page_text.lower():
                info.is_free = True

            return info

        except Exception as e:
            print(f"    Error parsing {album_url}: {e}")
            return None

    def check_tags_match(self, info: AlbumInfo, required_tags: Set[str]) -> bool:
        """Check if album has all required tags"""
        # Normalize album tags (lowercase, keep both hyphen and space versions)
        album_tags_normalized = set()
        for tag in info.tags:
            tag_lower = tag.lower().strip()
            album_tags_normalized.add(tag_lower)
            album_tags_normalized.add(tag_lower.replace(' ', '-'))
            album_tags_normalized.add(tag_lower.replace('-', ' '))

        # Check each required tag
        for req_tag in required_tags:
            req_lower = req_tag.lower().strip()
            req_variations = {
                req_lower,
                req_lower.replace('-', ' '),
                req_lower.replace(' ', '-'),
            }
            if not req_variations.intersection(album_tags_normalized):
                return False
        return True

    def is_cc_licensed(self, info: AlbumInfo, page_text: str = "") -> bool:
        """Check if album has CC license (from tags, license field, or page content)"""
        # Check tags
        cc_tag_patterns = ['creative commons', 'creative-commons', 'cc-by', 'cc0', 'public domain']
        for tag in info.tags:
            tag_lower = tag.lower()
            for pattern in cc_tag_patterns:
                if pattern in tag_lower:
                    return True

        # Check license field
        if info.license:
            license_lower = info.license.lower()
            if 'creative commons' in license_lower or 'cc-by' in license_lower or 'cc0' in license_lower:
                return True

        return False

    def get_album_dir(self, info: AlbumInfo) -> Path:
        """Get album directory path"""
        safe_artist = re.sub(r'[^\w\s-]', '', info.artist)[:40] or "Unknown"
        safe_title = re.sub(r'[^\w\s-]', '', info.title)[:40] or "Unknown"
        return self.output_dir / "audio" / f"{safe_artist} - {safe_title}"

    def is_album_downloaded(self, info: AlbumInfo) -> bool:
        """Check if album is already downloaded (has metadata.json)"""
        album_dir = self.get_album_dir(info)
        meta_path = album_dir / "metadata.json"
        return meta_path.exists()

    def download_track(self, track: TrackInfo, output_dir: Path) -> bool:
        """Download a single track"""
        safe_title = re.sub(r'[^\w\s-]', '', track.title)[:60]
        filename = f"{track.track_num:02d} - {safe_title}.mp3"
        output_path = output_dir / filename

        if output_path.exists():
            return True

        try:
            resp = self.session.get(track.url, stream=True, timeout=60)
            resp.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True

        except Exception as e:
            print(f"      Download error: {e}")
            return False

    def download_album(self, info: AlbumInfo, skip_existing: bool = True) -> bool:
        """Download all tracks from an album"""
        if not info.tracks:
            return False

        album_dir = self.get_album_dir(info)

        # Skip if already downloaded
        if skip_existing and self.is_album_downloaded(info):
            return True

        album_dir.mkdir(parents=True, exist_ok=True)

        success = 0
        for track in info.tracks:
            if self.download_track(track, album_dir):
                success += 1
            time.sleep(self.delay)

        # Save metadata
        meta_path = album_dir / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(asdict(info), f, indent=2)

        return success > 0


def auto_download(
    tags: List[str],
    output_dir: str,
    limit: int = 50,
    download: bool = True,
    headless: bool = True,
) -> List[AlbumInfo]:
    """
    Automatically find and download CC electronic albums.

    Args:
        tags: Tags to search for (e.g., ["creative-commons", "electronic"])
        output_dir: Output directory
        limit: Max albums to download
        download: Whether to download (False = list only)
        headless: Run browser in headless mode

    Returns:
        List of found/downloaded albums
    """
    print(f"\n{'='*50}")
    print(f"Bandcamp Auto-Downloader")
    print(f"Tags: {', '.join(tags)}")
    print(f"Limit: {limit} albums")
    print(f"{'='*50}\n")

    # Initialize crawler and downloader
    crawler = BandcampCrawler(headless=headless)
    downloader = BandcampDownloader(output_dir=output_dir)

    required_tags = set(t.lower() for t in tags)
    found_albums = []
    downloaded_albums = []

    try:
        # Step 1: Find album URLs
        print("[Step 1] Searching Bandcamp...")
        album_urls = crawler.search_tags(tags, max_albums=limit * 3)  # Get extra to filter

        if not album_urls:
            print("No albums found!")
            return []

        # Step 2: Check each album
        print(f"\n[Step 2] Checking {len(album_urls)} albums for tags + free/CC...")

        for i, url in enumerate(album_urls):
            if len(found_albums) >= limit:
                break

            print(f"  [{i+1}/{len(album_urls)}] {url[:60]}...", end=" ", flush=True)

            info = downloader.parse_album_page(url)
            if not info:
                print("✗ parse failed")
                continue

            # Check criteria
            # Bandcamp already filters by tags in URL, but verify and check CC license
            has_required_tags = downloader.check_tags_match(info, required_tags)
            is_cc_or_free = info.is_free or downloader.is_cc_licensed(info)

            if has_required_tags and is_cc_or_free and info.tracks:
                status = []
                if info.is_free:
                    status.append("FREE")
                if downloader.is_cc_licensed(info):
                    status.append("CC")
                print(f"✓ [{', '.join(status)}] {len(info.tracks)} tracks")
                found_albums.append(info)
            else:
                reasons = []
                if not has_required_tags:
                    reasons.append(f"tags:{info.tags[:3]}")
                if not is_cc_or_free:
                    reasons.append("not free/CC")
                if not info.tracks:
                    reasons.append("no tracks")
                print(f"✗ {', '.join(reasons)}")

        print(f"\nFound {len(found_albums)} matching albums")

        # Step 3: Download
        if download and found_albums:
            print(f"\n[Step 3] Downloading {len(found_albums)} albums...")

            for i, info in enumerate(found_albums):
                print(f"\n  [{i+1}/{len(found_albums)}] {info.artist} - {info.title}")

                # Check if already downloaded
                if downloader.is_album_downloaded(info):
                    print(f"      ⏭ Already downloaded, skipping")
                    downloaded_albums.append(info)
                    continue

                print(f"      {len(info.tracks)} tracks")

                if downloader.download_album(info, skip_existing=False):
                    downloaded_albums.append(info)
                    print(f"      ✓ Downloaded")
                else:
                    print(f"      ✗ Failed")

    finally:
        crawler.close()

    # Save album list
    album_list_path = Path(output_dir) / "album_list.json"
    data = {
        'tags': tags,
        'total_found': len(found_albums),
        'total_downloaded': len(downloaded_albums),
        'albums': [asdict(a) for a in found_albums],
    }
    with open(album_list_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Complete!")
    print(f"  Found: {len(found_albums)} albums")
    print(f"  Downloaded: {len(downloaded_albums)} albums")
    print(f"  Output: {output_dir}")
    print(f"{'='*50}")

    return found_albums


def main():
    parser = argparse.ArgumentParser(
        description="Download CC/free electronic music from Bandcamp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--auto",
        action="store_true",
        help="Auto-find and download albums with specified tags",
    )
    mode_group.add_argument(
        "--url",
        type=str,
        help="Download single album from URL",
    )
    mode_group.add_argument(
        "--url-file",
        type=str,
        help="Download albums from URLs in file",
    )

    # Options
    parser.add_argument(
        "--tags",
        type=str,
        default="creative-commons,electronic",
        help="Comma-separated tags for --auto mode (default: creative-commons,electronic)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max albums to download (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/bandcamp_cc",
        help="Output directory",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list albums, don't download",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Show browser window (for debugging)",
    )

    args = parser.parse_args()

    if args.auto:
        tags = [t.strip() for t in args.tags.split(',')]
        auto_download(
            tags=tags,
            output_dir=args.output_dir,
            limit=args.limit,
            download=not args.list_only,
            headless=not args.no_headless,
        )

    elif args.url:
        downloader = BandcampDownloader(output_dir=args.output_dir)
        info = downloader.parse_album_page(args.url)
        if info:
            print(f"Title: {info.title}")
            print(f"Artist: {info.artist}")
            print(f"Tags: {', '.join(info.tags)}")
            print(f"Free: {info.is_free}, License: {info.license}")
            print(f"Tracks: {len(info.tracks)}")

            if not args.list_only and info.tracks:
                downloader.download_album(info)

    elif args.url_file:
        url_file = Path(args.url_file)
        if not url_file.exists():
            print(f"Error: File not found: {args.url_file}")
            sys.exit(1)

        with open(url_file) as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        downloader = BandcampDownloader(output_dir=args.output_dir)
        for url in urls:
            info = downloader.parse_album_page(url)
            if info and info.tracks and not args.list_only:
                downloader.download_album(info)


if __name__ == "__main__":
    main()
