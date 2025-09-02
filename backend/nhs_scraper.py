# backend/nhs_scraper.py
from __future__ import annotations

import argparse
import csv
import datetime as dt
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import requests
from bs4 import BeautifulSoup
from urllib import robotparser

DEFAULT_URL = (
    "https://www.nhs.uk/conditions/breast-cancer-in-women/"
    "symptoms-of-breast-cancer-in-women/"
)

# Resolve /data next to this file (backend/ -> backend/data/)
DATA_DIR = (Path(__file__).resolve().parent / "data").resolve()
DEFAULT_OUT = DATA_DIR / "symptom_lexicon.csv"


def can_fetch(url: str, user_agent: str = "MSc-project/1.0") -> bool:
    """
    Best-effort robots.txt check. Returns True if we can't check or if allowed.
    (If robots.txt is unreachable, we default to True and proceed politely.)
    """
    try:
        base = url.split("/", 3)[:3]
        robots_url = "/".join(base) + "/robots.txt"
        rp = robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True  # do not hard-fail the run


def fetch_html(url: str, user_agent: str = "MSc-project/1.0", timeout: int = 10) -> BeautifulSoup:
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": user_agent})
    resp.raise_for_status()
    return BeautifulSoup(resp.content, "html.parser")


def extract_symptom_rows(soup: BeautifulSoup, url: str) -> List[Tuple[str, str, str, str]]:
    """
    Collect (canonical_term, nhs_cue, source_url, accessed_at) rows.
    Strategy:
      - Look for headings that look like symptom groups (h2/h3 containing
        'symptom', 'sign', 'see a gp')
      - For each, collect all <li> items until the next heading.
    """
    rows: List[Tuple[str, str, str, str]] = []
    today = dt.date.today().isoformat()

    # all headings in reading order
    headings = soup.find_all(["h2", "h3"])

    for idx, h in enumerate(headings):
        title = (h.get_text(strip=True) or "").lower()
        if not title.startswith(("symptom", "sign", "see a gp")):
            continue

        # Everything between this heading and next heading
        section_nodes: List = []
        n = h.next_sibling
        while n and n.name not in ("h2", "h3"):
            section_nodes.append(n)
            n = n.next_sibling

        # find list items inside that section
        lis = []
        for node in section_nodes:
            lis.extend(getattr(node, "find_all", lambda *_: [])("li"))

        for li in lis:
            term = li.get_text(" ", strip=True)
            if term:
                rows.append((term, title, url, today))

    # de-duplicate by term+cue (case-insensitive)
    seen: set = set()
    deduped: List[Tuple[str, str, str, str]] = []
    for r in rows:
        key = (r[0].strip().lower(), r[1].strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    return deduped


def write_csv(rows: Iterable[Tuple[str, str, str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["canonical_term", "nhs_cue", "source_url", "accessed_at"])
        w.writerows(rows)


def main(url: str, out_csv: Path, user_agent: str = "MSc-project/1.0") -> int:
    if not can_fetch(url, user_agent=user_agent):
        print(f"[warn] robots.txt disallows fetching {url}. Aborting politely.")
        return 0

    print(f"[info] fetching {url}")
    soup = fetch_html(url, user_agent=user_agent)
    # polite delay in case someone loops this script
    time.sleep(1.0)

    rows = extract_symptom_rows(soup, url=url)
    write_csv(rows, out_csv)
    print(f"[ok] wrote {len(rows)} rows → {out_csv}")
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape NHS symptom bullets into a CSV lexicon.")
    parser.add_argument("--url", default=DEFAULT_URL, help="NHS page to scrape")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output CSV path")
    args = parser.parse_args()

    main(args.url, Path(args.out))
