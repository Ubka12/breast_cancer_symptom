# backend/nhs_scraper.py
# ------------------------------------------------------------
# Purpose
#   Build a small CSV “lexicon” of NHS symptom bullet points.
#   The CSV is later used as reference text (e.g., for rules, examples).
#
# What it does
#   • Politely fetch one NHS page (default: symptoms of breast cancer in women)
#   • Parse headings and their bullet lists
#   • Extract each bullet as a row with its section cue and date
#   • Write rows to data/symptom_lexicon.csv
#
# Notes
#   • We respect robots.txt if it’s reachable (best-effort check).
#   • One-page, one-shot scraper (not a crawler).
#   • User agent is set and a small delay is used to be polite.
# ------------------------------------------------------------

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

# Default NHS page we read once
DEFAULT_URL = (
    "https://www.nhs.uk/conditions/breast-cancer-in-women/"
    "symptoms-of-breast-cancer-in-women/"
)

# Save output next to this file: backend/data/symptom_lexicon.csv
DATA_DIR = (Path(__file__).resolve().parent / "data").resolve()
DEFAULT_OUT = DATA_DIR / "symptom_lexicon.csv"


def can_fetch(url: str, user_agent: str = "MSc-project/1.0") -> bool:
    """
    Quick, best-effort robots.txt check.
    Returns True if:
      • robots.txt permits the fetch for our user agent, or
      • robots.txt cannot be reached (we default to True and proceed politely).
    We do not hard-fail the run if robots.txt is down.
    """
    try:
        base = url.split("/", 3)[:3]
        robots_url = "/".join(base) + "/robots.txt"
        rp = robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True  # proceed politely if robots.txt is unreachable


def fetch_html(url: str, user_agent: str = "MSc-project/1.0", timeout: int = 10) -> BeautifulSoup:
    """
    Download the page with a simple GET and return a BeautifulSoup parser.
    - Sets a custom User-Agent.
    - Raises for HTTP errors so callers can stop early.
    """
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": user_agent})
    resp.raise_for_status()
    return BeautifulSoup(resp.content, "html.parser")


def extract_symptom_rows(soup: BeautifulSoup, url: str) -> List[Tuple[str, str, str, str]]:
    """
    Pull out bullet points under relevant headings and return rows:
      (canonical_term, nhs_cue, source_url, accessed_at)

    Simple strategy:
      1) Walk H2/H3 headings in order.
      2) Keep only sections whose heading starts with:
           “symptom…”, “sign…”, or “see a gp…”
      3) Within that section (until the next H2/H3), collect all <li> items.
      4) Use the heading text as a short “cue” and the <li> text as the term.

    We also de-duplicate identical (term, cue) pairs (case-insensitive).
    """
    rows: List[Tuple[str, str, str, str]] = []
    today = dt.date.today().isoformat()

    # All headings in reading order
    headings = soup.find_all(["h2", "h3"])

    for idx, h in enumerate(headings):
        title = (h.get_text(strip=True) or "").lower()
        if not title.startswith(("symptom", "sign", "see a gp")):
            continue

        # Collect sibling nodes until the next H2/H3 (defines the section)
        section_nodes: List = []
        n = h.next_sibling
        while n and getattr(n, "name", None) not in ("h2", "h3"):
            section_nodes.append(n)
            n = n.next_sibling

        # Gather list items within this section
        lis = []
        for node in section_nodes:
            lis.extend(getattr(node, "find_all", lambda *_: [])("li"))

        for li in lis:
            term = li.get_text(" ", strip=True)
            if term:
                rows.append((term, title, url, today))

    # De-duplicate by (term, cue) ignoring case/spacing
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
    """
    Write rows to CSV with header:
      canonical_term, nhs_cue, source_url, accessed_at
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["canonical_term", "nhs_cue", "source_url", "accessed_at"])
        w.writerows(rows)


def main(url: str, out_csv: Path, user_agent: str = "MSc-project/1.0") -> int:
    """
    One-shot pipeline:
      • Check robots.txt (best-effort)
      • Fetch page
      • Extract bullets under relevant sections
      • Save CSV to disk
      • Return the number of rows written
    """
    if not can_fetch(url, user_agent=user_agent):
        print(f"[warn] robots.txt disallows fetching {url}. Aborting politely.")
        return 0

    print(f"[info] fetching {url}")
    soup = fetch_html(url, user_agent=user_agent)
    time.sleep(1.0)  # small delay in case this script is looped

    rows = extract_symptom_rows(soup, url=url)
    write_csv(rows, out_csv)
    print(f"[ok] wrote {len(rows)} rows → {out_csv}")
    return len(rows)


if __name__ == "__main__":
    # Small CLI so this can be run directly:
    #   python backend/nhs_scraper.py --url <page> --out data/symptom_lexicon.csv
    parser = argparse.ArgumentParser(description="Scrape NHS symptom bullets into a CSV lexicon.")
    parser.add_argument("--url", default=DEFAULT_URL, help="NHS page to scrape")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output CSV path")
    args = parser.parse_args()

    main(args.url, Path(args.out))
