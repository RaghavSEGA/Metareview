"""
Pulls every critic review for a Metacritic game directly from the JSON API the page itself
calls — no browser, no scrolling, no HTML parsing needed at all.

How this was found: the page's own Nuxt payload (visible in "view source") embeds the exact
API call it makes to fill in reviews, including working pagination links:

    https://backend.metacritic.com/reviews/metacritic/critic/games/<slug>/web
        ?offset=0&limit=10&filterBySentiment=all&sort=score
        &componentName=critic-reviews&componentDisplayName=critic+Reviews&componentType=ReviewList

That's the same request your browser's JS fires when you scroll — this script just calls it
directly and pages through `offset` in steps of 10 (the API appears to cap at 10 items per
call regardless of the `limit` value requested) until it stops returning new reviews.

Each item in the response's data.items already has everything the original scraping script
was trying to get out of the HTML (the review URL) plus a lot more for free: outlet name,
score, date, and the pull-quote — no BeautifulSoup needed.

Usage:
    python scrape_reviews_api.py persona-3-reload
    python scrape_reviews_api.py persona-3-reload --platform playstation-5
"""
import argparse
import csv
import sys
import time

import requests

BASE_URL = "https://backend.metacritic.com/reviews/metacritic/critic/games/{slug}/web"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.metacritic.com/",
    "Accept": "application/json",
}


def fetch_page(slug: str, offset: int, platform: str = None, page_size: int = 10) -> dict:
    params = {
        "offset": offset,
        "limit": page_size,
        "filterBySentiment": "all",
        "sort": "score",
        "componentName": "critic-reviews",
        "componentDisplayName": "critic Reviews",
        "componentType": "ReviewList",
    }
    if platform:
        params["platform"] = platform
    resp = requests.get(BASE_URL.format(slug=slug), params=params, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.json()


def get_all_reviews(slug: str, platform: str = None, pause_s: float = 0.5, max_pages: int = 50) -> list:
    reviews = []
    offset = 0
    for _ in range(max_pages):
        payload = fetch_page(slug, offset, platform=platform)
        items = payload.get("data", {}).get("items", [])
        if not items:
            break
        reviews.extend(items)
        print(f"offset={offset}: got {len(items)} reviews (total so far: {len(reviews)})")
        offset += len(items)
        time.sleep(pause_s)  # be polite between requests
    return reviews


def main():
    parser = argparse.ArgumentParser(description="Pull all critic reviews for a Metacritic game.")
    parser.add_argument("slug", help="Game slug from the URL, e.g. 'persona-3-reload'")
    parser.add_argument("--platform", default=None,
                         help="Optional platform slug to filter, e.g. 'playstation-5'")
    parser.add_argument("--csv", default=None, help="Optional path to write results as CSV")
    args = parser.parse_args()

    reviews = get_all_reviews(args.slug, platform=args.platform)
    print(f"\n{len(reviews)} review(s) found.\n")

    for r in reviews:
        print(r.get("url"))

    if args.csv:
        fieldnames = ["publicationName", "score", "date", "url", "quote", "platform"]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in reviews:
                writer.writerow(r)
        print(f"\nWrote {len(reviews)} rows to {args.csv}")


if __name__ == "__main__":
    main()