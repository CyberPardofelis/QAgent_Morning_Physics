import feedparser
from datetime import datetime

ARXIV_URL = "http://export.arxiv.org/rss/quant-ph"

def fetch_new_papers():
    feed = feedparser.parse(ARXIV_URL)
    papers = []

    for entry in feed.entries[:10]:
        papers.append({
            "title": entry.title,
            "summary": entry.summary,
            "link": entry.link,
            "published": entry.published
        })

    return papers