import json
import os
import re
from functools import lru_cache

import requests
from flask import Flask, render_template, request
from requests.adapters import HTTPAdapter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib3.util.retry import Retry

app = Flask(__name__)

GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"
FALLBACK_CACHE_PATH = os.path.join(os.path.dirname(__file__), "fallback_cache.json")

REGIONAL_LANGUAGES = [
    "hindi", "gujarati", "tamil", "marathi", "telugu",
    "malayalam", "bengali", "kannada", "punjabi", "urdu", "odia",
]


def _build_http_session():
    """Creates a requests session with automatic retries for transient API errors."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


http_session = _build_http_session()


def normalize_title(title):
    """Strips parentheses, subtitles, edition tags, and leading 'the/a/an'."""
    t = title.lower()
    t = re.sub(r"\(.*?\)", "", t)
    t = re.split(r"[:\-–]", t)[0]
    t = re.sub(r"collector'?s?\s*edition|special\s*edition|box\s*set", "", t)
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"^(the|a|an)\s+", "", t.strip())
    return " ".join(t.split())


def is_translated_edition(title):
    """Returns True if the title mentions a regional language translation."""
    t = title.lower()
    return any(lang in t for lang in REGIONAL_LANGUAGES)


def _load_fallback_cache():
    """Loads local fallback responses for offline demo resilience."""
    if not os.path.exists(FALLBACK_CACHE_PATH):
        return {}
    try:
        with open(FALLBACK_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


@lru_cache(maxsize=128)
def fetch_books(query, max_results=40):
    """Fetch books from Google Books with in-memory caching, retries, and offline fallback."""
    params = {
        "q": query,
        "maxResults": max_results,
        "printType": "books",
        "langRestrict": "en",
    }

    api_key = os.getenv("GOOGLE_BOOKS_API_KEY")
    if api_key:
        params["key"] = api_key

    items = None
    try:
        response = http_session.get(GOOGLE_BOOKS_URL, params=params, timeout=8)
        response.raise_for_status()
        items = response.json().get("items", [])
    except (requests.exceptions.RequestException, ValueError):
        # Offline Viva Insurance: fall back to cached JSON if available
        cache = _load_fallback_cache()
        items = cache.get(query.lower().strip(), [])

    books = []
    seen_titles = set()

    for item in items:
        volume_info = item.get("volumeInfo", {})
        title = volume_info.get("title", "").strip()
        description = volume_info.get("description", "").strip()

        if not title or not description:
            continue

        title_key = title.lower()
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)

        authors_list = volume_info.get("authors") or []
        authors = ", ".join(authors_list).strip() or "Unknown Author"

        categories_list = volume_info.get("categories") or []
        categories = ", ".join(categories_list).strip() or "General"

        thumbnail = volume_info.get("imageLinks", {}).get("thumbnail", "")
        thumbnail = thumbnail.replace("http://", "https://")
        info_link = volume_info.get("infoLink", "#") or "#"

        # Clean weighting: 3x category, 2x author, 1x description
        tags = " ".join([categories] * 3 + [authors] * 2 + [description])

        books.append({
            "title": title,
            "authors": authors,
            "categories": categories,
            "description": description,
            "thumbnail": thumbnail,
            "info_link": info_link,
            "tags": tags,
        })

    return tuple(books)


def extract_shared_keywords(tfidf, target_vec, candidate_vec, top_n=4):
    """Explainable AI (XAI): returns the top shared TF-IDF terms between two books."""
    shared = target_vec.multiply(candidate_vec)
    if shared.nnz == 0:
        return []

    feature_names = tfidf.get_feature_names_out()
    # Sort non-zero shared features by contribution weight descending
    sorted_indices = shared.indices[shared.data.argsort()[::-1]]
    return [feature_names[i] for i in sorted_indices[:top_n]]


def get_live_recommendations(user_query, max_per_author=2, top_k=5):
    # 1. Find the book the user searched for
    target_results = fetch_books(user_query, max_results=5)
    if not target_results:
        return None, []

    target_book = target_results[0]

    # 2. Build candidate pool from 3 sources
    first_author = target_book["authors"].split(",")[0].strip()
    primary_category = target_book["categories"].split(",")[0].strip()

    pool = []
    if first_author != "Unknown Author":
        pool.extend(fetch_books(f'inauthor:"{first_author}"', max_results=20))
    pool.extend(fetch_books(f'{target_book["title"]} {primary_category}', max_results=20))
    if primary_category != "General":
        pool.extend(fetch_books(f'subject:"{primary_category}"', max_results=20))

    # 3. Deduplicate editions, remove target book, and skip regional translations
    target_base = normalize_title(target_book["title"])
    candidates = []
    seen_base_titles = {target_base}

    for book in pool:
        if is_translated_edition(book["title"]):
            continue

        raw_clean = " ".join(re.sub(r"[^a-z0-9\s]", " ", book["title"].lower()).split())
        if target_base and target_base in raw_clean:
            continue

        base = normalize_title(book["title"])
        if not base:
            continue

        if any(base == seen or base in seen or seen in base for seen in seen_base_titles):
            continue

        seen_base_titles.add(base)
        candidates.append(book)

    if not candidates:
        return target_book, []

    # 4. Run TF-IDF + Cosine Similarity
    all_tags = [target_book["tags"]] + [b["tags"] for b in candidates]
    tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
    tfidf_matrix = tfidf.fit_transform(all_tags)

    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    scored_books = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    # 5. Rerank with Author Diversity Cap (Max 2 per Author) + XAI Keywords
    recommendations = []
    author_counts = {}

    for idx, score in scored_books:
        if len(recommendations) >= top_k:
            break

        book = candidates[idx]
        primary_author = book["authors"].split(",")[0].strip().lower()

        if primary_author != "unknown author" and author_counts.get(primary_author, 0) >= max_per_author:
            continue

        author_counts[primary_author] = author_counts.get(primary_author, 0) + 1
        shared_terms = extract_shared_keywords(tfidf, tfidf_matrix[0:1], tfidf_matrix[idx + 1:idx + 2])

        recommendations.append({
            "title": book["title"],
            "authors": book["authors"],
            "categories": book["categories"],
            "description": book["description"],
            "thumbnail": book["thumbnail"],
            "info_link": book.get("info_link", "#"),
            "match": round(float(score) * 100, 1),
            "matched_keywords": shared_terms,
        })

    return target_book, recommendations


@app.route("/", methods=["GET", "POST"])
def home():
    selected_book = ""
    target_book = None
    recommendations = []
    error_message = ""

    if request.method == "POST":
        selected_book = request.form.get("book_title", "").strip()
        if not selected_book:
            error_message = "Please enter a book title."
        else:
            target_book, recommendations = get_live_recommendations(selected_book)
            if not target_book:
                error_message = (
                    "Could not find that book or reach Google Books. "
                    "Check your internet connection and try another title."
                )

    return render_template(
        "index.html",
        selected_book=selected_book,
        target_book=target_book,
        recommendations=recommendations,
        error_message=error_message,
    )


if __name__ == "__main__":
    app.run(debug=True)