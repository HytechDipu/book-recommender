import os
import re

import requests
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"

REGIONAL_LANGUAGES = [
    "hindi", "gujarati", "tamil", "marathi", "telugu",
    "malayalam", "bengali", "kannada", "punjabi", "urdu", "odia"
]


def normalize_title(title):
    """Strips parentheses, subtitles, edition tags, and leading 'the/a/an'."""
    t = title.lower()
    t = re.sub(r"\(.*?\)", "", t)          # Remove anything in (parentheses)
    t = re.split(r"[:\-–]", t)[0]          # Keep only text before ':', '-', or '–'
    t = re.sub(r"collector'?s?\s*edition|special\s*edition|box\s*set", "", t)
    t = re.sub(r"[^a-z0-9\s]", "", t)      # Remove punctuation
    t = re.sub(r"^(the|a|an)\s+", "", t.strip())  # Remove leading 'the', 'a', 'an'
    return " ".join(t.split())


def is_translated_edition(title):
    """Returns True if the title mentions a regional language translation."""
    t = title.lower()
    return any(lang in t for lang in REGIONAL_LANGUAGES)


def fetch_books(query, max_results=40):
    """Fetch books from Google Books and convert the useful fields into one dictionary."""
    params = {
        "q": query,
        "maxResults": max_results,
        "printType": "books",
        "langRestrict": "en",
    }

    api_key = os.getenv("GOOGLE_BOOKS_API_KEY")
    if api_key:
        params["key"] = api_key

    try:
        response = requests.get(GOOGLE_BOOKS_URL, params=params, timeout=10)
        response.raise_for_status()
    except (
        requests.exceptions.HTTPError,
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.RequestException,
    ):
        return []

    try:
        items = response.json().get("items", [])
    except ValueError:
        return []

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

        books.append({
            "title": title,
            "authors": authors,
            "categories": categories,
            "description": description,
            "thumbnail": thumbnail,
            "info_link": info_link,
            # Higher weight for category (3x), medium for author (2x), normal for description (1x)
            "tags": f"{categories} {categories} {categories} {categories} {categories} {authors} {authors} {authors} {authors} {description}",
        })

    return books


def get_live_recommendations(user_query):
    # 1. Find the book the user searched for
    target_results = fetch_books(user_query, max_results=5)
    if not target_results:
        return None, []

    target_book = target_results[0]

    # 2. Build a smart candidate pool from 3 sources
    first_author = target_book["authors"].split(",")[0].strip()
    primary_category = target_book["categories"].split(",")[0].strip()

    pool = []
    # Source A: Other books by the same author (e.g., Amish Tripathi)
    if first_author != "Unknown Author":
        pool.extend(fetch_books(f'inauthor:"{first_author}"', max_results=20))

    # Source B: Books matching the title/topic keywords (finds similar authors)
    pool.extend(fetch_books(f'{target_book["title"]} {primary_category}', max_results=20))

    # Source C: Broader category books for variety
    if primary_category != "General":
        pool.extend(fetch_books(f'subject:"{primary_category}"', max_results=20))

    # 3. Deduplicate editions, remove target book, and skip regional translations
    # 3. Deduplicate editions, remove target book, and skip translations
    target_base = normalize_title(target_book["title"])
    candidates = []
    seen_base_titles = {target_base}

    for book in pool:
        if is_translated_edition(book["title"]):
            continue

        # Check full raw title first (catches "Meluha Ke Mritunjay (Immortals Of Meluha)")
        raw_clean = " ".join(re.sub(r"[^a-z0-9\s]", " ", book["title"].lower()).split())
        if target_base and target_base in raw_clean:
            continue

        base = normalize_title(book["title"])
        if not base:
            continue

        # Skip if it matches or overlaps with ANY already-accepted book title
        if any(base == seen or base in seen or seen in base for seen in seen_base_titles):
            continue

        seen_base_titles.add(base)
        candidates.append(book)

    # 4. Run TF-IDF + Cosine Similarity on the combined pool
    all_tags = [target_book["tags"]] + [b["tags"] for b in candidates]
    tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
    tfidf_matrix = tfidf.fit_transform(all_tags)

    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

    scored_books = list(enumerate(scores))
    scored_books.sort(key=lambda x: x[1], reverse=True)

    recommendations = []
    for idx, score in scored_books[:5]:
        book = candidates[idx]
        recommendations.append({
            "title": book["title"],
            "authors": book["authors"],
            "categories": book["categories"],
            "description": book["description"],
            "thumbnail": book["thumbnail"],
            "info_link": book.get("info_link", "#"),
            "match": round(float(score) * 100, 1),
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