# AI Book Recommendation System

A Flask mini-project that searches live book data from the free Google Books API and recommends related books using content-based filtering.

## How it works

1. The user enters a book title.
2. Google Books is searched for that title and the first valid result becomes the target book.
3. More live candidates are fetched using the target book's primary category (or author if no category exists).
4. TF-IDF converts each book's author, category, and description into a numerical text vector.
5. Cosine similarity compares the target vector with every candidate vector.
6. The five candidates with the highest similarity scores are displayed with percentage match values.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open `http://127.0.0.1:5000` in a browser.

## Optional Google Books API key

The app works without an API key. If Google rate limits the requests, set an environment variable:

**Windows PowerShell**

```powershell
$env:GOOGLE_BOOKS_API_KEY="your_api_key"
python app.py
```

**Windows Command Prompt**

```bat
set GOOGLE_BOOKS_API_KEY=your_api_key
python app.py
```
