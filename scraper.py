import re
import requests

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"


def fetch_wikipedia_text(title):
    """Fetch clean plain-text extract of a Wikipedia article."""
  
    headers = {
        "User-Agent": "CatFactsRAGBot/1.0 (myemail@example.com) Python-requests/2.0"
    }
    
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,   # plain text, no HTML/wiki markup
    }
    
    
    response = requests.get(WIKI_API_URL, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()

    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    if "extract" not in page:
        raise ValueError(f"No article found for title: {title}")
    return page["extract"]

# def fetch_wikipedia_text(title):
#     """Fetch clean plain-text extract of a Wikipedia article."""
#     params = {
#         "action": "query",
#         "format": "json",
#         "titles": title,
#         "prop": "extracts",
#         "explaintext": True,   # plain text, no HTML/wiki markup
#     }
#     response = requests.get(WIKI_API_URL, params=params)
#     response.raise_for_status()
#     data = response.json()

#     pages = data["query"]["pages"]
#     page = next(iter(pages.values()))
#     if "extract" not in page:
#         raise ValueError(f"No article found for title: {title}")
#     return page["extract"]
    

def split_into_chunks(text):
    """Split article text into clean sentence-level chunks (similar to cat-facts.txt format)."""
    # remove section headers like "== History ==" and empty lines
    text = re.sub(r"={2,}.*?={2,}", "", text)
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    chunks = []
    for line in lines:
        # split each paragraph into sentences
        sentences = re.split(r"(?<=[.!?])\s+", line)
        for s in sentences:
            s = s.strip()
            if len(s) > 20:  # skip very short/junk fragments
                chunks.append(s)
    return chunks

def scrape_and_save(titles, output_file="wiki-cat-facts.txt"):
    all_chunks = []
    for title in titles:
        print(f"Fetching: {title}")
        text = fetch_wikipedia_text(title)
        chunks = split_into_chunks(text)
        all_chunks.extend(chunks)
        print(f"  -> {len(chunks)} chunks extracted")

    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(chunk + "\n")

    print(f"\nSaved {len(all_chunks)} total chunks to {output_file}")

if __name__ == "__main__":
    titles = ["Cat", "Cat behavior", "Kitten", "Cat health"]
    scrape_and_save(titles)