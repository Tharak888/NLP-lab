import re
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def fetch_text(url):
    """Fetches text from the URL's paragraph tags and normalizes whitespace."""

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return ""

    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join(p.get_text() for p in paragraphs)

    return re.sub(r'\s+', ' ', text).strip()

def segment_text(url):
    """
    Fetches text, tokenizes it into sentences, and returns a specific slice.
    """
    text = fetch_text(url)

    if not text:
        return []

    sentences = sent_tokenize(text)
    return sentences[5:8]

if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/Natural_language_processing"
    chunks = segment_text(url)

    print("Extracted Chunks:")
    if chunks:
        for i, sentence in enumerate(chunks, 1):
            print(f"{i}. {sentence}")
    else:
        print("— Could not extract the desired sentences. Check connection/errors. —")
