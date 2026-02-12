import requests


def clean_query(text: str) -> str:
    t = text.lower()

    remove = [
        "is", "was", "are", "the", "a",
        "movie", "film", "song",
        "very", "so", "really"
    ]

    words = [w for w in t.split() if w not in remove]
    return " ".join(words[:5])   # keep short


def web_search(query: str) -> str:

    q = clean_query(query)

    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{q.replace(' ', '_')}"

    try:
        r = requests.get(url, timeout=6)

        if r.status_code == 200:
            data = r.json()
            return data.get("extract", "")

    except Exception as e:
        print("Wiki error:", e)

    return ""
