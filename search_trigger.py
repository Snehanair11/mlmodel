def should_web_search(text: str) -> bool:
    t = text.lower()

    # explicit knowledge queries
    if any(w in t for w in [
        "who is", "what is", "tell me about",
        "movie", "film", "song",
        "actor", "director",
        "company", "brand"
    ]):
        return True

    # short entity-style statements → likely knowledge
    words = t.split()
    if len(words) <= 6:
        return True

    return False
