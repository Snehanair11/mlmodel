import re


# -------- severity detectors --------

def is_grief_text(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in [
        "died", "passed away", "lost my", "death", "funeral"
    ])


def is_heavy_emotion(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in [
        "frustrated", "overwhelmed", "low", "depressed",
        "sad", "hurt", "broken", "tired of",
        "not okay", "cooked", "done", "give up"
    ])


# -------- dynamic sentence cap --------

def sentence_cap(user_text: str) -> int:

    words = len(user_text.split())
    t = user_text.lower().strip()

    # grief → fuller consolation
    if is_grief_text(t):
        return 4

    # distress → longer support
    if is_heavy_emotion(t):
        return 4

    # short input → still allow convo hook
    if words <= 3:
        return 3

    # greetings
    if t.startswith(("hi", "hello", "hey")):
        return 3

    return 3


# -------- limiter --------

def limit_sentences(text: str, user_text: str) -> str:

    cap = sentence_cap(user_text)

    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    parts = [p.strip() for p in parts if p.strip()]

    if not parts:
        return text.strip()

    trimmed = " ".join(parts[:cap])

    # ensure punctuation
    if trimmed[-1] not in ".!?":
        trimmed += "."

    # ---- continuity safety net ----
    if is_heavy_emotion(user_text) and "?" not in trimmed:
        trimmed += " What happened there?"

    return trimmed
