from transformers import pipeline
from langdetect import detect
import re

print("Loading emotion model...")

emotion_pipe = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=3
)

NEG = {"sadness", "anger", "frustration", "fear"}
POS = {"happiness", "joy", "surprise"}


# ---------- slang patterns ----------

DISTRESS_PATTERNS = [
    "i'm cooked", "im cooked", "i am cooked",
    "feeling cooked",
    "i'm done", "im done",
    "i'm finished", "im finished",
    "i cant anymore", "i can't anymore",
    "i give up",
    "i'm not okay", "im not okay",
    "this is bad for me"
]

WIN_PATTERNS = [
    "we cooked",
    "we are cooking",
    "we're cooking",
    "they got cooked",
    "he got cooked",
    "she got cooked"
]

GRIEF_PHRASES = [
    "died", "passed away", "lost my",
    "funeral", "death", "put to sleep"
]


# ---------- style signal detectors ----------

def detect_caps_intensity(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return sum(c.isupper() for c in letters) / len(letters) > 0.6


def detect_stretch_words(text: str) -> bool:
    return bool(re.search(r"(.)\1{3,}", text))


def detect_bro_style(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in ["bro", "dude", "man", "bruh"])


# ---------- override logic ----------

def grief_override(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in GRIEF_PHRASES)


def distress_slang_override(text: str) -> bool:
    t = text.lower()

    if any(p in t for p in WIN_PATTERNS):
        return False

    if any(p in t for p in DISTRESS_PATTERNS):
        return True

    if "cooked" in t and any(x in t for x in ["i'm", "im", "i am", "feeling", "so", "too"]):
        return True

    return False


# ---------- scoring ----------

def overall_emotion(emotions):

    if not emotions:
        return "neutral"

    label = emotions[0]["label"]
    score = emotions[0]["score"]

    if label in NEG:
        return "strong_distress" if score > 0.75 else "distress"

    if label in POS:
        return "super_happy" if score > 0.75 else "happy"

    return "neutral"


# ---------- main ----------

def analyze_text(text: str):

    if not text.strip():
        return {
            "language": "unknown",
            "emotions": [],
            "overall": "neutral"
        }

    clean = text.strip()

    try:
        lang = "en" if len(clean) < 20 else detect(clean)
    except:
        lang = "unknown"

    emo = emotion_pipe(clean)[0]

    emotions = [
        {"label": e["label"], "score": float(e["score"])}
        for e in emo
    ]

    overall = overall_emotion(emotions)

    if grief_override(clean):
        overall = "strong_distress"
    elif distress_slang_override(clean):
        overall = "strong_distress"

    return {
        "language": lang,
        "emotions": emotions,
        "overall": overall,
        "caps_intense": detect_caps_intensity(clean),
        "stretch_intense": detect_stretch_words(clean),
        "bro_style": detect_bro_style(clean)
    }
