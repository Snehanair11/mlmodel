from transformers import pipeline
from langdetect import detect
import re

# ---------------- MODEL LAZY LOAD ----------------

emotion_pipe = None
emotion_available = True   # global flag


def get_emotion_pipe():
    global emotion_pipe, emotion_available

    if emotion_pipe is None and emotion_available:
        try:
            print("Loading emotion model (lazy)...")
            emotion_pipe = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=3
            )
        except Exception as e:
            print("Emotion model load failed — fallback mode:", e)
            emotion_available = False
            emotion_pipe = None

    return emotion_pipe


# ---------------- LABEL GROUPS ----------------

NEG = {"sadness", "anger", "fear", "disgust"}
POS = {"joy", "happiness", "surprise", "love"}


# ---------------- SLANG PATTERNS ----------------

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


# ---------------- STYLE SIGNALS ----------------

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


# ---------------- OVERRIDES ----------------

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


# ---------------- FALLBACK HEURISTIC ----------------

def heuristic_emotion(text: str):
    t = text.lower()

    if any(w in t for w in ["yay", "finally", "won", "got it", "success"]):
        return "super_happy"

    if any(w in t for w in ["sad", "tired", "overwhelmed", "cry", "done"]):
        return "distress"

    return "neutral"


# ---------------- SCORING ----------------

def overall_emotion(emotions):

    if not emotions:
        return "neutral"

    label = emotions[0]["label"].lower()
    score = emotions[0]["score"]

    if label in NEG:
        return "strong_distress" if score > 0.75 else "distress"

    if label in POS:
        return "super_happy" if score > 0.75 else "happy"

    return "neutral"


# ---------------- MAIN ANALYZER ----------------

def analyze_text(text: str):

    if not text or not text.strip():
        return {
            "language": "unknown",
            "emotions": [],
            "overall": "neutral",
            "caps_intense": False,
            "stretch_intense": False,
            "bro_style": False
        }

    clean = text.strip()

    # ---- language detect ----
    try:
        lang = "en" if len(clean) < 20 else detect(clean)
    except:
        lang = "unknown"

    emotions = []
    overall = "neutral"

    # ---- model inference (safe lazy) ----
    pipe = get_emotion_pipe()

    if pipe:
        try:
            emo = pipe(clean)[0]
            emotions = [
                {"label": e["label"], "score": float(e["score"])}
                for e in emo
            ]
            overall = overall_emotion(emotions)
        except Exception as e:
            print("Emotion inference failed:", e)
            overall = heuristic_emotion(clean)
    else:
        overall = heuristic_emotion(clean)

    # ---- overrides ----
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
