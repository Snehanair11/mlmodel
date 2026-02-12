import os
import time
import requests
from dotenv import load_dotenv

load_dotenv(".env")

GROQ_KEY = os.getenv("GROQ_API_KEY")
print("GROQ KEY FOUND:", bool(GROQ_KEY))

URL = "https://api.groq.com/openai/v1/chat/completions"


# -------- prompt type detectors --------

def is_grief_prompt(prompt: str) -> bool:
    p = prompt.lower()
    return any(w in p for w in [
        "passed away", "died", "lost my",
        "funeral", "bereavement"
    ])


def is_distress_prompt(prompt: str) -> bool:
    p = prompt.lower()
    return any(w in p for w in [
        "strong_distress", "distress",
        "overwhelmed", "frustrated",
        "sad", "stress", "not okay"
    ])


def is_positive_prompt(prompt: str) -> bool:
    p = prompt.lower()
    return any(w in p for w in [
        "super_happy", "happy", "joy", "positive"
    ])


# -------- main call --------

def generate_reply(prompt: str) -> str:

    if not GROQ_KEY:
        return "System configuration error."

    # ----- system behavior contracts -----

    if is_grief_prompt(prompt):

        system_msg = (
            "User is grieving. Write 3 to 4 medium-long emotionally rich "
            "consolation sentences. Be warm, present, and human. "
            "Do NOT ask any questions. Only comfort and validate."
        )
        max_tokens = 360

    elif is_distress_prompt(prompt):

        system_msg = (
            "User is emotionally distressed. Write 3 to 5 medium-long "
            "supportive sentences. Validate feelings first. "
            "You MUST include exactly ONE gentle open-ended question "
            "asking what happened or what led to this. "
            "Do not ask more than one question. "
            "Keep conversation continuity."
        )
        max_tokens = 340

    elif is_positive_prompt(prompt):

        system_msg = (
            "User is feeling positive. Write 3 to 4 supportive, upbeat "
            "medium-length sentences. Encouraging and grounded. "
            "You MAY include one light forward-looking question."
        )
        max_tokens = 260

    else:

        system_msg = (
            "Write 2 to 3 medium-length friendly conversational sentences."
        )
        max_tokens = 220

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.75,
        "max_tokens": max_tokens
    }

    headers = {
        "Authorization": f"Bearer {GROQ_KEY}",
        "Content-Type": "application/json"
    }

    # ----- retry wrapper -----

    for attempt in range(3):
        try:
            r = requests.post(URL, headers=headers, json=payload, timeout=45)

            if r.status_code == 200:
                data = r.json()
                if data.get("choices"):
                    return data["choices"][0]["message"]["content"].strip()
            else:
                return f"LLM HTTP error {r.status_code}"

        except requests.exceptions.RequestException as e:
            print("LLM retry:", e)
            time.sleep(1.5 * (attempt + 1))

    return "Something glitched — can you say that again?"


# backward compatibility for your imports
def ask_llm(prompt: str) -> str:
    return generate_reply(prompt)
