from rapport_rules import RAPPORT_RULES


# ---------- intent detectors ----------

def detect_identity_question(text: str) -> bool:
    t = text.lower()
    return any(q in t for q in [
        "who are you", "what are you",
        "are you a bot", "what can you do"
    ])


def detect_shopping_or_lifestyle(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in [
        "buy", "shop", "shirt", "dress",
        "wear", "outfit", "choose", "which one"
    ])


def detect_grief(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in [
        "died", "passed away", "lost my",
        "death", "funeral", "my pet", "my dog"
    ])


def detect_injury(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in [
        "broke my", "fracture", "injured",
        "hurt my", "accident"
    ])


def detect_crying(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in [
        "i'll cry", "i will cry",
        "crying", "about to cry"
    ])


def detect_goodbye(text: str) -> bool:
    t = text.lower().strip()
    return t in ["bye", "goodbye", "gn", "good night", "see you"]


def detect_distress_hint(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in [
        "i'm done", "im done",
        "i give up", "cant anymore",
        "too much", "over it"
    ])


def detect_celebration(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in [
        "yay", "yess", "lets go", "let's go",
        "i got it", "i did it",
        "finally got", "won"
    ])


def detect_professional_stress(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in [
        "delivery", "deadline", "stakeholder",
        "client", "timeline", "behind schedule",
        "delay", "escalation", "project risk",
        "not on track"
    ])


# ---------- helpers ----------

def is_long_message(text: str) -> bool:
    return len(text.split()) > 40


def convo_depth(memory: str) -> int:
    if not memory:
        return 0
    return len([m for m in memory.split("|") if m.strip()])


# ---------- tone rules ----------

HYPE_RULE = """
User is celebrating or emotionally positive.
Respond with warm grounded excitement and supportive hype.
Include exactly ONE light curiosity question about what they achieved.
"""


CARE_RULE = """
User is emotionally distressed or overwhelmed.
Start with validation and support.
Be human and steady.
Include exactly ONE gentle open-ended question about what happened.
Add one small stabilizing or encouraging line.
"""


PRO_STRESS_RULE = """
User is describing professional or delivery pressure.
Respond calm and confidence-supportive.
Acknowledge seriousness without dramatizing.
Add subtle morale reinforcement.
Include one forward-looking question about next step.
No therapy tone. No hype tone.
"""


GRIEF_RULE = """
User is experiencing loss.
Respond with heartfelt comfort and presence.
Write 3–4 medium consoling sentences.
No questions. No advice.
"""


INJURY_RULE = """
User reports a physical injury.
Respond with care and concern.
Encourage rest and proper care.
Ask how it happened.
"""


CRY_RULE = """
User is crying or near tears.
Respond with soothing reassurance.
Normalize the feeling.
Encourage slow breathing.
Include one gentle supportive question.
"""


CASUAL_RULE = """
User asks a normal lifestyle or preference question.
Answer directly and naturally.
No emotional support tone.
"""


IDENTITY_RULE = """
User asks who you are.
Answer briefly as a supportive AI assistant who helps with conversations and feelings.
Friendly and natural.
"""


GOODBYE_RULE = """
User is ending the conversation.
Respond with a warm human goodbye.
If earlier distress exists, include a short care note.
No questions.
"""


# ---------- energy style rules ----------

ENERGY_RULE = """
User message shows strong emotional intensity (caps or stretched words).
Match energy slightly with warmer, more animated wording.
Do not use all caps.
"""


BRO_RULE = """
User uses bro-style slang.
You may lightly mirror with “bro” or “dude” once.
Do not overuse slang.
"""


# ---------- prompt builder ----------

def build_prompt(raw_text, analysis, memory, convo_summary, search_context=""):

    text = raw_text.lower()
    emotions = [e["label"] for e in analysis["emotions"]]
    overall = analysis["overall"]

    depth = convo_depth(memory)
    long_mode = is_long_message(raw_text)

    # optional analyzer style flags (safe defaults)
    caps = analysis.get("caps_intense", False)
    stretch = analysis.get("stretch_intense", False)
    bro = analysis.get("bro_style", False)

    extra_rule = ""
    emotional_mode = False

    # ---------- routing ----------

    if detect_goodbye(text):
        extra_rule = GOODBYE_RULE

    elif detect_grief(text):
        extra_rule = GRIEF_RULE
        emotional_mode = True

    elif detect_injury(text):
        extra_rule = INJURY_RULE
        emotional_mode = True

    elif detect_crying(text):
        extra_rule = CRY_RULE
        emotional_mode = True

    elif detect_professional_stress(text):
        extra_rule = PRO_STRESS_RULE

    elif detect_identity_question(text):
        extra_rule = IDENTITY_RULE

    elif detect_shopping_or_lifestyle(text):
        extra_rule = CASUAL_RULE

    elif detect_celebration(text) or overall in ["joy", "super_happy", "happy", "positive"]:
        extra_rule = HYPE_RULE
        emotional_mode = True

    elif overall in ["distress", "strong_distress", "sadness", "fear", "anger"] \
         or detect_distress_hint(text):
        extra_rule = CARE_RULE
        emotional_mode = True

    # ---------- continuity hook ----------

    if emotional_mode and depth <= 2 and "No questions" not in extra_rule:
        extra_rule += "\nInclude one gentle context question to keep conversation flowing."

    # ---------- long emotional message handling ----------

    if long_mode and emotional_mode:
        extra_rule += "\nAcknowledge at least two emotional elements from the message."

    # ---------- style energy injection ----------

    style_block = ""
    if caps or stretch:
        style_block += ENERGY_RULE

    if bro:
        style_block += BRO_RULE

    # ---------- search context ----------

    search_block = ""
    if search_context:
        search_block = f"\nHelpful background facts (use if relevant):\n{search_context}\n"

    # ---------- final prompt ----------

    return f"""
{RAPPORT_RULES}

{extra_rule}

{style_block}

Reply in 3–5 medium sentences.
Natural tone. No robotic phrasing.
No long paragraphs.
Reply in the SAME language as the user.

Conversation summary:
{convo_summary}

Recent turns:
{memory}

{search_block}

Detected emotions: {', '.join(emotions)} | overall: {overall}

User message:
{raw_text}
"""
