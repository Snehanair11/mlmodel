from datetime import datetime

chat_memory = {}
chat_summaries = {}

MAX_LINES = 50


def add_message(session_id: str, text: str, role: str = "user", emotion: str = ""):

    chat_memory.setdefault(session_id, []).append({
        "role": role,
        "text": text,
        "emotion": emotion,
        "ts": datetime.utcnow().isoformat()
    })

    chat_memory[session_id] = chat_memory[session_id][-MAX_LINES:]


def get_context(session_id: str):

    msgs = chat_memory.get(session_id, [])

    # return readable conversational context
    return [
        f'{m["role"]}: {m["text"]}'
        for m in msgs
    ]


def save_summary(session_id: str, summary: str):
    chat_summaries[session_id] = summary


def get_summary(session_id: str):
    return chat_summaries.get(session_id, "")
