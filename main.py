from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware   # ✅ ADD
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from analyzer import analyze_text
from memory import add_message, get_context, save_summary, get_summary
from prompt_builder import build_prompt
from reply_filter import limit_sentences
from summary_builder import build_convo_summary
from llm_client import ask_llm

# ✅ search layer
from search_trigger import should_web_search
from search_client import web_search


app = FastAPI()

# ✅ CORS FIX — ADD THIS BLOCK
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4000",
        "http://127.0.0.1:4000",
        "https://your-frontend-domain.com",  # optional — replace later
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- request schema ----------------

class ChatRequest(BaseModel):
    anon_userid: str
    text: str
    timestamp: Optional[str] = None


# ---------------- route ----------------

@app.post("/chat")
def chat(data: ChatRequest):

    text = data.text.strip()
    sid = data.anon_userid

    # ---- memory write ----
    add_message(sid, text)

    context = get_context(sid)
    analysis = analyze_text(text)

    memory_block = " | ".join(context[-6:])
    old_summary = get_summary(sid)

    # ---------------- web search layer ----------------

    search_context = ""

    if should_web_search(text):
        try:
            search_context = web_search(text)
        except Exception as e:
            print("Search failed:", e)
            search_context = ""

    # ---------------- prompt ----------------

    prompt = build_prompt(
        raw_text=text,
        analysis=analysis,
        memory=memory_block,
        convo_summary=old_summary,
        search_context=search_context
    )

    # ---------------- LLM ----------------

    raw_reply = ask_llm(prompt)

    # ---------------- guardrail ----------------

    reply = limit_sentences(raw_reply, text)

    # ---------------- rolling summary ----------------

    if len(context) > 0 and len(context) % 10 == 0:
        summary = build_convo_summary(context)
        save_summary(sid, summary)

    # ---------------- response ----------------

    return {
        "reply": reply,
        "overall_emotion": analysis["overall"],
        "analysis": analysis,
        "stored_summary": get_summary(sid),
        "timestamp": datetime.utcnow().isoformat(),
        "anon_userid": sid
    }
