from llm_client import ask_llm


def rewrite_if_needed(reply: str) -> str:

    check_prompt = f"""
Rewrite this reply to ensure emotional validation and supportive tone.
Keep the meaning the same.

Reply:
{reply}
"""

    try:
        return ask_llm(check_prompt)
    except Exception:
        return reply
