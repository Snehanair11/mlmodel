from llm_client import ask_llm


def build_convo_summary(context):

    if not context:
        return ""

    joined = "\n".join(context[-30:])

    prompt = f"""
Summarize this conversation in 2 short lines.

Include:
- main emotional theme
- main situation topic

Do not give advice.
Do not change tone.
Just neutral summary.

Conversation:
{joined}
"""

    return ask_llm(prompt)
