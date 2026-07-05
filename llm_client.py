import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
LLM_MODEL = "gemini-2.5-flash"
MAX_HISTORY_TURNS = 3


def init_llm_client() -> OpenAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set. Add it to .env or export it.")

    client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)
    print(f"[Info] LLM model: {LLM_MODEL}")
    return client


def _build_messages(question, context_chunks, history=None):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions strictly based "
                "on the provided document context. If the answer is not in the "
                "context, say so clearly instead of guessing."
            ),
        }
    ]

    # add recent conversation history
    if history:
        for past_q, past_a in history[-MAX_HISTORY_TURNS:]:
            messages.append({"role": "user", "content": past_q})
            messages.append({"role": "assistant", "content": past_a})

    # build context + question as the current user message
    context_block = "\n\n".join(
        f"[Chunk {i}]\n{chunk}" for i, chunk in enumerate(context_chunks, 1)
    )

    user_msg = (
        f"Here is the relevant context from the document:\n\n"
        f"{context_block}\n\n---\n\n"
        f"Question: {question}\n\n"
        f"Answer based only on the context above. Be concise."
    )
    messages.append({"role": "user", "content": user_msg})
    return messages


def answer_question(client, question, context_chunks, history=None):
    messages = _build_messages(question, context_chunks, history)

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"[LLM Error] {exc}"
