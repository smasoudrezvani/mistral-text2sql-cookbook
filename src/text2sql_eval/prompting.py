"""Prompt building, SQL extraction, and the Mistral chat call."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mistralai.client import Mistral

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert SQLite analyst. Given a database schema, an optional hint, and a "
    "question, write ONE SQLite query that answers the question.\n"
    "Rules:\n"
    "- Use only tables and columns that exist in the schema.\n"
    "- Quote identifiers that contain spaces or special characters with backticks.\n"
    "- Return exactly the columns the question asks for, in the order it asks for them.\n"
    "- Reply with the SQL inside a single ```sql code block and nothing else."
)


def build_messages(schema: str, question: str, evidence: str) -> list[dict[str, str]]:
    """Build chat messages for one question.

    Args:
        schema: Rendered schema text.
        question: Natural-language question.
        evidence: Expert hint (may be empty).

    Returns:
        A system and a user message.
    """
    hint = evidence.strip() or "(none)"
    user = f"Schema:\n{schema}\n\nHint: {hint}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


_FENCED = re.compile(r"```(?:sql|sqlite)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model reply.

    Takes the last fenced code block if there is one, otherwise the whole reply.
    A trailing semicolon is removed.

    Args:
        text: Model reply.

    Returns:
        The SQL string (may be empty).
    """
    blocks = _FENCED.findall(text or "")
    sql = blocks[-1] if blocks else (text or "")
    return sql.strip().rstrip(";").strip()


def content_text(content: object) -> str:
    """Get the answer text from a message's content.

    Reasoning models (e.g. Mistral Large 4) return a list of chunks: ``ThinkChunk`` (the
    reasoning) and ``TextChunk`` (the answer). Only the text chunks are kept.

    Args:
        content: ``message.content``: a string, a list of chunks, or None.

    Returns:
        The answer text.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(getattr(ch, "text", "") for ch in content if getattr(ch, "type", None) == "text")


@dataclass(frozen=True)
class Completion:
    """One model reply with token usage.

    Attributes:
        text: Reply content.
        prompt_tokens: Input tokens billed.
        completion_tokens: Output tokens billed.
    """

    text: str
    prompt_tokens: int
    completion_tokens: int


def chat(
    client: Mistral,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 8192,  # room for reasoning tokens before the answer
    retries: int = 6,
) -> Completion:
    """Call Mistral chat completion with retry on transient errors.

    A rate-limit error (HTTP 429) waits a full minute, because limits are per minute.
    Other client errors (4xx) are raised at once. Anything else backs off 1, 2, 4, ... seconds.

    Args:
        client: A ``mistralai.client.Mistral`` instance.
        model: Pinned model name, e.g. ``ministral-8b-2512``.
        messages: Chat messages.
        temperature: Sampling temperature (0 for repeatable runs).
        max_tokens: Cap on output tokens.
        retries: Attempts before giving up.

    Returns:
        The reply text and token counts.

    Raises:
        RuntimeError: If every attempt failed.
    """
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.complete(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=1.0,  # some models (Large 4) require top_p=1 with temperature 0
                max_tokens=max_tokens,
            )
            usage = resp.usage
            return Completion(
                text=content_text(resp.choices[0].message.content),
                prompt_tokens=usage.prompt_tokens or 0,
                completion_tokens=usage.completion_tokens or 0,
            )
        except Exception as e:  # SDK raises several error types; retry transient ones
            status = getattr(e, "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                raise  # a bad request won't get better by retrying
            last = e
            wait = 60 if "429" in str(e) else 2**attempt
            logger.warning("chat call failed (%.80s); retry %d in %ds", e, attempt + 1, wait)
            time.sleep(wait)
    raise RuntimeError(f"chat call failed after {retries} attempts: {last}")
