from types import SimpleNamespace

import pytest

import text2sql_eval.prompting as prompting
from text2sql_eval.prompting import chat


class FakeError(Exception):
    def __init__(self, status_code):
        super().__init__(f"Status {status_code}")
        self.status_code = status_code


def _reply(text="```sql\nSELECT 1\n```"):
    msg = SimpleNamespace(content=text)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=3)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.chat = SimpleNamespace(complete=self._complete)

    def _complete(self, **kwargs):
        self.calls.append(kwargs)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(prompting.time, "sleep", lambda s: None)


def test_sends_top_p_one_and_returns_usage():
    client = FakeClient([_reply()])
    c = chat(client, "m", [{"role": "user", "content": "x"}])
    assert client.calls[0]["top_p"] == 1.0
    assert client.calls[0]["temperature"] == 0.0
    assert (c.prompt_tokens, c.completion_tokens) == (10, 3)


def test_bad_request_is_not_retried():
    client = FakeClient([FakeError(400), _reply()])
    with pytest.raises(FakeError):
        chat(client, "m", [])
    assert len(client.calls) == 1


def test_rate_limit_and_server_errors_are_retried():
    client = FakeClient([FakeError(429), FakeError(503), _reply()])
    c = chat(client, "m", [])
    assert len(client.calls) == 3
    assert c.text.startswith("```sql")


def test_reasoning_chunks_are_dropped():
    think = SimpleNamespace(type="thinking", thinking=[SimpleNamespace(text="hmm")])
    text = SimpleNamespace(type="text", text="```sql\nSELECT 2\n```")
    client = FakeClient([_reply([think, text])])
    assert chat(client, "m", []).text == "```sql\nSELECT 2\n```"


def test_content_text_variants():
    assert prompting.content_text(None) == ""
    assert prompting.content_text("abc") == "abc"
    assert prompting.content_text([]) == ""


def test_gives_up_after_retries():
    client = FakeClient([FakeError(503)] * 3)
    with pytest.raises(RuntimeError):
        chat(client, "m", [], retries=3)
