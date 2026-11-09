from types import SimpleNamespace

from text2sql_eval.prompting import chat
from text2sql_eval.scorer import ExecResult
from text2sql_eval.voting import majority_vote


def ok(*rows):
    return ExecResult(rows=list(rows))


ERR = ExecResult(rows=None, error="no such column: x")


def test_most_common_result_wins():
    v = majority_vote([ok((42,)), ok((40,)), ok((42,)), ERR, ok((42,))])
    assert (v.winner, v.votes, v.n_ok, v.n_groups) == (0, 3, 4, 2)


def test_same_rows_in_other_order_are_one_group():
    v = majority_vote([ok(("a",), ("b",)), ok(("b",), ("a",)), ok(("c",))])
    assert v.winner == 0 and v.votes == 2


def test_errors_get_no_vote_and_winner_can_be_later_sample():
    v = majority_vote([ERR, ERR, ok((1,)), ok((2,)), ok((2,))])
    assert v.winner == 3 and v.votes == 2


def test_tie_goes_to_first_group():
    v = majority_vote([ok((1,)), ok((2,)), ok((2,)), ok((1,))])
    assert v.winner == 0 and v.votes == 2


def test_all_failed():
    v = majority_vote([ERR, ERR])
    assert (v.winner, v.votes, v.n_ok) == (0, 0, 0)


def test_chat_returns_all_samples():
    choices = [
        SimpleNamespace(
            message=SimpleNamespace(content=f"```sql\nSELECT {i}\n```"), finish_reason="stop"
        )
        for i in range(3)
    ]
    resp = SimpleNamespace(
        choices=choices, usage=SimpleNamespace(prompt_tokens=10, completion_tokens=9)
    )
    sent = {}

    def complete(**kw):
        sent.update(kw)
        return resp

    c = chat(
        SimpleNamespace(chat=SimpleNamespace(complete=complete)), "m", [], temperature=0.7, n=3
    )
    assert sent["n"] == 3 and sent["temperature"] == 0.7
    assert len(c.texts) == 3 and c.text == c.texts[0]
