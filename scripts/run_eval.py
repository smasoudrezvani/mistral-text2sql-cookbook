"""Run one approach with one model on a slice of BIRD mini-dev.

Approaches:
    full_schema  (a) all tables + 3 example rows in the prompt
    retrieval    (b) only the top-k tables by embedding similarity, plus FK-linked tables
    retry        (c) full schema, then run the SQL and retry with the error / empty-result feedback
    vote         full schema, sample N answers (temperature > 0), keep the most common result

Resumable: per-question results are appended to results/raw/<approach>_<model>_<slice>.jsonl
and questions already in that file are skipped. The summary is rebuilt from the whole file.

Usage:
    uv run python scripts/run_eval.py --approach full_schema --model ministral-8b-2512
    uv run python scripts/run_eval.py --approach retrieval --model mistral-large-4 --slice all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from mistralai.client import Mistral

from text2sql_eval.agent import solve_with_retry
from text2sql_eval.bird import Question, db_path, load_gold_cache, load_questions
from text2sql_eval.distractors import load_manifest
from text2sql_eval.notes import load_notes
from text2sql_eval.prompting import build_messages, chat, extract_sql
from text2sql_eval.retrieval import (
    embed,
    foreign_key_neighbors,
    select_tables,
    table_docs,
    tables_in_sql,
)
from text2sql_eval.schema import render_schema, table_names
from text2sql_eval.scorer import ExecResult, compare, execute_sql
from text2sql_eval.values import ValueIndex, format_hits
from text2sql_eval.voting import majority_vote

ROOT = Path(__file__).resolve().parents[1]
JSONL = ROOT / "data/raw/mini_dev_sqlite.jsonl"
DB_ROOT = ROOT / "data/raw/minidev/MINIDEV/dev_databases"
GOLD = ROOT / "data/cache/gold_results.json"
EMB_CACHE = ROOT / "data/cache/embeddings.json"
DISTRACT = ROOT / "data/cache/distractors"
VALUES = ROOT / "data/cache/values"
NOTES = ROOT / "data/cache/notes"
SLICES = ROOT / "slices"
RAW_OUT = ROOT / "results/raw"
SUMMARY_OUT = ROOT / "results/summary"

# $ per 1M tokens (input, output), docs.mistral.ai/inference/pricing, checked 2026-11-08
PRICES = {
    "ministral-8b-2512": (0.15, 0.15),
    "mistral-small-2603": (0.15, 0.60),
    "mistral-large-4": (0.70, 2.10),
}

logger = logging.getLogger("run_eval")


def load_embeddings(
    client: Mistral, questions: list[Question], extra_db_ids: set[str] | None = None
) -> dict[str, dict]:
    """Embed every table description and question once, caching to disk.

    Args:
        client: Mistral client.
        questions: Questions that need a vector.
        extra_db_ids: More databases whose tables need vectors (distractor sources).

    Returns:
        ``{"tables": {db: {table: vec}}, "questions": {qid: vec}}``.
    """
    cache = (
        json.loads(EMB_CACHE.read_text(encoding="utf-8"))
        if EMB_CACHE.exists()
        else {"tables": {}, "questions": {}}
    )
    wanted = {q.db_id for q in questions} | (extra_db_ids or set())
    for db_id in sorted(wanted - set(cache["tables"])):
        docs = table_docs(db_path(DB_ROOT, db_id), DB_ROOT / db_id / "database_description")
        vecs = embed(client, list(docs.values()))
        cache["tables"][db_id] = dict(zip(docs, vecs, strict=True))
        logger.info("embedded %d tables of %s", len(docs), db_id)
    todo = [q for q in questions if str(q.question_id) not in cache["questions"]]
    if todo:
        texts = [f"{q.question}\nHint: {q.evidence}" if q.evidence else q.question for q in todo]
        for q, v in zip(todo, embed(client, texts), strict=True):
            cache["questions"][str(q.question_id)] = v
        logger.info("embedded %d questions", len(todo))
    EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    EMB_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def summarize(raw_path: Path, approach: str, model: str, slice_name: str) -> dict[str, object]:
    """Build the run summary from all per-question records in a raw results file.

    Args:
        raw_path: The run's JSONL file.
        approach: Approach name.
        model: Model name (for pricing).
        slice_name: Slice name.

    Returns:
        Accuracy under both rules, failures, tokens, cost, and retrieval stats if present.
    """
    recs = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    n = len(recs)
    tok_in = sum(r["prompt_tokens"] for r in recs)
    tok_out = sum(r["completion_tokens"] for r in recs)
    p_in, p_out = PRICES[model]
    summary: dict[str, object] = {
        "run": raw_path.stem,
        "approach": approach,
        "model": model,
        "slice": slice_name,
        "n": n,
        "ex_bird": round(sum(r["bird"] for r in recs) / n, 4),
        "ex_strict": round(sum(r["strict"] for r in recs) / n, 4),
        "failed_to_run": sum(r["pred_error"] is not None for r in recs),
        "prompt_tokens": tok_in,
        "completion_tokens": tok_out,
        "cost_usd": round(tok_in / 1e6 * p_in + tok_out / 1e6 * p_out, 4),
    }
    if recs and "attempts" in recs[0]:
        summary["first_try_ex_bird"] = round(sum(r["first_bird"] for r in recs) / n, 4)
        summary["first_try_failed_to_run"] = sum(r["first_error"] is not None for r in recs)
        summary["avg_attempts"] = round(sum(r["attempts"] for r in recs) / n, 2)
        summary["retried"] = sum(r["attempts"] > 1 for r in recs)
        summary["fixed_by_retry"] = sum(r["bird"] and not r["first_bird"] for r in recs)
        summary["broken_by_retry"] = sum(r["first_bird"] and not r["bird"] for r in recs)
    if recs and "sample_bird" in recs[0]:
        summary["first_sample_ex_bird"] = round(sum(r["first_bird"] for r in recs) / n, 4)
        summary["any_of_samples_ex_bird"] = round(sum(r["any_bird"] for r in recs) / n, 4)
        summary["avg_distinct_results"] = round(sum(r["n_groups"] for r in recs) / n, 2)
        summary["no_sample_ran"] = sum(r["n_ok"] == 0 for r in recs)
    if recs and "value_hits" in recs[0]:
        summary["avg_value_hits"] = round(sum(len(r["value_hits"]) for r in recs) / n, 2)
        summary["value_used"] = sum(r["value_used"] for r in recs)
    if recs and "used_distractor" in recs[0]:
        summary["used_distractor"] = sum(r["used_distractor"] for r in recs)
    if recs and "table_recall" in recs[0]:
        summary["table_recall"] = round(sum(r["table_recall"] for r in recs) / n, 4)
        summary["avg_tables_kept"] = round(sum(len(r["tables"]) for r in recs) / n, 2)
        summary["avg_tables_total"] = round(sum(r["n_tables_total"] for r in recs) / n, 2)
        if "n_distractors_kept" in recs[0]:
            summary["avg_distractors_kept"] = round(
                sum(r["n_distractors_kept"] for r in recs) / n, 2
            )
    return summary


def main() -> None:
    """Run an approach on a slice, append per-question results, write a summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--approach", required=True, choices=["full_schema", "retrieval", "retry", "vote"]
    )
    parser.add_argument(
        "--max-attempts", type=int, default=3, help="model calls per question (retry)"
    )
    parser.add_argument("--samples", type=int, default=5, help="answers per question (vote)")
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="sampling temperature (vote)"
    )
    parser.add_argument("--model", required=True, choices=sorted(PRICES))
    parser.add_argument("--slice", default="dev50", help="'all' or a file name in slices/")
    parser.add_argument("--n-rows", type=int, default=3, help="example rows per table")
    parser.add_argument("--k", type=int, default=5, help="tables kept by similarity (retrieval)")
    parser.add_argument("--rpm", type=float, default=0, help="max requests per minute (0 = no cap)")
    parser.add_argument("--workers", type=int, default=1, help="questions answered in parallel")
    parser.add_argument(
        "--distractors",
        action="store_true",
        help="add 30 distractor tables per database (run scripts/build_distractors.py first)",
    )
    parser.add_argument(
        "--notes",
        action="store_true",
        help="add column notes: BIRD descriptions + data profile (run scripts/build_notes.py)",
    )
    parser.add_argument(
        "--values",
        action="store_true",
        help="add matching stored values to the prompt (run scripts/build_value_index.py first)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_dotenv(ROOT / ".env")
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    questions = load_questions(JSONL)
    if args.slice != "all":
        ids = set(json.loads((SLICES / f"{args.slice}.json").read_text())["question_ids"])
        questions = [q for q in questions if q.question_id in ids]
    gold = load_gold_cache(GOLD)
    manifests: dict[str, dict[str, str]] = {}
    if args.distractors:
        manifests = {
            db_id: load_manifest(DISTRACT / f"{db_id}.sqlite")
            for db_id in {q.db_id for q in questions}
        }
    sources = {src for m in manifests.values() for src in m.values()}
    emb = load_embeddings(client, questions, sources) if args.approach == "retrieval" else None
    schemas: dict[tuple[str, tuple[str, ...] | None], str] = {}
    neighbors: dict[str, dict[str, set[str]]] = {}

    RAW_OUT.mkdir(parents=True, exist_ok=True)
    suffix = (
        ("_distract" if args.distractors else "")
        + ("_values" if args.values else "")
        + ("_notes" if args.notes else "")
    )
    col_notes: dict[str, dict[str, dict[str, str]]] = {}
    if args.notes:
        col_notes = {d: load_notes(NOTES / f"{d}.json") for d in {q.db_id for q in questions}}
    value_idx: dict[str, ValueIndex] = {}
    if args.values:
        value_idx = {d: ValueIndex.load(VALUES / f"{d}.json") for d in {q.db_id for q in questions}}
    raw_path = RAW_OUT / f"{args.approach}_{args.model}_{args.slice}{suffix}.jsonl"
    done: set[int] = set()
    if raw_path.exists():
        done = {json.loads(x)["question_id"] for x in raw_path.read_text().splitlines()}
        logger.info("resuming: %d questions already done", len(done))
    todo = [q for q in questions if q.question_id not in done]
    min_gap = 60.0 / args.rpm if args.rpm else 0.0
    pace_lock = threading.Lock()
    last_call = [0.0]

    def pace() -> None:
        """Keep at least min_gap seconds between API calls (only used with --rpm)."""
        if not min_gap:
            return
        with pace_lock:
            wait = last_call[0] + min_gap - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            last_call[0] = time.monotonic()

    def solve(q: Question) -> dict[str, object]:
        """Answer and score one question; returns its result record."""
        db = db_path(DB_ROOT, q.db_id)
        extra: dict[str, object] = {}
        picked: tuple[str, ...] | None = None
        extra_dbs = [DISTRACT / f"{q.db_id}.sqlite"] if args.distractors else None
        fakes = manifests.get(q.db_id, {})
        if emb is not None:
            table_vecs = dict(emb["tables"][q.db_id])
            table_vecs.update({t: emb["tables"][src][t] for t, src in fakes.items()})
            if q.db_id not in neighbors:
                neighbors[q.db_id] = foreign_key_neighbors(db)
            picked = tuple(
                select_tables(
                    emb["questions"][str(q.question_id)],
                    table_vecs,
                    neighbors[q.db_id],
                    k=args.k,
                )
            )
            all_tables = table_names(db, extra_dbs)
            gold_tables = tables_in_sql(q.sql, all_tables)
            extra = {
                "tables": list(picked),
                "gold_tables": sorted(gold_tables),
                "table_recall": gold_tables <= set(picked),
                "n_tables_total": len(all_tables),
                "n_distractors_kept": sum(t in fakes for t in picked),
            }
        key = (q.db_id, picked)
        if key not in schemas:
            schemas[key] = render_schema(
                db,
                n_rows=args.n_rows,
                tables=list(picked) if picked else None,
                extra_dbs=extra_dbs,
                shuffle_seed=0 if extra_dbs and picked is None else None,
                notes=col_notes.get(q.db_id),
            )
        hits = value_idx[q.db_id].search(f"{q.question}\n{q.evidence}") if value_idx else []
        messages = build_messages(schemas[key], q.question, q.evidence, format_hits(hits))
        g = gold[q.question_id]
        pace()

        if args.approach == "retry":
            agent = solve_with_retry(
                client,
                args.model,
                messages,
                db,
                timeout_s=g.pred_timeout_s,
                max_attempts=args.max_attempts,
                attach=extra_dbs,
            )
            sql, pred = agent.sql, agent.result
            tok_in, tok_out = agent.prompt_tokens, agent.completion_tokens
            first = compare(agent.first_result, g.as_exec_result(), q.sql)
            extra = {
                "attempts": agent.attempts,
                "feedback": agent.feedback,
                "first_sql": agent.first_sql,
                "first_error": agent.first_result.error,
                "first_bird": first.bird,
                "first_strict": first.strict,
            }
        elif args.approach == "vote":
            reply = chat(client, args.model, messages, temperature=args.temperature, n=args.samples)
            sqls = [extract_sql(x) for x in reply.texts]
            runs = [
                execute_sql(db, s, timeout_s=g.pred_timeout_s, attach=extra_dbs)
                if s
                else ExecResult(rows=None, error="no SQL found in the reply")
                for s in sqls
            ]
            vote = majority_vote(runs)
            sql, pred = sqls[vote.winner], runs[vote.winner]
            tok_in, tok_out = reply.prompt_tokens, reply.completion_tokens
            sample_bird = [compare(r, g.as_exec_result(), q.sql).bird for r in runs]
            extra.update(
                {
                    "votes": vote.votes,
                    "n_ok": vote.n_ok,
                    "n_groups": vote.n_groups,
                    "sample_bird": sample_bird,
                    "first_bird": sample_bird[0],
                    "any_bird": any(sample_bird),
                }
            )
        else:
            reply = chat(client, args.model, messages)
            sql = extract_sql(reply.text)
            pred = execute_sql(db, sql, timeout_s=g.pred_timeout_s, attach=extra_dbs)
            tok_in, tok_out = reply.prompt_tokens, reply.completion_tokens
            extra["finish_reason"] = reply.finish_reason

        score = compare(pred, g.as_exec_result(), q.sql)
        if value_idx:
            low = sql.lower()
            extra["value_hits"] = [f"{h.table}.{h.column}={h.value}" for h in hits]
            extra["value_used"] = any(f"'{h.value.lower()}'" in low for h in hits)
        if fakes:
            extra["used_distractor"] = bool(tables_in_sql(sql, list(fakes)))
        return {
            "question_id": q.question_id,
            "db_id": q.db_id,
            "difficulty": q.difficulty,
            "pred_sql": sql,
            "pred_error": pred.error,
            "pred_rows": len(pred.rows) if pred.rows is not None else None,
            "bird": score.bird,
            "strict": score.strict,
            "prompt_tokens": tok_in,
            "completion_tokens": tok_out,
            **extra,
        }

    write_lock = threading.Lock()
    n_done = [0]
    n_failed = [0]
    with raw_path.open("a", encoding="utf-8") as out:

        def record_result(q: Question, fut: Future[dict[str, object]]) -> None:
            try:
                rec = fut.result()
            except Exception as e:  # one bad question must not stop the run; resume retries it
                n_failed[0] += 1
                logger.error("q%d failed, will be retried on resume: %.200s", q.question_id, e)
                return
            with write_lock:
                out.write(json.dumps(rec) + "\n")
                out.flush()
                n_done[0] += 1
                logger.info(
                    "%d/%d q%d bird=%s strict=%s%s",
                    n_done[0],
                    len(todo),
                    q.question_id,
                    rec["bird"],
                    rec["strict"],
                    f" error={str(rec['pred_error'])[:60]}" if rec["pred_error"] else "",
                )

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(solve, q): q for q in todo}
            for fut in as_completed(futures):
                record_result(futures[fut], fut)

    if n_failed[0]:
        logger.warning("%d questions failed; run the same command again to retry them", n_failed[0])

    summary = summarize(raw_path, args.approach, args.model, args.slice)
    SUMMARY_OUT.mkdir(parents=True, exist_ok=True)
    (SUMMARY_OUT / f"{raw_path.stem}.json").write_text(json.dumps(summary, indent=1) + "\n")
    logger.info("summary: %s", summary)


if __name__ == "__main__":
    main()
