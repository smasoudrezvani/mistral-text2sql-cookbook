# Literature review: fixing "dirty data" failures in text-to-SQL

*Compiled 2026-11-09 from four research reports (raw notes kept privately in `research.txt`),
with the two papers the decision rests on re-checked against the papers themselves.*

## The question

Our best system (Mistral Large 4 + execute-and-retry) gets **65.0%** on BIRD mini-dev (500
questions). In 50 labelled failures, **21 were dirty data**: 0 / '' / '-' meaning "unknown",
numbers stored as text, codes with leading zeros, lists packed into one field, placeholders like
'No Colour' (see [error_examples.md](error_examples.md)).

Our prompt has: `CREATE TABLE` statements + 3 example rows per table + **BIRD's per-question hint**
+ the question. It does **not** have BIRD's column descriptions, which already contain warnings
such as *"If the weight_kg is NULL or 0, it means the weight of the superhero is missing."*

**Is there published evidence for a cheap fix (no fine-tuning, closed API model)?**

## Bottom line

1. **Yes, there is solid evidence** that putting column descriptions and data profiles in the
   prompt helps on BIRD, from clean same-setup ablations (two papers verified below).
2. **But the gain is much smaller when the per-question hints are already in the prompt, which is
   our case.** The biggest published gains (+10 to +13 points) are *without* hints.
3. **The format of the notes matters.** Value patterns, ranges and missing-value rules written as
   instructions help; raw statistics (e.g. null counts) can hurt.
4. **Blanket self-correction ("check your query") can break correct answers**; only targeted
   checks that fire on a concrete problem are supported.
5. **Our own null result for value retrieval matches the literature**: retrieval fixes spelling,
   but most of our failures are about what values *mean*.

## Key evidence

### Verified against the papers (2026-11-09)

**Automatic Metadata Extraction for Text-to-SQL** — Shkapenyuk, Srivastava, Johnson, Ghane (AT&T), 2025,
[arXiv 2505.19988](https://arxiv.org/abs/2505.19988). Table 1, GPT-4o, BIRD mini-dev (500 questions),
same pipeline, only the metadata changes:

| Metadata in the prompt | No hints | With hints |
|---|---|---|
| None | 49.8 | *not reported* |
| BIRD descriptions | 59.6 | 72.0 |
| Automatic profiling (LLM-summarised column stats) | 61.2 | 72.6 |
| Both fused | 63.2 | 73.0 |

- Without hints, descriptions or profiles add **+10 to +13 points**.
- With hints, profiling adds only **+0.6 to +1.0 over BIRD descriptions**. The paper does not report
  "no descriptions, with hints", so the gain of descriptions *on top of hints* is unknown. The
  authors write that "the most powerful metadata available in Bird are the hints".
- Their profile: null counts, distinct count, min/max, length, "looks like a number", common
  prefixes, top values, turned into an English description by an LLM (one call per column, once).

**Synthetic SQL Column Descriptions and Their Impact on Text-to-SQL Performance** — Wretblad et al.,
2024, [arXiv 2408.04691](https://arxiv.org/abs/2408.04691). Table 3, zero-shot, full BIRD dev (1,534),
**no hints in the prompt** (template has only schema + question):

| Model | No descriptions | BIRD descriptions | Gold (human-fixed) | LLM-generated |
|---|---|---|---|---|
| GPT-4o | 30.13 | 31.34 | 33.28 | 36.78 |
| Mixtral 8x22B | 17.50 | 25.43 | 27.16 | 29.59 |
| Mistral 7B | 14.73 | 15.06 | 16.49 | 16.90 |

- BIRD's own descriptions help little for GPT-4o (+1.2) but a lot for Mixtral (+7.9).
- Richer, LLM-written descriptions beat the original ones.
- **Small models gain little** (Mistral 7B: +0.3 to +2.2), which matters for Ministral 8B.

### Reported by the research sources, not re-checked by us

| Paper | Finding | Note |
|---|---|---|
| BIRD (Li et al., 2023, [2305.03111](https://arxiv.org/abs/2305.03111)) | Per-question evidence lifts GPT-4 from 30.90 to 46.35 on dev | Evidence = the hints we already use |
| MCI-SQL (Wang et al., 2026, [2603.13390](https://arxiv.org/abs/2603.13390)) | Removing value-pattern + range notes: 67.5 -> 62.0; *adding* raw null counts: 67.5 -> 65.0 | Preprint, not peer-reviewed |
| XiYan-SQL M-Schema (Gao et al., 2024, [2411.08599](https://arxiv.org/abs/2411.08599)) | Column list with type, keys, description, example values instead of DDL: +2.03 avg over 4 models | Formatting only |
| REDSQL (Ren et al., PVLDB 2025) | Profile-based column documentation alone: +2.3 to +3.8; inside the full system only -0.3 to -0.5 when removed | gpt-3.5 refiner |
| CHASE-SQL (Pourreza et al., ICLR 2025, [2410.01943](https://arxiv.org/abs/2410.01943)) | Removing the query fixer: -3.78; removing LSH value retrieval: -2.92 | 21 candidates + fine-tuned selector |
| MAGIC (Askari et al., AAAI 2025, [2406.12692](https://arxiv.org/abs/2406.12692)) | A correction guideline distilled from training failures: DIN-SQL 56.52 -> 59.13 | Closest to "turn our 50 labels into rules" |
| SQLens (Gong et al., 2025, [2506.04494](https://arxiv.org/abs/2506.04494)) | Plain LLM self-reflection "breaks the most correct queries" | Supports targeted checks only |
| Distyl (Maamari et al., 2024, [2408.07702](https://arxiv.org/abs/2408.07702)) | Augmentation helped ft:GPT-4o (+2.7), not Gemini (0), slightly hurt Llama 405B | ~147-question subsample |
| Annotation audit (Jin et al., 2026, [2601.08778](https://arxiv.org/abs/2601.08778)) | 52.8% of mini-dev **v1** had annotation errors | Our 2026 release was corrected |

## Checks against our own data

- **Are our 21 dirty-data failures winnable?** One source warned that BIRD gold SQL sometimes
  ignores dirty values itself. In all 21 of our cases, the gold SQL *does* handle the trap
  (`weight_kg > 0`, `TRIM(...) <> ''`, `CAST(... AS REAL)`, `<> '-'`, splitting lists), so a model
  that respects the data would be scored correct.
- **`column_meaning.json`** (an LLM-condensed description file mentioned by one source) is not in the
  mini-dev GitHub repo or on Hugging Face. We have BIRD's `database_description` CSVs, which is enough.

## How reliable were the four sources?

| Source | Reliability | Why |
|---|---|---|
| 4 | High | Cites every number; flags preprints and uncertain figures; its key tables matched the papers |
| 2 | Good | Consistent with source 4; fewer details |
| 3 | Mixed | Mostly right, but says "no paper reports a gain from profiling", which AT&T contradicts |
| 1 | Low | Few citations, an unnamed "Refined BIRD" paper, an uncited 8.3% -> 78.3% claim, and inflated expected gains (+5 to +10 points) |

## Verdict for this project

| Idea | Evidence | Expected gain for us (hints already in prompt) | Cost | Test it? |
|---|---|---|---|---|
| **Column notes**: BIRD descriptions + value descriptions + a short profile line per risky column, written as instructions ("0 means missing; exclude from AVG"; "numbers stored as text: CAST") | Strong without hints, small with hints | Small: roughly +1 to +3 points (our estimate, not a published number); likely least for Ministral 8B | No extra calls; longer prompts; ~$3.5 to test | **Yes, one experiment** |
| Targeted checklist self-check (fires only on a concrete issue) | Some | +1 to +3 points | +1 call per question (doubles Large 4 cost) | Not now (budget) |
| Better value retrieval | Some, but our test showed none | ~0 | Index | No |
| Schema linking / pruning | Negative for strong models | <= 0 | Several calls | No |
| Fine-tuning, multi-candidate selectors | Strong but out of scope | n/a | High | No |

**Recommendation:** run one final, cheap experiment, "column notes", built offline and switched
on with a flag, on the same questions as before (Ministral and Small on 500, Large on the
200-question slice). Either result is useful for the notebook: a gain shows the fix, and no gain
shows that the hints already carry most of this information, which is what the AT&T table hints at.

## Update: what we measured (2026-11-09)

We ran the recommended experiment. With BIRD's hints already in the prompt, adding column notes
(BIRD descriptions + an automatic data-trap profile) to approach (a) gave **+5.8 points
(Ministral 8B), +12.2 (Small 4) and +10.5 (Large 4, 200 questions)**, all statistically clear
(McNemar p <= 0.01). This is much more than the +1 to +3 points we expected from the literature,
which never reported the "no descriptions, with hints" baseline. See `results/summary/*_notes.json`.
