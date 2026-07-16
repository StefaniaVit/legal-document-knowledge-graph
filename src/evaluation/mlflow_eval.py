"""
Computes precision metrics from the hand-labeled sample (data/labeled/) and
logs them to MLflow.

The labels themselves are AI-assisted review, not independent human-validated
ground truth: ~46% (32/69) were judged by Gemini (gemini-2.5-flash /
gemini-2.5-flash-lite, both free tier) and adjudicated by Claude afterward --
several of Gemini's own judgments were self-contradictory (marking an entity
"incorrect" while recommending the identical type back) or inconsistent with
this project's own established conventions (e.g. downgrading "supervisory
authority" from ORGANIZATION, which the graph's own hub-entity analysis
confirmed earlier). The remaining ~54% (37/69), including every relation and
every Data Governance Act entity, were reviewed directly by Claude with full
project context after Gemini's free-tier daily quota (20 requests/day on this
project, confirmed on two different model variants) was exhausted. See
CLAUDE.md for the full methodology note. Report results as "AI-assisted
review," not "human-validated," if this evaluation is described elsewhere.
"""
import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional

import mlflow

LABELED_DIR = Path(__file__).parents[2] / "data" / "labeled"
ENTITIES_PATH = LABELED_DIR / "entities_review.csv"
RELATIONS_PATH = LABELED_DIR / "relations_review.csv"

MLFLOW_EXPERIMENT = "legal-kg-extraction-eval"


def load_entity_rows() -> list[dict]:
    return list(csv.DictReader(ENTITIES_PATH.open(encoding="utf-8")))


def load_relation_rows() -> list[dict]:
    return list(csv.DictReader(RELATIONS_PATH.open(encoding="utf-8")))


def compute_metrics(entity_rows: list[dict], relation_rows: list[dict]) -> dict:
    n_entities = len(entity_rows)
    n_entities_correct = sum(1 for r in entity_rows if r["is_correct(Y/N)"] == "Y")

    n_relations = len(relation_rows)
    n_relations_correct = sum(1 for r in relation_rows if r["is_correct(Y/N)"] == "Y")

    by_type: dict = defaultdict(lambda: [0, 0])
    for r in entity_rows:
        by_type[r["entity_type"]][1] += 1
        if r["is_correct(Y/N)"] == "Y":
            by_type[r["entity_type"]][0] += 1

    metrics = {
        "entity_precision": n_entities_correct / n_entities,
        "entity_sample_size": n_entities,
        "relation_precision": n_relations_correct / n_relations,
        "relation_sample_size": n_relations,
    }
    for entity_type, (correct, total) in by_type.items():
        metrics[f"entity_precision_{entity_type.lower()}"] = correct / total
        metrics[f"entity_count_{entity_type.lower()}"] = total
    return metrics


def build_summary_report(entity_rows: list[dict], relation_rows: list[dict], metrics: dict) -> str:
    lines = [
        "# Extraction Quality Evaluation",
        "",
        "AI-assisted review (Gemini + Claude), not independent human-validated ground truth.",
        "See mlflow_eval.py module docstring for full methodology.",
        "",
        f"## Overall",
        f"- Entity precision: {metrics['entity_precision']:.1%} ({metrics['entity_sample_size']} sampled)",
        f"- Relation precision: {metrics['relation_precision']:.1%} ({metrics['relation_sample_size']} sampled)",
        "",
        "## Entity precision by type",
    ]
    by_type: dict = defaultdict(lambda: [0, 0])
    for r in entity_rows:
        by_type[r["entity_type"]][1] += 1
        if r["is_correct(Y/N)"] == "Y":
            by_type[r["entity_type"]][0] += 1
    for entity_type, (correct, total) in sorted(by_type.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        lines.append(f"- {entity_type}: {correct}/{total} = {correct/total:.0%}")

    predicted_relation_precision = metrics["entity_precision"] ** 2
    actual_relation_precision = metrics["relation_precision"]
    gap = abs(predicted_relation_precision - actual_relation_precision)
    closeness = "close to" if gap < 0.15 else "notably different from"
    small_sample_note = (
        f" (small-sample caveat: only {metrics['relation_sample_size']} relations were sampled here, "
        "so this comparison isn't very reliable)"
        if metrics["relation_sample_size"] < 20 else ""
    )
    lines += [
        "",
        "## Why relations tend to score lower than entities",
        "A relation requires both its subject and object entities to be correct, plus the",
        "connecting claim itself. If entity correctness is ~independent per mention, the naive",
        "expectation is that the chance both endpoints of a relation are correct is roughly "
        f"{metrics['entity_precision']:.0%} x {metrics['entity_precision']:.0%} = "
        f"{predicted_relation_precision:.0%}. This run's actual relation precision "
        f"({actual_relation_precision:.0%}) is {closeness} that naive estimate{small_sample_note}.",
        "Beyond whatever this run's entity-noise math suggests, relation errors also include",
        "genuine direction/type confusion even with clean entities (e.g. a real connection "
        "described with the wrong relation type, or a relation stated backwards relative to the",
        "source text) and occasional fabrication not actually stated in the source text.",
        "",
        "## Known entity error patterns (qualitative, accumulated across evaluation rounds --",
        "## not all necessarily present in this specific run's sample)",
        "- Type-boundary confusion: LEGAL_ACT / LEGAL_CONCEPT / ORGANIZATION mixed up, especially",
        "  for mechanisms/frameworks (e.g. 'binding corporate rules', 'cybersecurity certification",
        "  scheme') mistagged as specific enacted laws.",
        "- Span garbling: multiple distinct items merged into one entity string.",
        "- Truncated citation fragments: e.g. '(EU) 2016/679' missing the leading 'Regulation'.",
        "- Generic self-references: bare 'Directive'/'Regulation' with no specific identification",
        "  (also handled downstream in neo4j_loader.py's self-reference filter, which only",
        "  catches the 'this/the X' form, not the bare word).",
    ]
    return "\n".join(lines)


def log_run(run_name: Optional[str] = None) -> dict:
    entity_rows = load_entity_rows()
    relation_rows = load_relation_rows()
    metrics = compute_metrics(entity_rows, relation_rows)
    summary = build_summary_report(entity_rows, relation_rows, metrics)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "extraction_model": "qwen2.5-3b-instruct-q4_k_m",
            "review_method": "gemini_assisted_plus_claude_adjudication",
            "documents_covered": 4,
        })
        mlflow.log_metrics(metrics)
        mlflow.log_text(summary, "evaluation_summary.md")
        mlflow.log_artifact(str(ENTITIES_PATH))
        mlflow.log_artifact(str(RELATIONS_PATH))

    print(summary)
    return metrics


if __name__ == "__main__":
    log_run()
