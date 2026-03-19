import pandas as pd
from typing import Any

def group_answers_by_question(df: pd.DataFrame) -> dict[int, dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}

    # Use answer_label when available (resolved text), fall back to raw answer
    answer_col = "answer_label" if "answer_label" in df.columns else "answer"

    for question_id, group in df.groupby("question_id"):
        grouped[int(question_id)] = {
            "text":    group["question_ar"].iloc[0],
            "type":    group["question_type"].iloc[0],
            "answers": group[answer_col].dropna().tolist(),
        }

    return grouped


def get_distinct_answers_per_question(grouped: dict[int, dict[str, Any]]) -> dict[int, list[str]]:
    distinct: dict[int, list[str]] = {}

    for qid, info in grouped.items():
        unique_answers = sorted(set(str(a) for a in info["answers"] if a))
        distinct[qid] = unique_answers
        print(f"   Q{qid} ({info['text'][:40]}...): {len(unique_answers)} distinct answers")
        
    return distinct


def get_sample_answers_per_question(df: pd.DataFrame, n: int = 3) -> dict[int, list[str]]:
    """Returns up to `n` full row examples (all columns) per question for agent context."""
    samples: dict[int, list[str]] = {}

    for qid, group in df.groupby("question_id"):
        rows = group.head(n).to_dict(orient="records")
        samples[int(qid)] = [str(row) for row in rows]

    return samples
