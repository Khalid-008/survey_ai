"""
text_pipeline.py
================
Hybrid per-question + configurable batch text analysis.

Flow per question:
  Stage 1: For each batch of answers → LLM tags answers + produces batch summary
  Stage 2: After all batches → final LLM synthesises batch summaries into a report

Returns:
  enriched_df   : original df with added columns (sentiment, entities, topics)
  result_dict   : {question_id: final_report_dict}
"""

import json
import math
import traceback

import pandas as pd

from llms.models import model

from agents.prompt.text_analysis_batch_prompt import text_analysis_batch_prompt
from agents.prompt.text_analysis_synthesis_prompt import text_analysis_synthesis_prompt


# ────────────────────────────────────────────────────────────────────────────


def _parse_json_response(raw: str, context: str = "") -> dict | None:
    """
    Try to parse a JSON object from the LLM response.
    Handles cases where the model wraps the JSON in markdown fences.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first and last fence lines
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"   ⚠️  JSON parse error [{context}]: {e}")
        return None


# ────────────────────────────────────────────────────────────────────────────


def _run_batch_stage1(
    question_id: int,
    question_ar: str,
    batch_records: list[dict],
    batch_idx: int,
    total_batches: int,
) -> tuple[list[dict], dict | None]:
    """
    Call LLM Stage 1 for a single batch.

    Returns:
        answers_analysis : list of per-answer dicts (may be empty on failure)
        batch_summary    : {top_topics, top_entities, summary} or None on failure
    """
    label = f"Q{question_id} | Batch {batch_idx}/{total_batches}"

    answers_payload = [
        {"answer_id": int(r["answer_id"]), "answer": str(r.get("answer", "") or "")}
        for r in batch_records
    ]

    try:
        prompt   = text_analysis_batch_prompt.invoke({
            "question_id":  question_id,
            "question_ar":  question_ar,
            "answers_json": json.dumps(answers_payload, ensure_ascii=False),
        })
        response = model.invoke(prompt)
        parsed   = _parse_json_response(response.content, context=label)

        if not parsed:
            return [], None

        answers_analysis = parsed.get("answers_analysis", [])
        batch_summary = {
            "top_topics":   parsed.get("top_topics",   []),
            "top_entities": parsed.get("top_entities", []),
            "summary":      parsed.get("summary",      ""),
        }

        print(
            f"   ✅ {label} → Stage 1 done "
            f"({len(answers_analysis)} answers tagged)"
        )
        return answers_analysis, batch_summary

    except Exception as e:
        print(f"   ❌ {label} → Stage 1 failed: {e}")
        print(traceback.format_exc())
        return [], None


# ────────────────────────────────────────────────────────────────────────────


def _run_synthesis_stage2(
    question_id: int,
    question_ar: str,
    batch_summaries: list[dict],
    total_answers: int,
) -> dict | None:
    """
    Call LLM Stage 2 (synthesis) with all batch summaries for one question.

    Returns:
        final_report dict or None on failure
    """
    label = f"Q{question_id} | Stage 2 synthesis"

    try:
        prompt   = text_analysis_synthesis_prompt.invoke({
            "question_id":          question_id,
            "question_ar":          question_ar,
            "total_answers":        total_answers,
            "batch_summaries_json": json.dumps(batch_summaries, ensure_ascii=False),
        })
        response = model.invoke(prompt)
        parsed   = _parse_json_response(response.content, context=label)

        if parsed:
            print(f"   ✅ {label} → done")
        else:
            print(f"   ⚠️  {label} → parse failed, storing raw batch summaries")

        return parsed

    except Exception as e:
        print(f"   ❌ {label} → failed: {e}")
        print(traceback.format_exc())
        return None


# ────────────────────────────────────────────────────────────────────────────


def analyze_text_questions_batch(
    df: pd.DataFrame,
    batch_size: int = 30,
) -> tuple[pd.DataFrame, dict]:
    """
    Main entry point.

    Parameters
    ----------
    df          : DataFrame with columns including question_id, question_ar,
                  answer_id, answer
    batch_size  : number of answers per LLM Stage-1 call

    Returns
    -------
    enriched_df   : df with new columns: sentiment, entities, topics
    result_dict   : {str(question_id): final_report_dict}
    """

    print("=" * 60)
    print("TEXT PIPELINE — analyze_text_questions_batch")
    print(f"   batch_size : {batch_size}")
    print(f"   total rows : {len(df)}")
    print("=" * 60)

    # ── Output containers ──────────────────────────────────────────────────
    all_answer_tags: list[dict] = []   # accumulated per-answer results
    result_dict:     dict       = {}   # question_id → final_report

    question_ids = df["question_id"].unique().tolist()
    print(f"📋 Questions to process: {len(question_ids)}")

    for question_id in question_ids:

        q_df        = df[df["question_id"] == question_id]
        question_ar = (
            q_df["question_ar"].iloc[0]
            if "question_ar" in q_df.columns else str(question_id)
        )
        records     = q_df.to_dict(orient="records")
        n_answers   = len(records)

        total_batches = math.ceil(n_answers / batch_size)

        print(f"\n── Q{question_id} ─────────────────────────────────────")
        print(f"   '{question_ar[:60]}...' " if len(question_ar) > 60
              else f"   '{question_ar}'")
        print(f"   {n_answers} answers  →  {total_batches} batch(es)")

        batch_summaries: list[dict]  = []
        q_answer_tags:  list[dict]   = []

        # ── Stage 1: iterate batches ───────────────────────────────────────
        for i in range(total_batches):
            batch_records = records[i * batch_size : (i + 1) * batch_size]
            batch_idx     = i + 1

            answers_analysis, batch_summary = _run_batch_stage1(
                question_id   = question_id,
                question_ar   = question_ar,
                batch_records = batch_records,
                batch_idx     = batch_idx,
                total_batches = total_batches,
            )

            q_answer_tags.extend(answers_analysis)

            if batch_summary:
                batch_summaries.append(batch_summary)

        # ── Stage 2: synthesise ────────────────────────────────────────────
        if batch_summaries:
            final_report = _run_synthesis_stage2(
                question_id     = question_id,
                question_ar     = question_ar,
                batch_summaries = batch_summaries,
                total_answers   = n_answers,
            )
        else:
            final_report = None
            print(f"   ⚠️  Q{question_id} — no batch summaries, skipping Stage 2")

        # Store final report (fallback to batch_summaries if stage 2 failed)
        result_dict[str(question_id)] = final_report or {
            "question_id":       question_id,
            "question_ar":       question_ar,
            "batch_summaries":   batch_summaries,
            "note":              "Stage 2 synthesis failed — raw batch summaries kept",
        }

        all_answer_tags.extend(q_answer_tags)

    # ── Merge per-answer tags back into df ─────────────────────────────────
    print("\n🔗 Merging per-answer tags into DataFrame...")

    if all_answer_tags:
        tags_df = pd.DataFrame(all_answer_tags)                    # answer_id, sentiment, entities, topics
        tags_df["answer_id"] = tags_df["answer_id"].astype(df["answer_id"].dtype)

        enriched_df = df.merge(
            tags_df[["answer_id", "sentiment", "entities", "topics"]],
            on       = "answer_id",
            how      = "left",
        )
    else:
        # No tags produced — add empty columns
        enriched_df = df.copy()
        enriched_df["sentiment"] = None
        enriched_df["entities"]  = None
        enriched_df["topics"]    = None

    enriched_count = enriched_df["sentiment"].notna().sum()
    print(
        f"\n📊 Enrichment complete: "
        f"{enriched_count}/{len(enriched_df)} rows enriched "
        f"across {len(question_ids)} question(s)"
    )

    return enriched_df, result_dict
