import json
import math
import traceback
import ast
from collections import Counter
import pandas as pd
from agents.prompt.text_analysis_batch_prompt import text_analysis_batch_prompt
from agents.prompt.text_analysis_synthesis_prompt import text_analysis_synthesis_prompt
from analytics_pipeline.data_processing.processing_text_data import (
    analyze_text_batch,
    synthesize_batch_summaries,
    semantic_normalization,
)


def extract_top_topics_by_sentiment(
    enriched_df: pd.DataFrame, sentiment_filter: str, top_n: int = 10
) -> list[tuple[str, int]]:

    if "sentiment" not in enriched_df.columns or "topics" not in enriched_df.columns:
        return []

    filtered_df = enriched_df[
        enriched_df["sentiment"].str.lower() == sentiment_filter.lower()
    ].copy()

    all_topics = []
    for topics_val in filtered_df["topics"].dropna():
        if isinstance(topics_val, list):
            all_topics.extend(topics_val)
        elif isinstance(topics_val, str):
            try:
                parsed = ast.literal_eval(topics_val)
                if isinstance(parsed, list):
                    all_topics.extend(parsed)
                else:
                    all_topics.append(topics_val.strip())
            except Exception:
                all_topics.append(topics_val.strip())

    raw_counts = Counter(all_topics).most_common(50)
    
    if raw_counts:
        return semantic_normalization(raw_counts, top_n)
    
    return []


def analyze_text_questions_batch(
    df: pd.DataFrame,
    batch_size: int = 30,
) -> tuple[pd.DataFrame, dict]:

    print("=" * 60)
    print("TEXT PIPELINE — analyze_text_questions_batch")
    print(f"   batch_size : {batch_size}")
    print(f"   total rows : {len(df)}")
    print("=" * 60)

    # ── Output containers ──────────────────────────────────────────────────
    all_answer_tags: list[dict] = []  # accumulated per-answer results
    result_dict: dict = {}  # question_id → final_report

    question_ids = df["question_id"].unique().tolist()
    print(f"📋 Questions to process: {len(question_ids)}")

    for question_id in question_ids:

        q_df = df[df["question_id"] == question_id]
        question_ar = (
            q_df["question_ar"].iloc[0]
            if "question_ar" in q_df.columns
            else str(question_id)
        )
        records = q_df.to_dict(orient="records")
        n_answers = len(records)

        total_batches = math.ceil(n_answers / batch_size)

        print(f"\n── Q{question_id} ─────────────────────────────────────")
        print(
            f"   '{question_ar[:60]}...' "
            if len(question_ar) > 60
            else f"   '{question_ar}'"
        )
        print(f"   {n_answers} answers  →  {total_batches} batch(es)")

        batch_summaries: list[dict] = []
        q_answer_tags: list[dict] = []

        # ── Stage 1: iterate batches ───────────────────────────────────────
        for i in range(total_batches):
            batch_records = records[i * batch_size : (i + 1) * batch_size]
            batch_idx = i + 1

            answers_analysis, batch_summary = analyze_text_batch(
                question_id=question_id,
                question_ar=question_ar,
                batch_records=batch_records,
                batch_idx=batch_idx,
                total_batches=total_batches,
            )

            q_answer_tags.extend(answers_analysis)

            if batch_summary:
                batch_summaries.append(batch_summary)

        # ── Stage 2: synthesise ────────────────────────────────────────────
        if batch_summaries:
            final_report = synthesize_batch_summaries(
                question_id=question_id,
                question_ar=question_ar,
                batch_summaries=batch_summaries,
                total_answers=n_answers,
            )
        else:
            final_report = None
            print(f"   ⚠️  Q{question_id} — no batch summaries, skipping Stage 2")

        # Store final report (fallback to batch_summaries if stage 2 failed)
        result_dict[str(question_id)] = final_report or {
            "question_id": question_id,
            "question_ar": question_ar,
            "batch_summaries": batch_summaries,
            "note": "Stage 2 synthesis failed — raw batch summaries kept",
        }

        all_answer_tags.extend(q_answer_tags)

    # ── Merge per-answer tags back into df ─────────────────────────────────
    print("\n🔗 Merging per-answer tags into DataFrame...")

    if all_answer_tags:
        tags_df = pd.DataFrame(
            all_answer_tags
        )  # answer_id, sentiment, entities, topics
        tags_df["answer_id"] = tags_df["answer_id"].astype(df["answer_id"].dtype)

        enriched_df = df.merge(
            tags_df[["answer_id", "sentiment", "entities", "topics"]],
            on="answer_id",
            how="left",
        )
    else:
        # No tags produced — add empty columns
        enriched_df = df.copy()
        enriched_df["sentiment"] = None
        enriched_df["entities"] = None
        enriched_df["topics"] = None

    enriched_count = enriched_df["sentiment"].notna().sum()
    print(
        f"\n📊 Enrichment complete: "
        f"{enriched_count}/{len(enriched_df)} rows enriched "
        f"across {len(question_ids)} question(s)"
    )

    return enriched_df, result_dict
