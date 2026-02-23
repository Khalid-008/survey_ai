import os
import sys
import warnings
from typing import Any

import pandas as pd

# ─── path setup ───────────────────────────────────────────────────────────────
current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(current_file_path))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from llms.models import model
from data.operations import get_selection_questions_data, execute_raw_query
from analytics_pipeline.util.selection_utils import (
    extract_sql,
    format_questions_block,
    save_results_to_file,
)
from agents.prompt.selection_q1_statistical_summary_prompt  import q1_statistical_summary_prompt
from agents.prompt.selection_q2_satisfaction_tiers_prompt   import q2_satisfaction_tiers_prompt
from agents.prompt.selection_q3_cross_tabulation_prompt     import q3_cross_tabulation_prompt
from agents.prompt.selection_q4_correlation_analysis_prompt import q4_correlation_analysis_prompt

warnings.filterwarnings("ignore")


# ── Query definitions (label, prompt) ────────────────────────────────────────
_QUERY_DEFINITIONS = [
    ("Query 1 · Statistical Summary",            q1_statistical_summary_prompt),
    ("Query 2 · Satisfaction Tiers Distribution", q2_satisfaction_tiers_prompt),
    ("Query 3 · Cross-tabulation Segment Labels", q3_cross_tabulation_prompt),
    ("Query 4 · Correlation & Pattern Analysis",  q4_correlation_analysis_prompt),
]


# ============================================================================
# STEP 1 — FETCH & GROUP DATA
# ============================================================================

def fetch_selection_questions(survey_number: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"📋 Fetching selection questions for survey: {survey_number}")
    print(f"{'='*60}")
    df = get_selection_questions_data(survey_number)
    if df.empty:
        print("⚠️  No selection questions found for this survey.")
    else:
        print(f"✅ Found {df['question_id'].nunique()} question(s), {len(df)} answer rows.")
    return df


def group_answers_by_question(df: pd.DataFrame) -> dict[int, dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for question_id, group in df.groupby("question_id"):
        grouped[int(question_id)] = {
            "text":    group["question_ar"].iloc[0],
            "type":    group["question_type"].iloc[0],
            "answers": group["answer"].dropna().tolist(),
        }
    return grouped


def get_distinct_answers_per_question(grouped: dict[int, dict[str, Any]]) -> dict[int, list[str]]:
    distinct: dict[int, list[str]] = {}
    for qid, info in grouped.items():
        unique_answers = sorted(set(str(a) for a in info["answers"] if a))
        distinct[qid] = unique_answers
        print(f"   Q{qid} ({info['text'][:40]}...): {len(unique_answers)} distinct answers")
    return distinct


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_selection_pipeline(survey_number: str) -> dict[str, Any]:
    print(f"\n{'#'*60}")
    print(f"# SELECTION PIPELINE — Survey: {survey_number}")
    print(f"{'#'*60}\n")

    result: dict[str, Any] = {
        "questions_count":       0,
        "distinct_per_question": {},
        "query_results":         [],
        "errors":                [],
    }

    # 1. Fetch
    print("📥 Step 1 — Fetching selection questions...")
    df = fetch_selection_questions(survey_number)
    if df.empty:
        result["errors"].append("No selection questions found.")
        return result

    # 2. Group & distinct
    print("\n📊 Step 2 — Grouping answers...")
    grouped  = group_answers_by_question(df)
    distinct = get_distinct_answers_per_question(grouped)

    result["questions_count"]       = len(grouped)
    result["distinct_per_question"] = distinct

    questions_block = format_questions_block(grouped, distinct)
    print("questions_block:\n", questions_block)

    # 3. Generate → Execute → Store (one query at a time)
    total = len(_QUERY_DEFINITIONS)
    for idx, (label, prompt) in enumerate(_QUERY_DEFINITIONS, start=1):
        print(f"\n{'─'*60}")
        print(f"🤖 Step {idx + 2}/{total + 2} — Generating: {label}")
        print(f"{'─'*60}")

        entry: dict[str, Any] = {"label": label, "sql": "", "result": pd.DataFrame()}

        # Generate
        sql = ""
        try:
            chain    = prompt | model
            response = chain.invoke({"questions_block": questions_block})
            sql      = extract_sql(response.content)
            entry["sql"] = sql
            print(f"   ✅ SQL generated ({len(sql)} chars).")
            print(f"   SQL:\n{sql}\n")
        except Exception as e:
            err = f"[{label}] Generation failed: {e}"
            print(f"   ❌ {err}")
            result["errors"].append(err)
            result["query_results"].append(entry)
            continue

        # Execute
        print(f"   ⚡ Executing...")
        try:
            df_result    = execute_raw_query(sql)
            entry["result"] = df_result
            print(f"   ✅ {len(df_result)} rows returned.")
        except Exception as e:
            err = f"[{label}] Execution failed: {e}"
            print(f"   ❌ {err}")
            result["errors"].append(err)

        # Store
        result["query_results"].append(entry)
        print(f"   💾 Stored: {label}")

    # Summary
    total_rows = sum(len(qr["result"]) for qr in result["query_results"])
    print(f"\n{'='*60}")
    print(f"✅ DONE — Survey: {survey_number}")
    print(f"   Questions : {result['questions_count']}")
    print(f"   Queries   : {len(result['query_results'])}")
    print(f"   Rows      : {total_rows}")
    if result["errors"]:
        print(f"   ⚠️  Errors ({len(result['errors'])}):")
        for err in result["errors"]:
            print(f"      • {err}")
    print(f"{'='*60}\n")

    # Save
    result["report_path"] = save_results_to_file(
        survey_number=survey_number,
        grouped=grouped,
        distinct=distinct,
        query_results=result["query_results"],
        errors=result["errors"],
    )

    return result
