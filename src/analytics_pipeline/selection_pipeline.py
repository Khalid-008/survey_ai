import os
import sys
import warnings
from typing import Any
import pandas as pd
from llms.models import qwen3_model
from data.operations import execute_raw_query
from analytics_pipeline.util.selection_utils import (
    extract_sql,
    clean_sql_queries,
    validate_query,
    format_questions_block,
    save_results_to_file,
)
from agents.prompt.selection_analytics_prompt import selection_analytics_prompt

# ─── path setup ───────────────────────────────────────────────────────────────
current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(current_file_path))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

warnings.filterwarnings("ignore")


# MAIN PIPELINE


def generate_sql_analisys_queries(
    survey_id: str, prepared: dict[str, Any], analytical_questions: str = ""
) -> dict[str, Any]:

    print(f"\n{'#'*60}")

    print(f"# SELECTION PIPELINE — Survey: {survey_id}")

    print(f"{'#'*60}\n")

    result: dict[str, Any] = {
        "query_results": [],
        "errors": [],
    }

    # قراءة questions_block من النود السابق

    questions_block = prepared.get("questions_block", "")

    if not questions_block:

        result["errors"].append("No selection questions found.")
        return result

    # 3. Generate

    print(f"\n{'─'*60}")

    print(f"🤖 Step 3 — Generating 5 analytical queries...")

    print(f"{'─'*60}")

    try:

        prompt_messages = selection_analytics_prompt.invoke(
            {
                "questions_block": questions_block,
                "analytical_questions": analytical_questions,
            }
        )

        response = qwen3_model.invoke(prompt_messages)

        raw_output = response.content

        if not raw_output:

            print(f"   ⚠️ Empty content! Full response debug:")

            print(f"      type: {type(response)}")

            print(f"      additional_kwargs: {response.additional_kwargs}")

            print(f"      response_metadata: {response.response_metadata}")

        print(f"   ✅ LLM responded ({len(raw_output)} chars).")

    except Exception as e:

        err = f"Generation failed: {e}"

        print(f"   ❌ {err}")

        result["errors"].append(err)
        return result

    # 4. Clean & split into individual queries

    queries = clean_sql_queries(raw_output)

    print(f"   📋 {len(queries)} queries extracted after cleaning.")

    print(f"\n{'═'*60}")

    print("📄 ALL CLEANED QUERIES (before execution):")
    print(f"{'═'*60}")

    for i, sql in enumerate(queries, start=1):

        print(f"\n── Query {i} ──\n{sql}")

    print(f"\n{'═'*60}\n")

    # 5. Validate → Execute → Store (one query at a time)

    for i, sql in enumerate(queries, start=1):

        label = f"Query {i}"

        entry: dict[str, Any] = {"label": label, "sql": sql, "result": pd.DataFrame()}

        print(f"\n{'─'*60}")

        print(f"⚡ {label}")

        print(f"{'─'*60}")

        print(sql)

        # Validate

        is_valid, issues = validate_query(sql)

        if not is_valid:

            print(f"   ⚠️  Validation warnings (auto-fixed in clean step):")

            for issue in issues:

                print(f"      {issue}")

        # Execute

        try:

            df_result = execute_raw_query(sql)

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

    print(f"✅ DONE — Survey: {survey_id}")

    print(f"   Queries   : {len(result['query_results'])}")

    print(f"   Rows      : {total_rows}")

    if result["errors"]:

        print(f"   ⚠️  Errors ({len(result['errors'])}):")

        for err in result["errors"]:

            print(f"      • {err}")

    print(f"{'='*60}\n")

    # Save

    result["report_path"] = save_results_to_file(
        survey_number=survey_id,
        query_results=result["query_results"],
        errors=result["errors"],
    )

    return result
