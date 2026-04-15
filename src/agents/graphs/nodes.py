import json
import os
import re
import sys
import traceback
import warnings
from typing import Any, Dict, Literal

import pandas as pd
from langchain_core.messages import AIMessage
from langgraph.types import Command

# إعداد المسار للاستيراد
current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from agents.graphs.setup import State
from agents.prompt.analysis_questions_recommender_prompt import (
    analysis_questions_recommender_prompt,
)
from agents.prompt.selection_result_summary_prompt import (
    selection_result_summary_prompt,
)
from agents.prompt.synthesis_agent_prompt import synthesis_agent_prompt

# معالجة بيانات الخيارات
from analytics_pipeline.data_processing.processing_selection_data import (
    get_distinct_answers_per_question,
    get_sample_answers_per_question,
    group_answers_by_question,
)
from analytics_pipeline.selection_pipeline import generate_sql_analisys_queries
from analytics_pipeline.text_pipeline import (
    analyze_text_questions_batch,
    extract_top_topics_by_sentiment,
)
from analytics_pipeline.util.chart_utils import (
    extract_json_from_response,
    extract_python_code,
    format_analysis_summary,
    format_selection_results_for_prompt,
)
from analytics_pipeline.util.selection_utils import format_questions_block
from analytics_pipeline.util.synthesis_utils import (
    format_analytics_messages,
    save_report,
)
from data.operations import get_survey_df
from llms.models import model, qwen3_model

warnings.filterwarnings("ignore")


def _get_analysis_results(state: State):
    """Extract text and selection analysis results from state.

    Returns:
        tuple: (text_questions_result, selection_questions_result)
    """
    text_questions_result = state.get("text_questions_result", {})
    selection_questions_result = state.get("selection_questions_result", {})
    return text_questions_result, selection_questions_result


def _get_survey_title(state: State, text_questions_result: dict) -> str:
    """Extract survey title from text analysis results or fall back to Survey Number."""
    survey_title = f"Survey {state['survey_number']}"
    if text_questions_result:
        first_result = next(iter(text_questions_result.values()))
        survey_title = first_result.get("survey_title", survey_title)
    return survey_title


# ============================================================================


# NODE 1 — RETRIEVE SURVEY QUESTIONS


# استرجاع أسئلة الاستبيان


# ============================================================================


def retrieve_survey_question(state: State) -> Dict[str, Any]:
    try:

        survey_number = state["survey_number"]

        date_from = state.get("date_from")

        date_to = state.get("date_to")

        print("=" * 60)

        print("RETRIEVE SURVEY QUESTIONS NODE")

        print("=" * 60)

        print(f"📋 Survey Number : {survey_number}")
        if date_from or date_to:

            print(f"📅 Date filter: {date_from} → {date_to}")

        survey_df = get_survey_df(survey_number, date_from=date_from, date_to=date_to)

        rows = len(survey_df)

        print(f"📊 Loaded {rows} rows from database")

        if survey_df.empty:

            print("⚠️  No data found — stopping workflow")
            Command(
                update={
                    "survey_data": [],
                    "text_questions_data": [],
                    "selection_questions_data": [],
                    "stop_reason": f"No data found for survey {survey_number}.",
                    "messages": [
                        AIMessage(
                            content=f"No data found for survey {survey_number}. Analysis skipped."
                        )
                    ],
                },
                goto="__end__",
            )

        survey_data = survey_df.to_dict(orient="records")

        # ── تصنيف الأسئلة إلى مجموعتين ──────────────────────────────────────

        all_types = survey_df["question_type"].str.upper().str.strip().unique().tolist()

        print(f"🔍 All question types detected: {all_types}")

        TEXT_TYPE = "TEXT_INPUT"

        # ── المجموعة 1: أسئلة نصية (TEXT_INPUT) ──

        text_mask = survey_df["question_type"].str.upper().str.strip() == TEXT_TYPE

        text_df = survey_df[text_mask]

        text_q_ids = (
            text_df["question_id"].unique().tolist()
            if "question_id" in text_df.columns
            else []
        )

        # ── المجموعة 2: أسئلة الاختيارات (كل ما ليس TEXT_INPUT) ──

        selection_df = survey_df[~text_mask]

        sel_q_ids = (
            selection_df["question_id"].unique().tolist()
            if "question_id" in selection_df.columns
            else []
        )

        print("-" * 60)

        print(f"✅ Classification complete")

        print("=" * 60)

        return {
            "survey_data": survey_data,
            "text_questions_data": text_df.to_dict(orient="records"),
            "selection_questions_data": selection_df.to_dict(orient="records"),
            "messages": [
                AIMessage(
                    content=(
                        f"Survey data retrieved: {rows} rows. "
                        f"Classified → Text: {len(text_df)} rows ({len(text_q_ids)} questions), "
                        f"Selection: {len(selection_df)} rows ({len(sel_q_ids)} questions)."
                    )
                )
            ],
        }

    except Exception as e:

        error_msg = f"Failed to retrieve survey data: {str(e)}"

        print(f"❌ {error_msg}")

        print(traceback.format_exc())

        raise RuntimeError(error_msg)


# ============================================================================


# NODE 2 — PREPARE SELECTION DATA


# تجهيز بيانات الخيارات (تحويل، تحقق، تجميع)


# ============================================================================


def prepare_selection_data(state: State) -> Dict[str, Any]:

    survey_number = state["survey_number"]

    print("=" * 60)

    print("PREPARE SELECTION DATA NODE")

    print("=" * 60)

    raw_data = state.get("selection_questions_data", [])

    # 1. تحويل البيانات وتحقق

    print("📥 Step 1 — Preparing selection questions data...")

    df = pd.DataFrame(raw_data)

    if df.empty:

        print("⚠️  No selection questions found — skipping.")

        return {
            "selection_prepared": {},
            "messages": [
                AIMessage(
                    content="No selection questions found. Skipping selection analysis."
                )
            ],
        }

    print(f"   ✅ {len(df)} rows | {df['question_id'].nunique()} questions")

    # 2. Group & distinct

    print("\n📊 Step 2 — Grouping answers...")

    grouped = group_answers_by_question(df)

    distinct = get_distinct_answers_per_question(grouped)

    samples = get_sample_answers_per_question(df, n=3)

    questions_block = format_questions_block(
        grouped, distinct, sample_answers=samples, survey_number=str(survey_number)
    )

    return {
        "selection_prepared": questions_block,
        "messages": [
            AIMessage(
                content=(
                    f"Selection data prepared: {len(grouped)} questions, "
                    f"{sum(len(v) for v in distinct.values())} total distinct answers."
                ),
                name="prepare_selection_data",
            )
        ],
    }


# ============================================================================


# NODE 3 — ANALYZE SELECTION QUESTIONS


# تحليل أسئلة الخيارات


# ============================================================================


def analyze_selection_questions(state: State) -> Dict[str, Any]:

    try:

        survey_number = state["survey_number"]

        print("=" * 60)

        print("SELECTION ANALYSIS NODE")

        print("=" * 60)

        print(f"Survey Number: {survey_number}")

        # Step A — توليد الأسئلة التحليلية
        questions_block = state.get("selection_prepared", {})

        print("\n🧠 Generating analytical questions from questions_block...")

        num_questions = state.get("num_questions", 4)

        try:
            q_prompt = analysis_questions_recommender_prompt.invoke(
                {"questions_block": questions_block, "num_questions": num_questions}
            )
            q_response = model.invoke(q_prompt)
            analytical_questions = q_response.content
            print(
                f"   ✅ Analytical questions generated ({len(analytical_questions)} chars)."
            )
        except Exception as e:
            analytical_questions = ""
            print(f"   ⚠️ Could not generate analytical questions: {e}")

        # Step B — توليد SQL queries بناءً على الأسئلة التحليلية
        queries_results = generate_sql_analisys_queries(
            survey_number=survey_number,
            prepared=questions_block,
            analytical_questions=analytical_questions,
        )

        # تحويل DataFrames إلى قوائم قابلة للتسلسل

        # الهيكل الجديد: query_results = [{label, sql, result: DataFrame}, ...]

        querys_results_serialized = []

        total_rows = 0

        for entry in queries_results.get("query_results", []):

            df_result = entry.get("result", pd.DataFrame())
            rows = df_result.to_dict(orient="records") if not df_result.empty else []
            total_rows += len(rows)

            # Generate Arabic summary for this query result
            result_summary = ""
            question = entry.get("sql", "")
            if rows and question:
                try:
                    summary_prompt = selection_result_summary_prompt.invoke(
                        {
                            "question": question,
                            "results": json.dumps(
                                rows, ensure_ascii=False, default=str
                            ),
                        }
                    )
                    summary_response = model.invoke(summary_prompt)
                    result_summary = summary_response.content
                    # Clean special Unicode characters
                    result_summary = re.sub(
                        r"[\u200b\u200c\u200d\u200e\u200f\u202a-\u202f\u2060\ufeff]",
                        " ",
                        result_summary,
                    )
                    result_summary = re.sub(r" {2,}", " ", result_summary).strip()
                    print(f"   📝 Summary generated for: {question[:50]}...")
                except Exception as summary_err:
                    print(f"   ⚠️ Could not generate summary: {summary_err}")

            querys_results_serialized.append(
                {
                    "label": entry.get("label", ""),
                    "sql": entry.get("sql", ""),
                    "result": rows,
                    "result_summary": result_summary,
                }
            )

        # ── 2) نتائج أسئلة الخيارات ──────────────────────────────────────────────
        sel_parts = []
        for qr in querys_results_serialized:
            label = qr.get("label", "")
            rows = qr.get("result", [])
            result_summary = qr.get("result_summary", "")
            if rows:
                sel_parts.append(
                    f"{label} Results\n"
                    + json.dumps(rows, ensure_ascii=False, indent=2, default=str)
                    + "\n\n"
                    + f"Result Summary: {result_summary}"
                )

        serializable_result = {
            "results": sel_parts,
            "query_results": querys_results_serialized,
            "errors": queries_results["errors"],
        }

        print(
            f"✅ Selection analysis complete: {len(querys_results_serialized)} queries, {total_rows} total rows"
        )

        return {
            "selection_questions_result": serializable_result,
            "messages": [
                AIMessage(
                    content=(
                        f"Selection analysis complete: {len(querys_results_serialized)} queries, "
                        f"Total rows: {total_rows}."
                    ),
                    name="analyze_selection_questions",
                )
            ],
        }

    except Exception as e:

        error_msg = f"Selection analysis failed: {str(e)}"

        print(f"⚠️ {error_msg}")

        print(traceback.format_exc())

        # خطأ غير حرج — نكمل بدون نتائج خيارات

        return {
            "selection_questions_result": {"errors": [error_msg]},
            "messages": [
                AIMessage(
                    content=f"Warning: {error_msg}. Proceeding to text analysis.",
                    name="analyze_selection_questions",
                )
            ],
        }


# ============================================================================


# NODE 3 — ANALYZE TEXT QUESTIONS (ENRICH)


# تحليل وإثراء أسئلة النصوص


# ============================================================================


def analyze_text_questions(state: State) -> Dict[str, Any]:

    try:

        print("=" * 60)

        print("TEXT ANALYSIS NODE")

        print("=" * 60)

        if not state.get("text_questions_data"):

            return {
                "text_questions_data": [],
                "text_questions_result": {},
                "messages": [
                    AIMessage(content="No survey data available for text analysis.")
                ],
            }

        text_questions_data = pd.DataFrame(state["text_questions_data"])

        rows_before = len(text_questions_data)

        print(f"📊 Loaded {rows_before} rows for survey {state['survey_number']}")

        if len(text_questions_data) == 0:

            return {
                "survey_data": [],
                "text_questions_result": {},
                "messages": [
                    AIMessage(content="Survey data is empty — skipping text analysis.")
                ],
            }

        survey_title = (
            text_questions_data["survey_title"].iloc[0]
            if "survey_title" in text_questions_data.columns
            else f"Survey {state['survey_number']}"
        )

        print(f"🔬 Starting text enrichment for: {survey_title}")

        # ── Two-stage batch analysis ──────────────────────────────────────
        enriched_df, text_questions_result = analyze_text_questions_batch(
            df=text_questions_data,
            batch_size=15,
        )

        enriched_data = enriched_df.to_dict(orient="records")

        # top_positive_topics
        top_positive_topics = extract_top_topics_by_sentiment(
            enriched_df, "positive", 10
        )

        if top_positive_topics:
            print("أكثر المواضيع الإيجابية:")
            for topic, count in top_positive_topics:
                print(f"   - {topic}: {count} مرة")

            # تضمينها في النتيجة
            for qt_idx in text_questions_result:
                text_questions_result[qt_idx][
                    "top_positive_topics"
                ] = top_positive_topics

        n_questions = len(text_questions_result)
        n_enriched = (
            enriched_df["sentiment"].notna().sum()
            if "sentiment" in enriched_df.columns
            else 0
        )

        print(
            f"📈 Enrichment complete: {n_enriched}/{rows_before} rows | {n_questions} question(s)"
        )

        return {
            "survey_data": enriched_data,
            "text_questions_result": text_questions_result,
            "messages": [
                AIMessage(
                    content=(
                        f"Text analysis complete: {n_questions} TEXT_INPUT question(s) analysed, "
                        f"{n_enriched}/{rows_before} answers enriched."
                    )
                )
            ],
        }

    except Exception as e:

        error_msg = f"Text analysis failed: {str(e)}"

        print(f"⚠️ {error_msg}")

        print(traceback.format_exc())

        # خطأ غير حرج — نكمل بالبيانات الأساسية

        return {
            "survey_data": state.get("survey_data", []),
            "text_questions_result": state.get("text_questions_result", {}),
            "messages": [
                AIMessage(content=f"Warning: {error_msg}. Proceeding with basic data.")
            ],
        }


# ============================================================================


# NODE 4 — SYNTHESIS AGENT


# وكيل التلخيص التنفيذي


# ============================================================================


def synthesis_agent(state: State) -> Dict[str, Any]:

    try:

        print("=" * 60)

        print("SYNTHESIS AGENT")

        print("=" * 60)

        print(f"Survey Number: {state['survey_number']}")

        text_questions_result, selection_questions_result = _get_analysis_results(state)

        if not text_questions_result and not selection_questions_result:

            return Command(
                update={
                    "messages": [
                        AIMessage(
                            content="No analysis results to synthesize.",
                            name="synthesis_agent",
                        )
                    ]
                }
            )

        survey_title = _get_survey_title(state, text_questions_result)

        print(f"Survey: {survey_title}")

        print(f"Text questions analyzed  : {len(text_questions_result)}")

        print(
            f"Selection questions count: {len(selection_questions_result.get('results', []))}"
        )

        # تنسيق التحليلات النصية (قائمة من الرسائل)

        text_questions_result = format_analytics_messages(text_questions_result)

        prompt_messages = synthesis_agent_prompt.invoke(
            {
                "survey_subject": survey_title,
                "selection_questions_result": selection_questions_result,
                "text_questions_result": text_questions_result,
            }
        )

        response = model.invoke(prompt_messages)

        synthesis_content = response.content

        save_report(synthesis_content, state["survey_number"])

        print("✅ Synthesis complete")

        return Command(
            update={
                "messages": [
                    AIMessage(content=synthesis_content, name="synthesis_agent")
                ]
            }
        )

    except Exception as e:

        error_msg = f"Synthesis failed: {str(e)}"

        print(f"❌ {error_msg}")

        print(traceback.format_exc())

        return Command(
            update={"messages": [AIMessage(content=error_msg, name="synthesis_agent")]}
        )


# ============================================================================


# NODE 5 — GENERATE CHARTS AGENT


# وكيل توليد الرسوم البيانية


# ============================================================================


def _build_charts_payload(
    text_questions_result: dict, selection_questions_result: dict
) -> dict | None:
    """Build a structured JSON payload for the visualization engine."""

    text_analysis = {}
    for q_id, result in text_questions_result.items():
        q_text = result.get("question_ar", result.get("survey_question", f"Q{q_id}"))
        entry = {"question": q_text}

        if result.get("dominant_sentiment"):
            entry["dominant_sentiment"] = result["dominant_sentiment"]

        if result.get("top_positive_topics"):
            entry["top_topics"] = [
                {"topic": t, "count": c} for t, c in result["top_positive_topics"]
            ]

        if result.get("top_entities"):
            entry["top_entities"] = result["top_entities"]
        text_analysis[q_id] = entry

    selection_queries = [
        {
            "label": item.get("label", ""),
            "result": item.get("result", []),
        }
        for item in selection_questions_result.get("query_results", [])
        if item.get("result")
    ]

    if not text_analysis and not selection_queries:
        return None

    return {"text_analysis": text_analysis, "selection_queries": selection_queries}


def _call_visualization_api(data: dict, prompt: str) -> list:
    """Call the Visual 0.2 engine API and return chart configs."""
    import requests

    url = os.environ.get("VISUALIZATION_API_URL", "http://localhost:3001/api/analyze")

    import datetime, decimal

    def _default(obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        return str(obj)

    payload = {
        "question": prompt,
        "data": json.loads(json.dumps(data, default=_default)),
    }

    sel_count = len(data.get("selection_queries", []))
    txt_count = len(data.get("text_analysis", {}))
    print(
        f"📊 Calling visualization API: {url}  (text_questions={txt_count}, selection_queries={sel_count})"
    )
    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()

    charts = response.json()
    if isinstance(charts, dict):
        if "error" in charts:
            print(f"⚠️ Visualization API error: {charts['error']}")
            return []
        charts = [charts]

    print(f"✅ Visualization API returned {len(charts)} chart(s)")
    return charts


def generate_charts_agent(state: State) -> Command[Literal["__end__"]]:

    try:

        print("=" * 60)

        print("CHART GENERATION AGENT")

        print("=" * 60)

        print(f"Survey Number: {state['survey_number']}")

        text_questions_result, selection_questions_result = _get_analysis_results(state)

        charts_data = _build_charts_payload(
            text_questions_result, selection_questions_result
        )

        if not charts_data:
            print("⚠️ No data available for chart generation")
            return Command(
                update={
                    "chart_configs": [],
                    "messages": [
                        AIMessage(
                            content="No data available for chart generation.",
                            name="chart_generation_agent",
                        )
                    ],
                },
                goto="__end__",
            )

        # Build a prompt that instructs the visualization engine
        survey_title = _get_survey_title(state, text_questions_result)
        prompt = (
            f"Survey: {survey_title}. "
            "Generate 9 unique dashboard charts covering all data aspects. "
            "Mix chart types (pie, doughnut, line, bar). "
            "Use pie/doughnut for long-text labels. No duplicate data or titles."
        )

        #         "Prioritize: entity frequency breakdowns, topic frequency breakdowns, "
        # "and selection question results. "

        chart_configs = _call_visualization_api(charts_data, prompt)

        print(f"   ✅ {len(chart_configs)} chart(s) generated")

        return Command(
            update={
                "chart_configs": chart_configs,
                "messages": [
                    AIMessage(
                        content=f"Chart generation complete: {len(chart_configs)} chart(s).",
                        name="chart_generation_agent",
                    )
                ],
            },
            goto="__end__",
        )

    except Exception as e:

        error_msg = f"Chart generation failed: {str(e)}"

        print(f"❌ {error_msg}")

        print(traceback.format_exc())

        return Command(
            update={
                "chart_configs": [],
                "messages": [
                    AIMessage(content=error_msg, name="chart_generation_agent")
                ],
            },
            goto="__end__",
        )
