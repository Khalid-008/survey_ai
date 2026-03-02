import os
import sys
import json
import traceback
import warnings
from typing import Literal, Dict, Any

import pandas as pd
from langchain_core.messages import AIMessage
from langgraph.types import Command

# إعداد المسار للاستيراد
current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from llms.models import model
from agents.graphs.setup import State
from data.operations import get_survey_df
from agents.prompt.synthesis_agent_prompt import synthesis_agent_prompt
from agents.prompt.chart_generation_prompt import chart_generation_prompt

# معالجة بيانات النصوص
from analytics_pipeline.data_processing.data_cleaner import clean_survey_data
from analytics_pipeline.data_processing.data_enricher import process_survey_questions

# معالجة بيانات الخيارات
from analytics_pipeline.selection_pipeline import run_selection_pipeline

from helper.utils import make_serializable
from analytics_pipeline.util.synthesis_utils import format_analytics_messages, save_report
from analytics_pipeline.util.chart_utils import extract_json_from_response, format_analysis_summary

warnings.filterwarnings('ignore')


# ============================================================================
# NODE 1 — RETRIEVE SURVEY QUESTIONS
# استرجاع أسئلة الاستبيان
# ============================================================================

def retrieve_survey_question(state: State) -> Dict[str, Any]:
    """
    STAGE 1: استرجاع بيانات الاستبيان الخام من قاعدة البيانات.
    لا يحدث أي إثراء أو تنظيف هنا — فقط تحميل البيانات.
    """
    try:
        survey_id = state["survey_id"]
        date_from = state.get("date_from")
        date_to   = state.get("date_to")

        survey_df = get_survey_df(survey_id, date_from=date_from, date_to=date_to)

        rows = len(survey_df)
        print(f"📊 Loaded {rows} rows for survey {survey_id}"
              + (f" [filter: {date_from} → {date_to}]" if date_from or date_to else ""))

        if survey_df.empty:
            return {
                "survey_data": [],
                "messages": [
                    AIMessage(content=f"No data found for survey {survey_id}. Analysis skipped.")
                ]
            }

        survey_data = survey_df.to_dict(orient="records")

        return {
            "survey_data": survey_data,
            "text_questions_result": {},
            "selection_questions_result": {},
            "messages": [
                AIMessage(
                    content=f"Survey data retrieved: {rows} rows ready for analysis."
                )
            ]
        }

    except Exception as e:
        error_msg = f"Failed to retrieve survey data: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        raise RuntimeError(error_msg)


# ============================================================================
# NODE 2 — ANALYZE SELECTION QUESTIONS
# تحليل أسئلة الخيارات
# ============================================================================

def analyze_selection_questions(state: State) -> Dict[str, Any]:
    """
    STAGE 2: تحليل أسئلة الخيارات (غير TEXT_INPUT) باستخدام
    selection_pipeline الذي يولّد كويريات MySQL عبر وكيلين LLM:
      - وكيل 1: يولّد كويري توزيع الإجابات لكل سؤال
      - وكيل 2: يولّد كويري العلاقة بين الأسئلة
    ثم ينفّذ الكويريين ويحفظ النتائج في الـ state.
    """
    try:
        survey_id = state["survey_id"]
        date_from = state.get("date_from")
        date_to   = state.get("date_to")

        print("=" * 60)
        print("SELECTION ANALYSIS NODE")
        print("=" * 60)
        print(f"Survey ID: {survey_id}")
        if date_from or date_to:
            print(f"📅 Date filter: {date_from} → {date_to}")

        # تشغيل مسار تحليل الخيارات
        pipeline_result = run_selection_pipeline(survey_id, date_from=date_from, date_to=date_to)

        # تحويل DataFrames إلى قوائم قابلة للتسلسل
        # الهيكل الجديد: query_results = [{label, sql, result: DataFrame}, ...]
        query_results_serialized = []
        total_rows = 0
        for entry in pipeline_result.get("query_results", []):
            df_result = entry.get("result", pd.DataFrame())
            rows = df_result.to_dict(orient="records") if not df_result.empty else []
            total_rows += len(rows)
            query_results_serialized.append({
                "label":  entry.get("label", ""),
                "sql":    entry.get("sql", ""),
                "result": rows,
            })

        serializable_result = {
            "questions_count":       pipeline_result["questions_count"],
            "questions_metadata":    pipeline_result.get("questions_metadata", []),
            "distinct_per_question": pipeline_result["distinct_per_question"],
            "query_results":         query_results_serialized,
            "errors":                pipeline_result["errors"],
        }

        q_count = pipeline_result["questions_count"]

        print(f"✅ Selection analysis complete: {q_count} questions, "
              f"{len(query_results_serialized)} queries, {total_rows} total rows")

        return {
            "selection_questions_result": serializable_result,
            "messages": [
                AIMessage(
                    content=(
                        f"Selection analysis complete: {q_count} questions analyzed. "
                        f"Queries: {len(query_results_serialized)}, Total rows: {total_rows}."
                    ),
                    name="analyze_selection_questions"
                )
            ]
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
                    name="analyze_selection_questions"
                )
            ]
        }


# ============================================================================
# NODE 3 — ANALYZE TEXT QUESTIONS (ENRICH)
# تحليل وإثراء أسئلة النصوص
# ============================================================================

def analyze_text_questions(state: State) -> Dict[str, Any]:
    """
    STAGE 3: تنظيف البيانات وإثراء أسئلة TEXT_INPUT بـ NLP.

    أ) التنظيف — يُزيل التكرارات
    ب) يتحقق من وجود أسئلة نصية
    ج) إذا وُجدت → يُطبّق تحليل المشاعر، NER، واستخراج المواضيع
    د) إذا لم توجد → يمرر البيانات كما هي
    """
    try:
        print("=" * 60)
        print("TEXT ANALYSIS NODE")
        print("=" * 60)

        if not state.get("survey_data"):
            return {
                "survey_data": [],
                "text_questions_result": {},
                "messages": [AIMessage(content="No survey data available for text analysis.")]
            }

        survey_df = pd.DataFrame(state["survey_data"])

        # أ) تنظيف البيانات
        rows_before = len(survey_df)
        print(f"📊 Loaded {rows_before} rows for survey {state['survey_id']}")

        survey_df = clean_survey_data(survey_df)

        rows_after = len(survey_df)
        print(f"✨ Cleaned data: {rows_after} rows ({rows_before - rows_after} duplicates removed)")

        if survey_df.empty:
            return {
                "survey_data": [],
                "text_questions_result": {},
                "messages": [AIMessage(content="Survey data is empty after cleaning.")]
            }

        # ب) التحقق من وجود أسئلة نصية
        question_types = survey_df["QuestionType"].str.upper().str.strip().unique()
        has_text_questions = "TEXT_INPUT" in question_types
        print(f"🔬 Unique QuestionTypes: {list(question_types)}")
        print(f"🔬 Has TEXT_INPUT questions: {has_text_questions}")

        survey_title = (
            survey_df["SurveyTitle"].iloc[0]
            if "SurveyTitle" in survey_df.columns
            else f"Survey {state['survey_id']}"
        )

        if not has_text_questions:
            # لا توجد أسئلة نصية — نمرر البيانات بدون إثراء
            print("ℹ️  No TEXT_INPUT questions found. Skipping NLP enrichment.")
            survey_data = survey_df.to_dict(orient="records")
            return {
                "survey_data": survey_data,
                "analysis_results": {},
                "messages": [
                    AIMessage(
                        content="No TEXT_INPUT questions found. Skipping NLP enrichment."
                    )
                ]
            }

        # ج) تطبيق إثراء NLP على أسئلة TEXT_INPUT
        print(f"🔬 Starting enrichment for: {survey_title}")
        enriched_df, analysis_results, metrics = process_survey_questions(
            survey_df, survey_title
        )

        print(f"📈 Enrichment complete:")
        print(f"   • TEXT_INPUT questions: {metrics.text_input_questions}")
        print(f"   • Successfully enriched: {metrics.enriched_questions}")
        print(f"   • Failed              : {metrics.failed_questions}")

        enriched_data = enriched_df.to_dict(orient="records")
        serializable_analysis = make_serializable(analysis_results)

        return {
            "survey_data": enriched_data,
            "text_questions_result": serializable_analysis,
            "messages": [
                AIMessage(
                    content=(
                        f"Text analysis complete: {metrics.enriched_questions}/"
                        f"{metrics.text_input_questions} TEXT_INPUT questions analyzed."
                    )
                )
            ]
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
            ]
        }


# ============================================================================
# NODE 4 — SYNTHESIS AGENT
# وكيل التلخيص التنفيذي
# ============================================================================

def synthesis_agent(state: State) -> Dict[str, Any]:
    """
    STAGE 4: توليد تقرير تنفيذي ملخّص يجمع نتائج:
      - تحليل أسئلة الخيارات (selection_results)
      - تحليل أسئلة النصوص (analysis_results)
    """
    try:
        print("=" * 60)
        print("SYNTHESIS AGENT")
        print("=" * 60)
        print(f"Survey ID: {state['survey_id']}")

        text_questions_result      = state.get("text_questions_result", {})
        selection_questions_result = state.get("selection_questions_result", {})

        if not text_questions_result and not selection_questions_result:
            return Command(
                update={
                    "messages": [
                        AIMessage(
                            content="No analysis results to synthesize.",
                            name="synthesis_agent"
                        )
                    ]
                }
            )

        # استخراج عنوان الاستبيان
        survey_title = f"Survey {state['survey_id']}"
        if text_questions_result:
            first_result = next(iter(text_questions_result.values()))
            survey_title = first_result.get("survey_title", survey_title)

        print(f"Survey: {survey_title}")
        print(f"Text questions analyzed  : {len(text_questions_result)}")
        print(f"Selection questions count: {selection_questions_result.get('questions_count', 0)}")

        # تنسيق التحليلات النصية (قائمة من الرسائل)
        analytics_messages = format_analytics_messages(text_questions_result)

        prompt_messages = synthesis_agent_prompt.invoke({
            "survey_subject":             survey_title,
            "selection_questions_result": selection_questions_result,
            "text_questions_result":      analytics_messages
        })

        response = model.invoke(prompt_messages)
        synthesis_content = response.content

        report_path = save_report(synthesis_content, state['survey_id'])
        print("✅ Synthesis complete")

        # ── حفظ snapshot للاختبار (يُقرأ لاحقاً بواسطة test_generate_charts_agent.py) ──
        try:
            snapshot = {
                "survey_id":                  state["survey_id"],
                "text_questions_result":      state.get("text_questions_result", {}),
                "selection_questions_result": state.get("selection_questions_result", {}),
            }
            snapshot_path = os.path.join(src_dir, "tests", "chart_agent_snapshot.json")
            os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2, default=str)
            print(f"💾 State snapshot saved → {snapshot_path}")
        except Exception as snap_err:
            print(f"⚠️ Could not save snapshot: {snap_err}")

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
            update={
                "messages": [
                    AIMessage(content=error_msg, name="synthesis_agent")
                ]
            }
        )


# ============================================================================
# NODE 5 — GENERATE CHARTS AGENT
# وكيل توليد الرسوم البيانية
# ============================================================================

def generate_charts_agent(state: State) -> Command[Literal["__end__"]]:
    """
    STAGE 5: توليد إعدادات الرسوم البيانية بناءً على نتائج التحليل.
    يستخدم نموذج لغوي لإنشاء إعدادات Chart.js المتوافقة مع Vue.js.
    """
    try:
        print("=" * 60)
        print("CHART GENERATION AGENT")
        print("=" * 60)
        print(f"Survey ID: {state['survey_id']}")

        text_questions_result      = state.get("text_questions_result", {})
        selection_questions_result = state.get("selection_questions_result", {})

        if not text_questions_result and not selection_questions_result:
            print("⚠️ No analysis results available for chart generation")
            return Command(
                update={
                    "chart_configs": [],
                    "messages": [
                        AIMessage(
                            content="No analysis data available to generate charts.",
                            name="chart_generation_agent"
                        )
                    ]
                },
                goto="__end__"
            )

        # استخراج عنوان الاستبيان
        survey_title = f"Survey {state['survey_id']}"
        if text_questions_result:
            first_result = next(iter(text_questions_result.values()))
            survey_title = first_result.get("survey_title", survey_title)

        print(f"Survey: {survey_title}")

        # تنسيق ملخص التحليل للنموذج اللغوي
        analytics_summary = format_analysis_summary(text_questions_result)
        print(f"Analytics summary length: {len(analytics_summary)} chars")

        prompt_messages = chart_generation_prompt.invoke({
            "survey_subject": survey_title,
            "analytics_summary": analytics_summary
        })

        print("🤖 Invoking LLM for chart generation...")
        response = model.invoke(prompt_messages)
        response_text = response.content
        print(f"📥 LLM response length: {len(response_text)} chars")

        chart_configs = extract_json_from_response(response_text)

        if len(chart_configs) < 4:
            print(f"⚠️ Only {len(chart_configs)} charts generated, expected 4-6.")
        elif len(chart_configs) > 6:
            print(f"⚠️ {len(chart_configs)} charts generated, trimming to 6.")
            chart_configs = chart_configs[:6]

        print(f"✅ Generated {len(chart_configs)} chart configurations")
        for i, chart in enumerate(chart_configs):
            print(f"   {i+1}. {chart.get('type','?')}: {chart.get('title','Untitled')}")

        return Command(
            update={
                "chart_configs": chart_configs,
                "messages": [
                    AIMessage(
                        content=f"Generated {len(chart_configs)} chart configurations successfully.",
                        name="chart_generation_agent"
                    )
                ]
            },
            goto="__end__"
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
                ]
            },
            goto="__end__"
        )