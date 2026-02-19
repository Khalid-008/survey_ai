import os
import sys
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

# استيراد الأدوات من الملفات المنفصلة
from analytics_pipeline.data_processing.data_cleaner import clean_survey_data
from analytics_pipeline.data_processing.data_enricher import process_survey_questions
from helper.utils import make_serializable
from analytics_pipeline.util.synthesis_utils import format_analytics_messages, save_report
from analytics_pipeline.util.chart_utils import extract_json_from_response, format_analysis_summary

warnings.filterwarnings('ignore')


# ============================================================================
# WORKFLOW NODES
# ============================================================================


def retrieve_survey_question(state: State) -> Dict[str, Any]:
    """
    STAGE 1: استرجاع وتنظيف بيانات الاستبيان.

    هذه طبقة تنظيف البيانات - لا يحدث إثراء هنا.
    """
    try:
        survey_id = state["survey_id"]
        survey_df = get_survey_df(survey_id)

        rows_before = len(survey_df)
        print(f"📊 Loaded {rows_before} rows for survey {survey_id}")

        # تنظيف البيانات
        survey_df = clean_survey_data(survey_df)

        rows_after = len(survey_df)
        print(f"✨ Cleaned data: {rows_after} rows ({rows_before - rows_after} duplicates removed)")

        # التحقق من البيانات
        if survey_df.empty:
            return {
                "survey_data": [],
                "analysis_results": {},
                "messages": [
                    AIMessage(content=f"No data found for survey {survey_id}. Analysis skipped.")
                ]
            }

        # تحويل إلى قاموس (لم يُحفظ بعد - مجرد تحضير)
        survey_data = survey_df.to_dict(orient="records")

        return {
            "survey_data": survey_data,
            "analysis_results": {},
            "messages": [
                AIMessage(
                    content=f"Survey data retrieved and cleaned: {rows_after} responses ready for analysis."
                )
            ]
        }

    except Exception as e:
        error_msg = f"Failed to retrieve survey data: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        raise RuntimeError(error_msg)


def enrich_data(state: State) -> Dict[str, Any]:
    """
    STAGE 2: تطبيق إثراء NLP على بيانات الاستبيان.

    يعالج أسئلة TEXT_INPUT بتحليل المشاعر وNER واستخراج المواضيع.
    البيانات تُحفظ فقط بعد اكتمال الإثراء.
    """
    try:
        # التحقق من المدخلات
        if not state.get("survey_data"):
            return {
                "survey_data": [],
                "messages": [AIMessage(content="No survey data available for enrichment.")]
            }

        survey_df = pd.DataFrame(state["survey_data"])
        
        # سجلات لتصحيح المسار ومعرفة أنواع الأسئلة
        print(f"🔬 Unique QuestionTypes found: {survey_df['QuestionType'].unique()}")
        print(f"🔬 Starting enrichment for: {state.get('survey_title', 'Unknown Survey')}")

        if survey_df.empty:
            return {
                "survey_data": [],
                "messages": [AIMessage(content="Survey data is empty, skipping enrichment.")]
            }

        # الحصول على بيانات الاستبيان الوصفية
        survey_title = (
            survey_df["SurveyTitle"].iloc[0]
            if "SurveyTitle" in survey_df.columns
            else f"Survey {state['survey_id']}"
        )

        print(f"🔬 Starting enrichment for: {survey_title}")

        # معالجة جميع الأسئلة
        enriched_df, analysis_results, metrics = process_survey_questions(
            survey_df, survey_title
        )

        # طباعة المقاييس
        print(f"📈 Enrichment complete:")
        print(f"   • TEXT_INPUT questions: {metrics.text_input_questions}")
        print(f"   • Successfully enriched: {metrics.enriched_questions}")
        print(f"   • Failed: {metrics.failed_questions}")

        # تسلسل النتائج
        enriched_data = enriched_df.to_dict(orient="records")
        serializable_analysis = make_serializable(analysis_results)

        return {
            "survey_data": enriched_data,
            "analysis_results": serializable_analysis,
            "messages": [
                AIMessage(
                    content=f"Data enriched: {metrics.enriched_questions}/{metrics.text_input_questions} "
                            f"TEXT_INPUT questions analyzed successfully."
                )
            ]
        }

    except Exception as e:
        error_msg = f"Enrichment failed: {str(e)}"
        print(f"⚠️ {error_msg}")
        print(traceback.format_exc())

        # خطأ غير حرج: المتابعة بالبيانات الأساسية
        return {
            "survey_data": state.get("survey_data", []),
            "analysis_results": state.get("analysis_results", {}),
            "messages": [
                AIMessage(
                    content=f"Warning: {error_msg}. Proceeding with basic data."
                )
            ]
        }


def synthesis_agent(state: State) -> Dict[str, Any]:
    """
    STAGE 3: توليد تقرير تنفيذي ملخّص.

    يلخّص جميع نتائج التحليل في تقرير تنفيذي متكامل.
    """
    try:
        print("=" * 60)
        print("SYNTHESIS AGENT")
        print("=" * 60)
        print(f"Survey ID: {state['survey_id']}")

        analysis_results = state.get("analysis_results", {})

        if not analysis_results:
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

        # استخراج بيانات الاستبيان الوصفية
        first_result = next(iter(analysis_results.values()))
        survey_title = first_result.get("survey_title", f"Survey {state['survey_id']}")

        print(f"Survey: {survey_title}")
        print(f"Questions analyzed: {len(analysis_results)}")

        # تنسيق التحليلات للنموذج اللغوي
        analytics_messages = format_analytics_messages(analysis_results)

        # توليد التلخيص
        prompt_messages = synthesis_agent_prompt.invoke({
            "survey_subject": survey_title,
            "analytics_messages": analytics_messages
        })

        response = model.invoke(prompt_messages)
        synthesis_content = response.content

        # حفظ التقرير
        report_path = save_report(synthesis_content, state['survey_id'])

        print("✅ Synthesis complete")

        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=synthesis_content,
                        name="synthesis_agent"
                    )
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
                    AIMessage(
                        content=error_msg,
                        name="synthesis_agent"
                    )
                ]
            }
        )


def generate_charts_agent(state: State) -> Command[Literal["__end__"]]:
    """
    STAGE 4: توليد إعدادات الرسوم البيانية بناءً على نتائج التحليل.

    يستخدم نموذج لغوي لإنشاء إعدادات Chart.js التي تتوافق مع مكونات Vue.js.
    """
    try:
        print("=" * 60)
        print("CHART GENERATION AGENT")
        print("=" * 60)
        print(f"Survey ID: {state['survey_id']}")

        analysis_results = state.get("analysis_results", {})

        if not analysis_results:
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

        # استخراج بيانات الاستبيان الوصفية
        first_result = next(iter(analysis_results.values()))
        survey_title = first_result.get("survey_title", f"Survey {state['survey_id']}")

        print(f"Survey: {survey_title}")
        print(f"Questions analyzed: {len(analysis_results)}")

        # تنسيق ملخص التحليل للنموذج اللغوي
        analytics_summary = format_analysis_summary(analysis_results)

        print(f"Analytics summary length: {len(analytics_summary)} chars")

        # توليد إعدادات الرسوم البيانية باستخدام النموذج اللغوي
        prompt_messages = chart_generation_prompt.invoke({
            "survey_subject": survey_title,
            "analytics_summary": analytics_summary
        })

        print("🤖 Invoking LLM for chart generation...")
        response = model.invoke(prompt_messages)
        response_text = response.content

        print(f"📥 LLM response length: {len(response_text)} chars")

        # تحليل رد JSON
        chart_configs = extract_json_from_response(response_text)

        # التحقق من عدد الرسوم (يجب أن يكون 4-6)
        if len(chart_configs) < 4:
            print(f"⚠️ Only {len(chart_configs)} charts generated, expected 4-6. Using what we have.")
        elif len(chart_configs) > 6:
            print(f"⚠️ {len(chart_configs)} charts generated, trimming to 6.")
            chart_configs = chart_configs[:6]

        print(f"✅ Generated {len(chart_configs)} chart configurations")

        # طباعة أنواع الرسوم
        for i, chart in enumerate(chart_configs):
            chart_type = chart.get("type", "unknown")
            chart_title = chart.get("title", "Untitled")
            print(f"   {i+1}. {chart_type}: {chart_title}")

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
                    AIMessage(
                        content=error_msg,
                        name="chart_generation_agent"
                    )
                ]
            },
            goto="__end__"
        )