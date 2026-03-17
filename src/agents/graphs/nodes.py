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



from llms.models import model, qwen3_model


from agents.graphs.setup import State


from data.operations import get_survey_df


from agents.prompt.synthesis_agent_prompt import synthesis_agent_prompt


from agents.prompt.chart_generation_prompt import chart_generation_prompt


from agents.prompt.analysis_questions_recommender_prompt import analysis_questions_recommender_prompt



# معالجة بيانات النصوص


from analytics_pipeline.data_processing.data_cleaner import clean_survey_data


from analytics_pipeline.data_processing.data_enricher import enrich_survey_df, get_analysis_results, get_survey_metrics



# معالجة بيانات الخيارات


from analytics_pipeline.selection_pipeline import (

    generate_sql_analisys_queries,

    group_answers_by_question,

    get_distinct_answers_per_question,

    get_sample_answers_per_question,
)


from analytics_pipeline.util.selection_utils import format_questions_block



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

    STAGE 1: استرجاع بيانات الاستبيان الخام وتصنيف الأسئلة إلى مجموعتين:

      • text_questions_data     — أسئلة TEXT_INPUT

      • selection_questions_data — باقي أنواع الأسئلة (اختيارات)
    """


    try:


        survey_id = state["survey_id"]


        date_from = state.get("date_from")


        date_to   = state.get("date_to")



        print("=" * 60)

        print("RETRIEVE SURVEY QUESTIONS NODE")

        print("=" * 60)

        print(f"📋 Survey ID : {survey_id}"
)
        if date_from or date_to:

            print(f"📅 Date filter: {date_from} → {date_to}")


        survey_df = get_survey_df(survey_id, date_from=date_from, date_to=date_to)


        rows = len(survey_df)


        print(f"📊 Loaded {rows} rows from database")


        if survey_df.empty:


            print("⚠️  No data found — returning empty groups")


            return {


                "survey_data":             [],


                "text_questions_data":      [],


                "selection_questions_data": [],


                "messages": [


                    AIMessage(content=f"No data found for survey {survey_id}. Analysis skipped.")


                ]


            }



        survey_data = survey_df.to_dict(orient="records")



        # ── تصنيف الأسئلة إلى مجموعتين ──────────────────────────────────────


        all_types = survey_df["question_type"].str.upper().str.strip().unique().tolist()

        print(f"🔍 All question types detected: {all_types}")



        TEXT_TYPE = "TEXT_INPUT"


        # ── المجموعة 1: أسئلة نصية (TEXT_INPUT) ──

        text_mask           = survey_df["question_type"].str.upper().str.strip() == TEXT_TYPE

        text_df             = survey_df[text_mask]

        text_questions_data = text_df.to_dict(orient="records")


        text_q_ids = (

            text_df["question_id"].unique().tolist()

            if "question_id" in text_df.columns else []
        )


        # ── المجموعة 2: أسئلة الاختيارات (كل ما ليس TEXT_INPUT) ──

        selection_df             = survey_df[~text_mask]

        selection_questions_data = selection_df.to_dict(orient="records")


        sel_q_ids = (

            selection_df["question_id"].unique().tolist()

            if "question_id" in selection_df.columns else []
        )

        sel_types = (

            selection_df["question_type"].str.upper().str.strip().unique().tolist()

            if not selection_df.empty else []
        )



        print("-" * 60)

        print(f"✅ Classification complete")

        print(f"   • Total rows     : {rows}")

        print(f"   • Text rows      : {len(text_questions_data)} ({len(text_q_ids)} questions)")

        print(f"   • Selection rows : {len(selection_questions_data)} ({len(sel_q_ids)} questions)")

        print("=" * 60)



        return {


            "survey_data":             survey_data,

            "text_questions_data":      text_questions_data,

            "selection_questions_data": selection_questions_data,

            "messages": [

                AIMessage(
                    content=(

                        f"Survey data retrieved: {rows} rows. "

                        f"Classified → Text: {len(text_questions_data)} rows ({len(text_q_ids)} questions), "

                        f"Selection: {len(selection_questions_data)} rows ({len(sel_q_ids)} questions)."
                    )
                )

            ]

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
    """

    تحويل selection_questions_data إلى DataFrame وتجميع الإجابات

    وبناء questions_block — ثم تمرير النتائج للـ state.
    """

    survey_id = state["survey_id"]


    print("=" * 60)

    print("PREPARE SELECTION DATA NODE")

    print("=" * 60)


    raw_data  = state.get("selection_questions_data", [])


    # 1. تحويل البيانات وتحقق

    print("📥 Step 1 — Preparing selection questions data...")


    if isinstance(raw_data, list):

        df = pd.DataFrame(raw_data)

    else:

        df = raw_data


    if df.empty:

        print("⚠️  No selection questions found — skipping.")

        return {

            "selection_prepared": {},

            "messages": [AIMessage(content="No selection questions found. Skipping selection analysis.")],

        }


    print(f"   ✅ {len(df)} rows | {df['question_id'].nunique()} questions")


    # 2. Group & distinct

    print("\n📊 Step 2 — Grouping answers...")


    grouped  = group_answers_by_question(df)

    distinct = get_distinct_answers_per_question(grouped)

    samples  = get_sample_answers_per_question(df, n=3)


    questions_block = format_questions_block(
        grouped, distinct, sample_answers=samples, survey_number=str(survey_id)
    )


    return {

        "selection_prepared": {

            "questions_block": questions_block,

        },

        "messages": [

            AIMessage(
                content=(

                    f"Selection data prepared: {len(grouped)} questions, "

                    f"{sum(len(v) for v in distinct.values())} total distinct answers."

                ),

                name="prepare_selection_data"
            )

        ],

    }



# ============================================================================


# NODE 3 — ANALYZE SELECTION QUESTIONS


# تحليل أسئلة الخيارات


# ============================================================================



def analyze_selection_questions(state: State) -> Dict[str, Any]:


    try:


        survey_id = state["survey_id"]



        print("=" * 60)


        print("SELECTION ANALYSIS NODE")


        print("=" * 60)


        print(f"Survey ID: {survey_id}")



        # Step A — توليد الأسئلة التحليلية
        questions_block = state.get("selection_prepared", {}).get("questions_block", "")

        print("\n🧠 Generating analytical questions from questions_block...")

        try:
            q_prompt   = analysis_questions_recommender_prompt.invoke({"questions_block": questions_block})
            q_response = model.invoke(q_prompt)
            analytical_questions = q_response.content
            print(f"   ✅ Analytical questions generated ({len(analytical_questions)} chars).")
        except Exception as e:
            analytical_questions = ""
            print(f"   ⚠️ Could not generate analytical questions: {e}")

        # Step B — توليد SQL queries بناءً على الأسئلة التحليلية
        queries_results = generate_sql_analisys_queries(
            survey_id=survey_id,
            prepared=state.get("selection_prepared", {}),
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


            querys_results_serialized.append({
                "label":  entry.get("label", ""),
                "sql":    entry.get("sql", ""),
                "result": rows,
            })



        serializable_result = {
            "query_results":         querys_results_serialized,
            "errors":                queries_results["errors"]
        }



        print(f"✅ Selection analysis complete: {len(querys_results_serialized)} queries, {total_rows} total rows")



        return {
            "selection_questions_result": serializable_result,
            "messages": [
                AIMessage(
                    content=(
                        f"Selection analysis complete: {len(querys_results_serialized)} queries, "
                        f"Total rows: {total_rows}."
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



    try:


        print("=" * 60)


        print("TEXT ANALYSIS NODE")


        print("=" * 60)



        if not state.get("text_questions_data"):


            return {


                "text_questions_data": [],


                "text_questions_result": {},


                "messages": [AIMessage(content="No survey data available for text analysis.")]


            }



        text_questions_data = pd.DataFrame(state["text_questions_data"])



        # أ) تنظيف البيانات


        rows_before = len(text_questions_data)


        print(f"📊 Loaded {rows_before} rows for survey {state['survey_id']}")



        text_questions_data = clean_survey_data(text_questions_data)


        rows_after = len(text_questions_data)


        print(f"✨ Cleaned data: {rows_after} rows ({rows_before - rows_after} duplicates removed)")



        if text_questions_data.empty:


            return {


                "survey_data": [],


                "text_questions_result": {},


                "messages": [AIMessage(content="Survey data is empty after cleaning.")]


            }




        survey_title = (


            text_questions_data["survey_title"].iloc[0]


            if "survey_title" in text_questions_data.columns


            else f"Survey {state['survey_id']}"
        )



        print(f"🔬 Starting enrichment for: {survey_title}")


        enriched_df       = enrich_survey_df(text_questions_data, survey_title)

        analysis_results  = get_analysis_results(text_questions_data, survey_title)

        metrics           = get_survey_metrics(text_questions_data, survey_title)


        print(f"📈 Enrichment complete:"
)
        print(f"   • TEXT_INPUT questions: {metrics.text_input_questions}")

        print(f"   • Successfully enriched: {metrics.enriched_questions}")

        print(f"   • Failed              : {metrics.failed_questions}")


        enriched_data         = enriched_df.to_dict(orient="records")

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

        text_questions_result = format_analytics_messages(text_questions_result)


        prompt_messages = synthesis_agent_prompt.invoke({


            "survey_subject":             survey_title,


            "selection_questions_result": selection_questions_result,


            "text_questions_result":      text_questions_result

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


        analytics_summary = format_analysis_summary(text_questions_result, selection_questions_result)


        print(f"Analytics summary length: {len(analytics_summary)} chars")



        prompt_messages = chart_generation_prompt.invoke({


            "survey_subject": survey_title,


            "analytics_summary": analytics_summary


        })



        print("🤖 Invoking LLM for chart generation...")


        response = qwen3_model.invoke(prompt_messages)


        response_text = response.content


        print(f"📥 LLM response length: {len(response_text)} chars")



        chart_configs = extract_json_from_response(response_text)



        if len(chart_configs) < 5:


            print(f"⚠️ Only {len(chart_configs)} charts generated, expected exactly 5.")


        elif len(chart_configs) > 5:


            print(f"⚠️ {len(chart_configs)} charts generated, trimming to 5.")


            chart_configs = chart_configs[:5]



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