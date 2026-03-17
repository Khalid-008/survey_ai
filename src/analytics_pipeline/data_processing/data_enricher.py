import os
import sys

import pandas as pd

# إعداد المسار للاستيراد
# الملف موجود في: src/analytics_pipeline/data_processing/
# نحتاج الصعود 3 مستويات للوصول إلى: src/
current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from analytics_pipeline.util.config import DEFAULT_SENTIMENT, DEFAULT_TOPIC, DEFAULT_TOPIC_ID
from analytics_pipeline.text_pipeline import run_full_analysis_pipeline


# ============================================================================
# SURVEY METRICS
# مقاييس تتبع معالجة بيانات الاستبيان
# ============================================================================


class SurveyMetrics:
    """مقاييس لتتبع معالجة بيانات الاستبيان."""

    def __init__(
        self,
        rows_before_cleaning,
        rows_after_cleaning,
        text_input_questions,
        enriched_questions,
        failed_questions
    ):
        self.rows_before_cleaning = rows_before_cleaning
        self.rows_after_cleaning = rows_after_cleaning
        self.text_input_questions = text_input_questions
        self.enriched_questions = enriched_questions
        self.failed_questions = failed_questions


# ============================================================================
# DATA ENRICHMENT
# إثراء بيانات الاستبيان باستخدام NLP
# ============================================================================


def add_default_enrichment_columns(df):
    """إضافة أعمدة الإثراء الافتراضية للأسئلة غير النصية."""
    df = df.copy()

    if "sentiment" not in df.columns:
        df["sentiment"] = DEFAULT_SENTIMENT
    if "entities" not in df.columns:
        df["entities"] = "[]"
    if "topic_label" not in df.columns:
        df["topic_label"] = DEFAULT_TOPIC
    if "topic_id" not in df.columns:
        df["topic_id"] = DEFAULT_TOPIC_ID

    return df


def enrich_text_question(df, survey_title, question_text, question_id):
    """
    تشغيل مسار التحليل الكامل على سؤال نصي.

    يُرجع: قاموس يحتوي على enriched_df ونتائج التحليل
    """
    print(f"🔍 Analyzing TEXT_INPUT question [{question_id}]: {question_text[:50]}...")

    results = run_full_analysis_pipeline(df, survey_title, question_text)
    results['survey_question'] = question_text
    results['survey_title'] = survey_title
    results['question_id'] = question_id

    print(f"✅ Question {question_id} analyzed successfully")
    return results


def _run_enrichment(survey_df, survey_title):
    """
    دالة داخلية — تُشغّل الإثراء وتُرجع كل النتائج الخام في قاموس واحد.
    """
    enriched_groups = []
    all_analysis_results = {}
    text_input_count = 0
    enriched_count = 0
    failed_count = 0

    for question_id, group_df in survey_df.groupby("question_id"):
        question_type = str(group_df["question_type"].iloc[0]).upper().strip()
        question_text = group_df["question_ar"].iloc[0]

        if question_type == "TEXT_INPUT":
            text_input_count += 1
            try:
                results = enrich_text_question(
                    group_df, survey_title, question_text, str(question_id)
                )
                enriched_groups.append(results["enriched_df"])
                all_analysis_results[str(question_id)] = results
                enriched_count += 1
            except Exception as e:
                print(f"⚠️ Failed to enrich question {question_id}: {e}")
                enriched_groups.append(add_default_enrichment_columns(group_df))
                failed_count += 1
        else:
            enriched_groups.append(add_default_enrichment_columns(group_df))

    enriched_df = pd.concat(enriched_groups, ignore_index=True) if enriched_groups else pd.DataFrame()

    return {
        "enriched_df":          enriched_df,
        "analysis_results":     all_analysis_results,
        "text_input_count":     text_input_count,
        "enriched_count":       enriched_count,
        "failed_count":         failed_count,
    }


def enrich_survey_df(survey_df, survey_title):
    """يُرجع DataFrame مُثرى بأعمدة NLP."""
    return _run_enrichment(survey_df, survey_title)["enriched_df"]


def get_analysis_results(survey_df, survey_title):
    """يُرجع قاموس نتائج التحليل لكل سؤال TEXT_INPUT."""
    return _run_enrichment(survey_df, survey_title)["analysis_results"]


def get_survey_metrics(survey_df, survey_title):
    """يُرجع كائن SurveyMetrics يحتوي على إحصائيات المعالجة."""
    raw = _run_enrichment(survey_df, survey_title)
    return SurveyMetrics(
        rows_before_cleaning=0,
        rows_after_cleaning=0,
        text_input_questions=raw["text_input_count"],
        enriched_questions=raw["enriched_count"],
        failed_questions=raw["failed_count"],
    )

