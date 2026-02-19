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


def process_survey_questions(survey_df, survey_title):
    """
    معالجة جميع أسئلة الاستبيان مع إثراء أسئلة TEXT_INPUT.

    يُرجع: (enriched_df, analysis_results, metrics)
    """
    enriched_groups = []
    all_analysis_results = {}
    text_input_count = 0
    enriched_count = 0
    failed_count = 0

    for question_id, group_df in survey_df.groupby("QuestionID"):
        question_type = str(group_df["QuestionType"].iloc[0]).upper().strip()
        question_text = group_df["Questions"].iloc[0]

        if question_type == "TEXT_INPUT":
            text_input_count += 1
            try:
                results = enrich_text_question(
                    group_df, survey_title, question_text, str(question_id)
                )
                enriched_groups.append(results['enriched_df'])
                all_analysis_results[str(question_id)] = results
                enriched_count += 1

            except Exception as e:
                print(f"⚠️ Failed to enrich question {question_id}, using raw data")
                enriched_groups.append(add_default_enrichment_columns(group_df))
                failed_count += 1
        else:
            # الأسئلة غير النصية تحصل على أعمدة إثراء افتراضية
            enriched_groups.append(add_default_enrichment_columns(group_df))

    enriched_df = pd.concat(enriched_groups, ignore_index=True)

    metrics = SurveyMetrics(
        rows_before_cleaning=0,   # يُحدد من قِبل المُستدعي
        rows_after_cleaning=0,    # يُحدد من قِبل المُستدعي
        text_input_questions=text_input_count,
        enriched_questions=enriched_count,
        failed_questions=failed_count
    )

    return enriched_df, all_analysis_results, metrics
