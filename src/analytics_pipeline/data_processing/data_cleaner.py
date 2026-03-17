import os
import sys

import pandas as pd

# إعداد المسار للاستيراد
current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from analytics_pipeline.util.config import ANSWER_MAPPING, CLEANING_PATTERNS
from analytics_pipeline.text_pipeline import normalize_arabic, convert_arabic_time_to_24h


# ============================================================================
# DATA CLEANING
# تنظيف بيانات الاستبيان
# ============================================================================


def normalize_answers(df):
    """تطبيق التطبيع على إجابات الاستبيان."""
    df = df.copy()

    # تحويل الوقت العربي يجب أن يحدث قبل التطبيع العام
    df["Answer_normalized"] = df["Answer"].apply(convert_arabic_time_to_24h)
    df["Answer_normalized"] = df["Answer_normalized"].apply(normalize_arabic)

    # معالجة القيم الفارغة
    df["Answer_normalized"] = df["Answer_normalized"].fillna("NO ANSWER")

    # تطبيع حالة الأحرف
    df["Answer_normalized"] = df["Answer_normalized"].str.lower().str.strip()

    return df


def apply_answer_mapping(df):
    """تعيين الإجابات العربية الشائعة إلى قيم موحدة."""
    df = df.copy()
    df["Answer_normalized"] = df["Answer_normalized"].replace(ANSWER_MAPPING)
    return df


def clean_text_patterns(df):
    """إزالة الأنماط غير المرغوبة من النص."""
    df = df.copy()

    for pattern, replacement in CLEANING_PATTERNS:
        df["Answer_normalized"] = df["Answer_normalized"].str.replace(
            pattern, replacement, regex=True
        )

    df["Answer_normalized"] = df["Answer_normalized"].str.strip()
    return df


def remove_duplicates(df):
    """إزالة الإجابات المكررة لنفس السؤال."""
    return df.drop_duplicates(subset=["QuestionID", "Answer_normalized"])


def clean_survey_data(df):
    """تشغيل مسار التنظيف الكامل على بيانات الاستبيان."""
    df = normalize_answers(df)
    df = apply_answer_mapping(df)
    df = clean_text_patterns(df)

    # فلترة جميع أسئلة TEXT_INPUT — لا يتم تضمينها في التحليل
    if "question_type" in df.columns:
        is_text = df["question_type"].str.upper().str.strip() == "TEXT_INPUT"
        df = df[~is_text].reset_index(drop=True)

    return df
