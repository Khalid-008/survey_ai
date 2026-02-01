# مثال على استخدام دوال التحليل بالذكاء الصناعي
# ================================================

"""
هذا الملف يوضح كيفية استخدام جميع دوال التحليل بالذكاء الصناعي المتوفرة في analsysis_functions.py

الدوال المتاحة:
1. تحليل المواضيع (Topics)
2. تحليل الكيانات (Entities) 
3. تحليل الكلمات (Words)
"""

import pandas as pd
from analsysis_functions import (
    get_topics_with_full_context_for_llm,
    get_entities_with_full_context_for_llm,
    get_words_with_full_context_for_llm,
    analyze_topics_with_llm,
    analyze_data_with_llm
)

# ============================================================================
# إعدادات الاستبيان
# ============================================================================

input_file = r'exports/survey_data_enriched_S25120024_20260129_174603.csv'
survey_title = "تقييم تجربة التاجر مع مندوب المبيعات - موبايل شوب"
survey_question = "رأيك يهمنا، فضلاً شاركنا تعليق او مقترح يساعد في تحسين التجربة"

# قراءة البيانات
df = pd.read_csv(input_file)

# ============================================================================
# 1. تحليل المواضيع السلبية
# ============================================================================

print("=" * 80)
print("1️⃣  تحليل المواضيع السلبية")
print("=" * 80)

# استخراج المواضيع مع السياق الكامل
topics_data = get_topics_with_full_context_for_llm(
    df, 
    sentiment_label='negative', 
    top_n=5
)

# تحليل باستخدام LLM
if not isinstance(topics_data, str):
    topics_analysis = analyze_topics_with_llm(
        topics_data,
        survey_title,
        survey_question
    )
    
    # حفظ النتيجة
    with open('exports/llm_analysis_topics.txt', 'w', encoding='utf-8') as f:
        f.write(topics_analysis)
    print("✅ تم حفظ تحليل المواضيع")

# ============================================================================
# 2. تحليل الكيانات السلبية
# ============================================================================

print("\n" + "=" * 80)
print("2️⃣  تحليل الكيانات السلبية")
print("=" * 80)

# استخراج الكيانات مع السياق الكامل
entities_data = get_entities_with_full_context_for_llm(
    df,
    sentiment_label='negative',
    top_n=5
)

# تحليل باستخدام LLM
if not isinstance(entities_data, str):
    entities_analysis = analyze_data_with_llm(
        entities_data,
        data_type='entities',
        survey_title=survey_title,
        survey_question=survey_question
    )
    
    # حفظ النتيجة
    with open('exports/llm_analysis_entities.txt', 'w', encoding='utf-8') as f:
        f.write(entities_analysis)
    print("✅ تم حفظ تحليل الكيانات")

# ============================================================================
# 3. تحليل الكلمات السلبية
# ============================================================================

print("\n" + "=" * 80)
print("3️⃣  تحليل الكلمات السلبية")
print("=" * 80)

# استخراج الكلمات مع السياق الكامل
words_data = get_words_with_full_context_for_llm(
    df,
    sentiment_label='negative',
    top_n=5
)

# تحليل باستخدام LLM
if not isinstance(words_data, str):
    words_analysis = analyze_data_with_llm(
        words_data,
        data_type='words',
        survey_title=survey_title,
        survey_question=survey_question
    )
    
    # حفظ النتيجة
    with open('exports/llm_analysis_words.txt', 'w', encoding='utf-8') as f:
        f.write(words_analysis)
    print("✅ تم حفظ تحليل الكلمات")

# ============================================================================
# 4. تحليل المواضيع الإيجابية (مثال إضافي)
# ============================================================================

print("\n" + "=" * 80)
print("4️⃣  تحليل المواضيع الإيجابية")
print("=" * 80)

# استخراج المواضيع الإيجابية
positive_topics_data = get_topics_with_full_context_for_llm(
    df,
    sentiment_label='positive',
    top_n=5
)

# تحليل باستخدام LLM
if not isinstance(positive_topics_data, str):
    positive_analysis = analyze_data_with_llm(
        positive_topics_data,
        data_type='topics',
        survey_title=survey_title,
        survey_question=survey_question
    )
    
    # حفظ النتيجة
    with open('exports/llm_analysis_positive_topics.txt', 'w', encoding='utf-8') as f:
        f.write(positive_analysis)
    print("✅ تم حفظ تحليل المواضيع الإيجابية")

print("\n" + "=" * 80)
print("✅ اكتمل التحليل بنجاح!")
print("=" * 80)
