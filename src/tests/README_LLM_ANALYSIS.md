# دليل استخدام دوال التحليل بالذكاء الصناعي

## نظرة عامة

تم إضافة دوال جديدة لتحليل بيانات الاستبيانات باستخدام الذكاء الصناعي (LLM). هذه الدوال تستخرج البيانات مع السياق الكامل وترسلها إلى LLM للحصول على تحليل شامل.

## الدوال المتاحة

### 1. تحليل المواضيع (Topics)

```python
# استخراج المواضيع مع السياق الكامل
topics_data = get_topics_with_full_context_for_llm(
    df, 
    sentiment_label='negative',  # أو 'positive' أو 'neutral'
    top_n=5  # عدد المواضيع المطلوبة
)

# تحليل باستخدام LLM (طريقة مخصصة)
analysis = analyze_topics_with_llm(
    topics_data,
    survey_title="عنوان الاستبيان",
    survey_question="السؤال المطروح"
)
```

### 2. تحليل الكيانات (Entities)

```python
# استخراج الكيانات (أسماء، منتجات، منظمات) مع السياق الكامل
entities_data = get_entities_with_full_context_for_llm(
    df,
    sentiment_label='negative',
    top_n=5
)

# تحليل باستخدام LLM (طريقة عامة)
analysis = analyze_data_with_llm(
    entities_data,
    data_type='entities',
    survey_title="عنوان الاستبيان",
    survey_question="السؤال المطروح"
)
```

### 3. تحليل الكلمات (Words)

```python
# استخراج الكلمات الأكثر تكراراً مع السياق الكامل
words_data = get_words_with_full_context_for_llm(
    df,
    sentiment_label='negative',
    top_n=5
)

# تحليل باستخدام LLM
analysis = analyze_data_with_llm(
    words_data,
    data_type='words',
    survey_title="عنوان الاستبيان",
    survey_question="السؤال المطروح"
)
```

## الفرق بين الدوال القديمة والجديدة

### الدوال القديمة (بدون LLM)
- `get_top_topics()` - تعيد فقط عدد التكرارات
- `get_top_negative_entities()` - تعيد فقط قائمة الكيانات
- `get_top_words_in_text()` - تعيد فقط الكلمات الأكثر تكراراً

### الدوال الجديدة (مع LLM)
- `get_topics_with_full_context_for_llm()` - تعيد المواضيع مع **جميع** الإجابات
- `get_entities_with_full_context_for_llm()` - تعيد الكيانات مع **جميع** الإجابات
- `get_words_with_full_context_for_llm()` - تعيد الكلمات مع **جميع** الإجابات
- `analyze_data_with_llm()` - تحلل البيانات باستخدام LLM وتعيد تحليل نصي شامل

## مثال كامل

```python
import pandas as pd
from analsysis_functions import (
    get_topics_with_full_context_for_llm,
    analyze_topics_with_llm
)

# قراءة البيانات
df = pd.read_csv('exports/survey_data_enriched.csv')

# معلومات الاستبيان
survey_title = "تقييم تجربة التاجر"
survey_question = "رأيك يهمنا"

# استخراج المواضيع السلبية
topics_data = get_topics_with_full_context_for_llm(
    df, 
    sentiment_label='negative', 
    top_n=5
)

# تحليل باستخدام LLM
if not isinstance(topics_data, str):
    analysis = analyze_topics_with_llm(
        topics_data,
        survey_title,
        survey_question
    )
    
    # حفظ التحليل
    with open('exports/analysis_result.txt', 'w', encoding='utf-8') as f:
        f.write(analysis)
    
    print("✅ تم التحليل بنجاح!")
```

## الملفات المُنتجة

عند تشغيل التحليل، سيتم إنشاء الملفات التالية:

1. **البرومبت المرسل إلى LLM:**
   - `exports/llm_prompt_topics_input.txt`
   - `exports/llm_prompt_entities_input.txt`
   - `exports/llm_prompt_words_input.txt`

2. **نتائج التحليل:**
   - `exports/llm_analysis_topics.txt`
   - `exports/llm_analysis_entities.txt`
   - `exports/llm_analysis_words.txt`

## ملاحظات مهمة

1. **السياق الكامل:** الدوال الجديدة ترسل **جميع** الإجابات المتعلقة بكل موضوع/كيان/كلمة، وليس فقط عينات
2. **حفظ البرومبت:** يتم حفظ البرومبت الكامل المرسل إلى LLM في ملف منفصل للمراجعة
3. **طباعة في الكونسل:** يتم طباعة البرومبت الكامل في الكونسل قبل إرساله إلى LLM
4. **القيمة الافتراضية:** `top_n=5` هي القيمة الافتراضية لجميع الدوال

## مثال متقدم: تحليل متعدد

راجع ملف `llm_analysis_example.py` لمثال كامل يوضح كيفية تحليل:
- المواضيع السلبية
- الكيانات السلبية
- الكلمات السلبية
- المواضيع الإيجابية

في نفس الوقت وحفظ كل تحليل في ملف منفصل.
