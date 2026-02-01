import pandas as pd
import os
import re
import ast
from collections import Counter
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llms.models import model

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def get_sentiment_distribution(df):
    """1. حساب التوزيع العام للمشاعر (إيجابي / سلبي / محايد)"""
    if 'sentiment' not in df.columns:
        return "Sentiment column not found"
    
    dist = df['sentiment'].value_counts()
    perc = df['sentiment'].value_counts(normalize=True) * 100
    
    result = pd.DataFrame({
        'العدد': dist,
        'النسبة المئوية (%)': perc.round(2)
    })
    return result

def get_top_topics(df, top_n=10):
    """2. ما أكثر 10 مواضيع تكراراً؟"""
    if 'topic_label' not in df.columns:
        return "Topic column not found"
    
    # Clean topics: remove 'متنوع', 'غير محدد', and split if comma-separated
    topics_series = df['topic_label'].dropna().astype(str)
    
    all_topics = []
    for t in topics_series:
        # Split by comma if multiple keywords
        parts = [p.strip() for p in t.split(',')]
        for p in parts:
            if p not in ['متنوع', 'غير محدد', 'خطأ', '']:
                all_topics.append(p)
    
    top_topics = pd.Series(all_topics).value_counts().head(top_n)
    return top_topics

def get_top_topics_by_sentiment(df, sentiment_label='negative', top_n=5):
    """حساب أكثر المواضيع تكراراً لمشاعر معينة"""
    if 'sentiment' not in df.columns or 'topic_label' not in df.columns:
        return "Required columns not found"
    
    # Filter by sentiment
    filtered_df = df[df['sentiment'].str.contains(sentiment_label, case=False, na=False)]
    
    topics_series = filtered_df['topic_label'].dropna().astype(str)
    
    all_topics = []
    for t in topics_series:
        parts = [p.strip() for p in t.split(',')]
        for p in parts:
            if p not in ['متنوع', 'غير محدد', 'خطأ', '']:
                all_topics.append(p)
    
    return pd.Series(all_topics).value_counts().head(top_n)

def get_topics_with_full_context_for_llm(df, sentiment_label='negative', top_n=5):
    """
    استخراج المواضيع مع كل الإجابات الكاملة لإرسالها إلى LLM للتحليل
    Returns: dict with topic as key and list of all answers as value
    """
    if 'sentiment' not in df.columns or 'topic_label' not in df.columns:
        return "Required columns not found"
    
    # Filter by sentiment
    filtered_df = df[df['sentiment'].str.contains(sentiment_label, case=False, na=False)]
    
    # Build topic -> answers mapping with ALL occurrences
    topic_to_answers = {}
    
    for idx, row in filtered_df.iterrows():
        topic = str(row.get('topic_label', ''))
        answer = str(row.get('Answer_normalized', ''))
        
        # Split topics if comma-separated
        parts = [p.strip() for p in topic.split(',')]
        for t in parts:
            if t not in ['متنوع', 'غير محدد', 'خطأ', '', 'nan']:
                if t not in topic_to_answers:
                    topic_to_answers[t] = []
                topic_to_answers[t].append(answer)
    
    # Count and sort
    topic_counts = {topic: len(answers) for topic, answers in topic_to_answers.items()}
    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Build result with ALL answers for each topic
    result = {}
    for topic, count in sorted_topics:
        result[topic] = {
            'count': count,
            'answers': topic_to_answers[topic]  # ALL answers, not just samples
        }
    
    return result

def analyze_topics_with_llm(topics_data, survey_title, survey_question):
    """
    تحليل المواضيع باستخدام LLM
    
    Args:
        topics_data: dict from get_topics_with_full_context_for_llm
        survey_title: عنوان الاستبيان
        survey_question: السؤال المطروح في الاستبيان
    
    Returns:
        str: تحليل LLM للبيانات
    """
    # Build the prompt with full context
    prompt = f"""أنت محلل بيانات خبير متخصص في تحليل استبيانات رضا العملاء باللغة العربية.

    مهمتك: تحليل بيانات الاستبيان المرفقة واستخراج الرؤى الأساسية بطريقة واضحة ومفيدة.

    **معلومات الاستبيان:**
    - عنوان الاستبيان: "{survey_title}"
    - السؤال المطروح: "{survey_question}"

    ## خطوات التحليل:

    1. **فهم السياق:**
    - اقرأ عنوان الاستبيان والسؤال المطروح
    - افهم طبيعة البيانات والمواضيع المذكورة

    2. **تحديد المواضيع الرئيسية:**
    - صنف المواضيع حسب عدد التكرارات (من الأكثر إلى الأقل)
    - اجمع المواضيع المتشابهة معاً
    - اذكر عدد التكرارات لكل موضوع

    3. **تحليل تفاصيل كل موضوع:**
    - استخرج النقاط الفرعية لكل موضوع رئيسية
    - اذكر أمثلة محددة من الإجابات
    - سلط الضوء على أي أسماء أو حالات خاصة مذكورة

    4. **استخلاص الأنماط:**
    - ابحث عن أنماط متكررة في المواضيع
    - حدد أي تناقضات أو حالات استثنائية (مثل: تمييز، محاباة، إلخ)

    5. **تقديم الانطباع العام:**
    - لخص المزاج العام للعملاء/التجار
    - حدد مستوى الرضا/الاستياء
    - اذكر أهم نقاط الضعف التي تحتاج معالجة فورية

    ## صيغة العرض المطلوبة:

    استخدم العناوين والتنسيق التالي:

    ```
    ## المواضيع الرئيسية:

    **1. [اسم الموضوع] (عدد التكرارات)**
    - نقطة فرعية 1
    - نقطة فرعية 2
    - أي تفاصيل مهمة أو أمثلة محددة

    **2. [الموضوع الثانية]**
    ...

    ## الانطباع العام:
    [تلخيص واضح ومباشر للوضع]
    ```

    ## ملاحظات مهمة:
    - استخدم لغة عربية واضحة ومباشرة
    - ركز على الحقائق والأرقام
    - اذكر أي أسماء أو تفاصيل محددة تم ذكرها
    - كن موضوعياً ولا تتجاهل أي موضوع مهما كانت صغيرة
    - اجعل التحليل عملياً وقابلاً للتنفيذ

    ---

    **البيانات المطلوب تحليلها:**

    """
    
    # Add each topic with ALL its answers
    for topic, data in topics_data.items():
        prompt += f"\n{'='*80}\n"
        prompt += f"**الموضوع: {topic}**\n"
        prompt += f"**عدد التكرارات: {data['count']}**\n\n"
        prompt += "**جميع الإجابات المتعلقة بهذا الموضوع:**\n\n"
        
        for i, answer in enumerate(data['answers'], 1):
            prompt += f"{i}. {answer}\n"
    
    prompt += f"\n{'='*80}\n"

    # Call LLM
    print("\n🤖 إرسال البيانات إلى LLM للتحليل...")
    print(f"📊 عدد المواضيع: {len(topics_data)}")
    total_answers = sum(data['count'] for data in topics_data.values())
    print(f"📝 إجمالي عدد الإجابات: {total_answers}")
    
    # Save the full prompt to a file
    prompt_file = r'exports/llm_prompt_input.txt'
    try:
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f"💾 تم حفظ البرومبت الكامل في: {prompt_file}")
    except Exception as e:
        print(f"⚠️ خطأ في حفظ البرومبت: {str(e)}")
    
    # Print the prompt to console
    print("\n" + "="*80)
    print("📋 البيانات المرسلة إلى LLM:")
    print("="*80)
    print(prompt)
    print("="*80 + "\n")
    
    try:
        response = model.invoke(prompt)
        return response.content
    except Exception as e:
        return f"❌ خطأ في الاتصال بـ LLM: {str(e)}"

def get_entities_with_full_context_for_llm(df, sentiment_label='negative', top_n=5):
    """
    استخراج الكيانات مع كل الإجابات الكاملة لإرسالها إلى LLM للتحليل
    Returns: dict with entity as key and list of all answers as value
    """
    if 'sentiment' not in df.columns or 'entities' not in df.columns:
        return "Required columns not found"
    
    # Filter by sentiment
    filtered_df = df[df['sentiment'].str.contains(sentiment_label, case=False, na=False)]
    
    # Build entity -> answers mapping with ALL occurrences
    entity_to_answers = {}
    
    for idx, row in filtered_df.iterrows():
        answer = str(row.get('Answer_normalized', ''))
        try:
            entities = ast.literal_eval(row['entities'])
            if isinstance(entities, list):
                for ent in entities:
                    if ent['label'] in ['شخص', 'منتج', 'منظمة']:
                        entity_key = f"{ent['text']} ({ent['label']})"
                        if entity_key not in entity_to_answers:
                            entity_to_answers[entity_key] = []
                        entity_to_answers[entity_key].append(answer)
        except:
            continue
    
    # Count and sort
    entity_counts = {entity: len(answers) for entity, answers in entity_to_answers.items()}
    sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Build result with ALL answers for each entity
    result = {}
    for entity, count in sorted_entities:
        result[entity] = {
            'count': count,
            'answers': entity_to_answers[entity]  # ALL answers, not just samples
        }
    
    return result

def get_words_with_full_context_for_llm(df, sentiment_label='negative', top_n=5):
    """
    استخراج الكلمات مع كل الإجابات الكاملة لإرسالها إلى LLM للتحليل
    Returns: dict with word as key and list of all answers as value
    """
    if 'sentiment' not in df.columns or 'Answer_normalized' not in df.columns:
        return "Required columns not found"
    
    # Filter by sentiment
    filtered_df = df[df['sentiment'].str.contains(sentiment_label, case=False, na=False)]
    
    # Basic Arabic Stopwords
    stopwords = set([
        'في', 'من', 'على', 'إلى', 'عن', 'مع', 'ما', 'يا', 'ب', 'ل', 'و', 'ك', 
        'لا', 'نعم', 'كان', 'يكون', 'هذا', 'هذه', 'ذلك', 'الي', 'اللي', 'ان',
        'او', 'هل', 'كل', 'بعد', 'قبل', 'عند', 'حتى', 'بس', 'تم', 'تمت',
        'عدم', 'عدم وجود', 'التي', 'الذي', 'اذا', 'إذا', 'لو', 'غير', 'بدون',
        'ممكن', 'يوجد', 'ليس', 'لم', 'لن', 'أو', 'أن', 'كنت', 'كانت', 'يعني'
    ])
    
    # Build word -> answers mapping with ALL occurrences
    word_to_answers = {}
    
    for idx, row in filtered_df.iterrows():
        answer = str(row.get('Answer_normalized', ''))
        # Clean text: keep only Arabic letters
        clean_text = re.sub(r'[^\u0600-\u06FF\s]', '', answer)
        
        for word in clean_text.split():
            if len(word) > 2 and word not in stopwords:
                if word not in word_to_answers:
                    word_to_answers[word] = []
                word_to_answers[word].append(answer)
    
    # Count and sort
    word_counts = {word: len(answers) for word, answers in word_to_answers.items()}
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Build result with ALL answers for each word
    result = {}
    for word, count in sorted_words:
        result[word] = {
            'count': count,
            'answers': word_to_answers[word]  # ALL answers, not just samples
        }
    
    return result

def analyze_data_with_llm(data_dict, data_type, survey_title, survey_question):
    """
    تحليل أي نوع من البيانات باستخدام LLM (مواضيع، كيانات، كلمات)
    
    Args:
        data_dict: dict from get_*_with_full_context_for_llm functions
        data_type: نوع البيانات ('topics', 'entities', 'words')
        survey_title: عنوان الاستبيان
        survey_question: السؤال المطروح في الاستبيان
    
    Returns:
        str: تحليل LLM للبيانات
    """
    # Map data type to Arabic
    type_labels = {
        'topics': 'الموضوع',
        'entities': 'الكيان',
        'words': 'الكلمة'
    }
    
    type_label = type_labels.get(data_type, 'العنصر')
    
    # Build the prompt with full context
    prompt = f"""أنت محلل بيانات خبير متخصص في تحليل استبيانات رضا العملاء باللغة العربية.

    مهمتك: تحليل بيانات الاستبيان المرفقة واستخراج الرؤى الأساسية بطريقة واضحة ومفيدة.

    **معلومات الاستبيان:**
    - عنوان الاستبيان: "{survey_title}"
    - السؤال المطروح: "{survey_question}"

    ## خطوات التحليل:

    1. **فهم السياق:**
    - اقرأ عنوان الاستبيان والسؤال المطروح
    - افهم طبيعة البيانات والمواضيع المذكورة

    2. **تحديد المواضيع الرئيسية:**
    - صنف المواضيع حسب عدد التكرارات (من الأكثر إلى الأقل)
    - اجمع المواضيع المتشابهة معاً
    - اذكر عدد التكرارات لكل موضوع

    3. **تحليل تفاصيل كل موضوع:**
    - استخرج النقاط الفرعية لكل موضوع رئيسية
    - اذكر أمثلة محددة من الإجابات
    - سلط الضوء على أي أسماء أو حالات خاصة مذكورة

    4. **استخلاص الأنماط:**
    - ابحث عن أنماط متكررة في المواضيع
    - حدد أي تناقضات أو حالات استثنائية (مثل: تمييز، محاباة، إلخ)

    5. **تقديم الانطباع العام:**
    - لخص المزاج العام للعملاء/التجار
    - حدد مستوى الرضا/الاستياء
    - اذكر أهم نقاط الضعف التي تحتاج معالجة فورية

    ## صيغة العرض المطلوبة:

    استخدم العناوين والتنسيق التالي:

    ```
    ## المواضيع الرئيسية:

    **1. [اسم الموضوع] (عدد التكرارات)**
    - نقطة فرعية 1
    - نقطة فرعية 2
    - أي تفاصيل مهمة أو أمثلة محددة

    **2. [الموضوع الثانية]**
    ...

    ## الانطباع العام:
    [تلخيص واضح ومباشر للوضع]
    ```

    ## ملاحظات مهمة:
    - استخدم لغة عربية واضحة ومباشرة
    - ركز على الحقائق والأرقام
    - اذكر أي أسماء أو تفاصيل محددة تم ذكرها
    - كن موضوعياً ولا تتجاهل أي موضوع مهما كانت صغيرة
    - اجعل التحليل عملياً وقابلاً للتنفيذ

    ---

    **البيانات المطلوب تحليلها:**

    """
    
    # Add each item with ALL its answers
    for item, data in data_dict.items():
        prompt += f"\n{'='*80}\n"
        prompt += f"**{type_label}: {item}**\n"
        prompt += f"**عدد التكرارات: {data['count']}**\n\n"
        prompt += f"**جميع الإجابات المتعلقة بهذا {type_label}:**\n\n"
        
        for i, answer in enumerate(data['answers'], 1):
            prompt += f"{i}. {answer}\n"
    
    prompt += f"\n{'='*80}\n"
    
    # Call LLM
    print(f"\n🤖 إرسال بيانات {type_label} إلى LLM للتحليل...")
    print(f"📊 عدد العناصر: {len(data_dict)}")
    total_answers = sum(data['count'] for data in data_dict.values())
    print(f"📝 إجمالي عدد الإجابات: {total_answers}")
    
    # Save the full prompt to a file
    prompt_file = f'exports/llm_prompt_{data_type}_input.txt'
    try:
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f"💾 تم حفظ البرومبت الكامل في: {prompt_file}")
    except Exception as e:
        print(f"⚠️ خطأ في حفظ البرومبت: {str(e)}")
    
    # Print the prompt to console
    print("\n" + "="*80)
    print(f"📋 البيانات المرسلة إلى LLM ({type_label}):")
    print("="*80)
    print(prompt)
    print("="*80 + "\n")
    
    try:
        response = model.invoke(prompt)
        return response.content
    except Exception as e:
        return f"❌ خطأ في الاتصال بـ LLM: {str(e)}"

def get_top_negative_entities(df, top_n=5):
    """ما أكثر الكيانات (منتج / خدمة / موظف) المرتبطة بالسلبية؟"""
    if 'sentiment' not in df.columns or 'entities' not in df.columns:
        return "Required columns not found"
    
    # Filter by negative sentiment
    negative_df = df[df['sentiment'].str.contains('negative', case=False, na=False)]
    
    entity_counts = []
    
    for _, row in negative_df.iterrows():
        try:
            # entities are stored as strings of lists in CSV
            entities = ast.literal_eval(row['entities'])
            if isinstance(entities, list):
                for ent in entities:
                    # Filter for specific types: Person (موظف), Product (منتج), Organization (خدمة)
                    if ent['label'] in ['شخص', 'منتج', 'منظمة']:
                        entity_counts.append(f"{ent['text']} ({ent['label']})")
        except:
            continue
            
    return pd.Series(entity_counts).value_counts().head(top_n)

def get_top_words_in_text(df, sentiment_label='negative', top_n=5):
    """9. ما الكلمات الأكثر تكراراً في التعليقات (سلبية مثلاً)؟"""
    if 'sentiment' not in df.columns or 'Answer_normalized' not in df.columns:
        return "Required columns not found"
    
    # Filter by sentiment
    filtered_df = df[df['sentiment'].str.contains(sentiment_label, case=False, na=False)]
    
    # Combine all texts
    texts = filtered_df['Answer_normalized'].dropna().astype(str).tolist()
    
    # Basic Arabic Stopwords
    stopwords = set([
        'في', 'من', 'على', 'إلى', 'عن', 'مع', 'ما', 'يا', 'ب', 'ل', 'و', 'ك', 
        'لا', 'نعم', 'كان', 'يكون', 'هذا', 'هذه', 'ذلك', 'الي', 'اللي', 'ان',
        'او', 'هل', 'كل', 'بعد', 'قبل', 'عند', 'حتى', 'بس', 'تم', 'تمت',
        'عدم', 'عدم وجود', 'التي', 'الذي', 'اذا', 'إذا', 'لو', 'غير', 'بدون',
        'ممكن', 'يوجد', 'ليس', 'لم', 'لن', 'أو', 'أن', 'كنت', 'كانت', 'يعني'
    ])
    
    words = []
    for text in texts:
        # Clean text: keep only Arabic letters
        clean_text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
        for word in clean_text.split():
            if len(word) > 2 and word not in stopwords:
                words.append(word)
    
    top_words = pd.Series(words).value_counts().head(top_n)
    return top_words


# ============================================================================
# NEW FUNCTIONS WITH CONTEXT
# ============================================================================

def get_topics_with_context(df, sentiment_label=None, top_n=5):
    """
    Extract topics with full answer context.
    Returns a DataFrame with topic, count, and sample answers.
    """
    if 'topic_label' not in df.columns:
        return "Topic column not found"
    
    # Filter by sentiment if specified
    if sentiment_label:
        filtered_df = df[df['sentiment'].str.contains(sentiment_label, case=False, na=False)].copy()
    else:
        filtered_df = df.copy()
    
    # Build topic -> answers mapping
    topic_to_answers = {}
    
    for idx, row in filtered_df.iterrows():
        topic = str(row.get('topic_label', ''))
        answer = str(row.get('Answer_normalized', ''))
        
        # Split topics if comma-separated
        parts = [p.strip() for p in topic.split(',')]
        for t in parts:
            if t not in ['متنوع', 'غير محدد', 'خطأ', '', 'nan']:
                if t not in topic_to_answers:
                    topic_to_answers[t] = []
                topic_to_answers[t].append(answer)
    
    # Count and sort
    topic_counts = {topic: len(answers) for topic, answers in topic_to_answers.items()}
    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Build result DataFrame
    result_data = []
    for topic, count in sorted_topics:
        answers = topic_to_answers[topic]
        # Take up to 5 sample answers
        sample_answers = '\n\n---\n\n'.join(answers[:5])
        result_data.append({
            'الموضوع': topic,
            'التكرار': count,
            'أمثلة من الإجابات': sample_answers
        })
    
    return pd.DataFrame(result_data)

def get_entities_with_context(df, sentiment_label='negative', top_n=5):
    """
    Extract entities (products/employees/services) with full answer context.
    Returns a DataFrame with entity, type, count, and sample answers.
    """
    if 'sentiment' not in df.columns or 'entities' not in df.columns:
        return "Required columns not found"
    
    # Filter by sentiment
    filtered_df = df[df['sentiment'].str.contains(sentiment_label, case=False, na=False)].copy()
    
    # Build entity -> answers mapping
    entity_to_answers = {}
    
    for idx, row in filtered_df.iterrows():
        answer = str(row.get('Answer_normalized', ''))
        try:
            entities = ast.literal_eval(row['entities'])
            if isinstance(entities, list):
                for ent in entities:
                    if ent['label'] in ['شخص', 'منتج', 'منظمة']:
                        entity_key = f"{ent['text']} ({ent['label']})"
                        if entity_key not in entity_to_answers:
                            entity_to_answers[entity_key] = []
                        entity_to_answers[entity_key].append(answer)
        except:
            continue
    
    # Count and sort
    entity_counts = {entity: len(answers) for entity, answers in entity_to_answers.items()}
    sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Build result DataFrame
    result_data = []
    for entity, count in sorted_entities:
        answers = entity_to_answers[entity]
        # Take up to 5 sample answers
        sample_answers = '\n\n---\n\n'.join(answers[:5])
        result_data.append({
            'الكيان': entity,
            'التكرار': count,
            'أمثلة من الإجابات': sample_answers
        })
    
    return pd.DataFrame(result_data)

def get_words_with_context(df, sentiment_label='negative', top_n=5):
    """
    Extract top words with full answer context.
    Returns a DataFrame with word, count, and sample answers.
    """
    if 'sentiment' not in df.columns or 'Answer_normalized' not in df.columns:
        return "Required columns not found"
    
    # Filter by sentiment
    filtered_df = df[df['sentiment'].str.contains(sentiment_label, case=False, na=False)].copy()
    
    # Basic Arabic Stopwords
    stopwords = set([
        'في', 'من', 'على', 'إلى', 'عن', 'مع', 'ما', 'يا', 'ب', 'ل', 'و', 'ك', 
        'لا', 'نعم', 'كان', 'يكون', 'هذا', 'هذه', 'ذلك', 'الي', 'اللي', 'ان',
        'او', 'هل', 'كل', 'بعد', 'قبل', 'عند', 'حتى', 'بس', 'تم', 'تمت',
        'عدم', 'عدم وجود', 'التي', 'الذي', 'اذا', 'إذا', 'لو', 'غير', 'بدون',
        'ممكن', 'يوجد', 'ليس', 'لم', 'لن', 'أو', 'أن', 'كنت', 'كانت', 'يعني'
    ])
    
    # Build word -> answers mapping
    word_to_answers = {}
    
    for idx, row in filtered_df.iterrows():
        answer = str(row.get('Answer_normalized', ''))
        # Clean text: keep only Arabic letters
        clean_text = re.sub(r'[^\u0600-\u06FF\s]', '', answer)
        
        for word in clean_text.split():
            if len(word) > 2 and word not in stopwords:
                if word not in word_to_answers:
                    word_to_answers[word] = []
                word_to_answers[word].append(answer)
    
    # Count and sort
    word_counts = {word: len(answers) for word, answers in word_to_answers.items()}
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Build result DataFrame
    result_data = []
    for word, count in sorted_words:
        answers = word_to_answers[word]
        # Take up to 5 sample answers
        sample_answers = '\n\n---\n\n'.join(answers[:5])
        result_data.append({
            'الكلمة': word,
            'التكرار': count,
            'أمثلة من الإجابات': sample_answers
        })
    
    return pd.DataFrame(result_data)

# ============================================================================
# EXECUTION / TESTING
# ============================================================================

if __name__ == "__main__":
    # Path to the latest export
    input_file = r'exports/survey_data_enriched_S25120024_20260129_174603.csv'
    
    # Survey metadata
    survey_title = "تقييم تجربة التاجر مع مندوب المبيعات - موبايل شوب"
    survey_question = "رأيك يهمنا، فضلاً شاركنا تعليق او مقترح يساعد في تحسين التجربة"

    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found!")
    else:
        print(f"📖 Reading data from {input_file}...")
        df = pd.read_csv(input_file)

        # ============================================================================
        # 1. تحليل المواضيع السلبية
        # ============================================================================
        print("\n" + "=" * 80)
        print("1️⃣  تحليل المواضيع السلبية باستخدام الذكاء الصناعي")
        print("=" * 80)
        
        print("\n📊 استخراج المواضيع السلبية مع جميع التكرارات...")
        negative_topics_data = get_topics_with_full_context_for_llm(
            df, 
            sentiment_label='negative', 
            top_n=5
        )
        
        if isinstance(negative_topics_data, str):
            print(f"❌ {negative_topics_data}")
        else:
            print(f"\n✅ تم استخراج {len(negative_topics_data)} موضوع سلبي")
            print("\nملخص المواضيع:")
            for topic, data in negative_topics_data.items():
                print(f"  • {topic}: {data['count']} تكرار")
            
            # Send to LLM for analysis
            print("\n" + "=" * 80)
            topics_analysis = analyze_topics_with_llm(
                negative_topics_data,
                survey_title,
                survey_question
            )
            
            print("\n" + "=" * 80)
            print("📝 تحليل الذكاء الصناعي للمواضيع:")
            print("=" * 80)
            print(topics_analysis)
            print("\n" + "=" * 80)
            
            # Save the analysis
            output_file = r'exports/llm_analysis_negative_topics.txt'
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write(f"تحليل الذكاء الصناعي للمواضيع السلبية\n")
                    f.write(f"الاستبيان: {survey_title}\n")
                    f.write(f"السؤال: {survey_question}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(topics_analysis)
                print(f"\n💾 تم حفظ تحليل المواضيع في: {output_file}")
            except Exception as e:
                print(f"⚠️ خطأ في حفظ الملف: {str(e)}")

        # ============================================================================
        # 2. تحليل الكيانات السلبية
        # ============================================================================
        print("\n" + "=" * 80)
        print("2️⃣  تحليل الكيانات السلبية باستخدام الذكاء الصناعي")
        print("=" * 80)
        
        print("\n📊 استخراج الكيانات السلبية مع جميع التكرارات...")
        negative_entities_data = get_entities_with_full_context_for_llm(
            df,
            sentiment_label='negative',
            top_n=5
        )
        
        if isinstance(negative_entities_data, str):
            print(f"❌ {negative_entities_data}")
        else:
            print(f"\n✅ تم استخراج {len(negative_entities_data)} كيان سلبي")
            print("\nملخص الكيانات:")
            for entity, data in negative_entities_data.items():
                print(f"  • {entity}: {data['count']} تكرار")
            
            # Send to LLM for analysis
            print("\n" + "=" * 80)
            entities_analysis = analyze_data_with_llm(
                negative_entities_data,
                data_type='entities',
                survey_title=survey_title,
                survey_question=survey_question
            )
            
            print("\n" + "=" * 80)
            print("📝 تحليل الذكاء الصناعي للكيانات:")
            print("=" * 80)
            print(entities_analysis)
            print("\n" + "=" * 80)
            
            # Save the analysis
            output_file = r'exports/llm_analysis_negative_entities.txt'
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write(f"تحليل الذكاء الصناعي للكيانات السلبية\n")
                    f.write(f"الاستبيان: {survey_title}\n")
                    f.write(f"السؤال: {survey_question}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(entities_analysis)
                print(f"\n💾 تم حفظ تحليل الكيانات في: {output_file}")
            except Exception as e:
                print(f"⚠️ خطأ في حفظ الملف: {str(e)}")

        # ============================================================================
        # 3. تحليل الكلمات السلبية
        # ============================================================================
        print("\n" + "=" * 80)
        print("3️⃣  تحليل الكلمات السلبية باستخدام الذكاء الصناعي")
        print("=" * 80)
        
        print("\n📊 استخراج الكلمات السلبية مع جميع التكرارات...")
        negative_words_data = get_words_with_full_context_for_llm(
            df,
            sentiment_label='negative',
            top_n=5
        )
        
        if isinstance(negative_words_data, str):
            print(f"❌ {negative_words_data}")
        else:
            print(f"\n✅ تم استخراج {len(negative_words_data)} كلمة سلبية")
            print("\nملخص الكلمات:")
            for word, data in negative_words_data.items():
                print(f"  • {word}: {data['count']} تكرار")
            
            # Send to LLM for analysis
            print("\n" + "=" * 80)
            words_analysis = analyze_data_with_llm(
                negative_words_data,
                data_type='words',
                survey_title=survey_title,
                survey_question=survey_question
            )
            
            print("\n" + "=" * 80)
            print("📝 تحليل الذكاء الصناعي للكلمات:")
            print("=" * 80)
            print(words_analysis)
            print("\n" + "=" * 80)
            
            # Save the analysis
            output_file = r'exports/llm_analysis_negative_words.txt'
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write(f"تحليل الذكاء الصناعي للكلمات السلبية\n")
                    f.write(f"الاستبيان: {survey_title}\n")
                    f.write(f"السؤال: {survey_question}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(words_analysis)
                print(f"\n💾 تم حفظ تحليل الكلمات في: {output_file}")
            except Exception as e:
                print(f"⚠️ خطأ في حفظ الملف: {str(e)}")
        
        # ============================================================================
        # ملخص نهائي
        # ============================================================================
        print("\n" + "=" * 80)
        print("✅ اكتمل التحليل الشامل بنجاح!")
        print("=" * 80)
        print("\nالملفات المُنتجة:")
        print("  1. exports/llm_analysis_negative_topics.txt")
        print("  2. exports/llm_analysis_negative_entities.txt")
        print("  3. exports/llm_analysis_negative_words.txt")
        print("  4. exports/llm_prompt_topics_input.txt")
        print("  5. exports/llm_prompt_entities_input.txt")
        print("  6. exports/llm_prompt_words_input.txt")
        print("=" * 80 + "\n")
