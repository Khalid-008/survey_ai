import pandas as pd
import os
import re
import ast
from collections import Counter

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

def get_top_topics_by_sentiment(df, sentiment_label='negative', top_n=10):
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

def get_top_negative_entities(df, top_n=50):
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

def get_top_words_in_text(df, sentiment_label='negative', top_n=10):
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

def get_topics_with_context(df, sentiment_label=None, top_n=10):
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

def get_entities_with_context(df, sentiment_label='negative', top_n=20):
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

def get_words_with_context(df, sentiment_label='negative', top_n=20):
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
    output_file = r'exports/survey_summary_analysis_with_context.xlsx'

    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found!")
    else:
        print(f"📖 Reading data from {input_file}...")
        df = pd.read_csv(input_file)

        print("-" * 50)
        
        # 1. Sentiment Distribution
        print("\n1️⃣  توزيع المشاعر:")
        sent_dist = get_sentiment_distribution(df)
        print(sent_dist)

        # 2. Top 10 Topics (basic count)
        print("\n2️⃣  أكثر 10 مواضيع تكراراً:")
        top_10_topics = get_top_topics(df, 10)
        print(top_10_topics)
        
        # 2b. Top 10 Topics with Context
        print("\n2️⃣b  أكثر 10 مواضيع مع السياق الكامل:")
        top_10_topics_context = get_topics_with_context(df, sentiment_label=None, top_n=10)
        print(f"Found {len(top_10_topics_context)} topics with context")

        # 3. Top 5 Negative Topics (basic count)
        print("\n3️⃣  أكثر 5 مواضيع مرتبطة بمشاعر سلبية:")
        top_5_neg_topics = get_top_topics_by_sentiment(df, 'negative', 5)
        print(top_5_neg_topics)
        
        # 3b. Top Negative Topics with Context
        print("\n3️⃣b  مواضيع سلبية مع السياق الكامل:")
        neg_topics_context = get_topics_with_context(df, sentiment_label='negative', top_n=10)
        print(f"Found {len(neg_topics_context)} negative topics with context")

        # 4. Top 5 Positive Topics
        print("\n7️⃣  أكثر 5 مواضيع مرتبطة بمشاعر إيجابية:")
        top_5_pos_topics = get_top_topics_by_sentiment(df, 'positive', 5)
        print(top_5_pos_topics)
        
        # 4b. Top Positive Topics with Context
        print("\n7️⃣b  مواضيع إيجابية مع السياق الكامل:")
        pos_topics_context = get_topics_with_context(df, sentiment_label='positive', top_n=10)
        print(f"Found {len(pos_topics_context)} positive topics with context")
        
        # 5. Top Negative Entities (basic count)
        print("\n🔍 أكثر الكيانات المرتبطة بالسلبية:")
        top_neg_entities = get_top_negative_entities(df, 10)
        print(top_neg_entities)
        
        # 5b. Negative Entities with Context
        print("\n🔍b  كيانات سلبية مع السياق الكامل:")
        neg_entities_context = get_entities_with_context(df, sentiment_label='negative', top_n=20)
        print(f"Found {len(neg_entities_context)} entities with context")

        # 6. Top 10 words in negative comments (basic count)
        print("\n9️⃣  الكلمات الأكثر تكراراً في التعليقات السلبية:")
        top_neg_words = get_top_words_in_text(df, 'negative', 10)
        print(top_neg_words)
        
        # 6b. Negative Words with Context
        print("\n9️⃣b  كلمات سلبية مع السياق الكامل:")
        neg_words_context = get_words_with_context(df, sentiment_label='negative', top_n=20)
        print(f"Found {len(neg_words_context)} words with context")

        # Save to Excel
        print(f"\n💾 Saving detailed results to {output_file}...")
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Basic summaries
                sent_dist.to_excel(writer, sheet_name='توزيع المشاعر')
                top_10_topics.to_excel(writer, sheet_name='أهم 10 مواضيع', header=['التكرار'])
                top_5_neg_topics.to_excel(writer, sheet_name='أكثر 5 مشاكل', header=['التكرار'])
                top_5_pos_topics.to_excel(writer, sheet_name='نقاط إيجابية', header=['التكرار'])
                top_neg_entities.to_excel(writer, sheet_name='كيانات سلبية', header=['التكرار'])
                top_neg_words.to_excel(writer, sheet_name='كلمات سلبية', header=['التكرار'])
                
                # Context-rich sheets
                top_10_topics_context.to_excel(writer, sheet_name='مواضيع عامة + سياق', index=False)
                neg_topics_context.to_excel(writer, sheet_name='مواضيع سلبية + سياق', index=False)
                pos_topics_context.to_excel(writer, sheet_name='مواضيع إيجابية + سياق', index=False)
                neg_entities_context.to_excel(writer, sheet_name='كيانات سلبية + سياق', index=False)
                neg_words_context.to_excel(writer, sheet_name='كلمات سلبية + سياق', index=False)
                
                # Auto-adjust column widths for context sheets
                for sheet_name in ['مواضيع عامة + سياق', 'مواضيع سلبية + سياق', 
                                   'مواضيع إيجابية + سياق', 'كيانات سلبية + سياق', 
                                   'كلمات سلبية + سياق']:
                    worksheet = writer.sheets[sheet_name]
                    worksheet.column_dimensions['A'].width = 30
                    worksheet.column_dimensions['B'].width = 15
                    worksheet.column_dimensions['C'].width = 100
                
            print("✅ Analysis completed successfully!")
            print(f"✅ Results saved to: {output_file}")
        except Exception as e:
            print(f"⚠️ Error saving Excel file: {str(e)}")