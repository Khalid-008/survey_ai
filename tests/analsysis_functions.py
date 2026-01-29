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
# EXECUTION / TESTING
# ============================================================================

if __name__ == "__main__":
    # Path to the latest export
    input_file = r'exports/survey_data_enriched_S25120024_20260129_174603.csv'
    output_file = r'exports/survey_summary_analysis.xlsx'

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

        # 2. Top 10 Topics
        print("\n2️⃣  أكثر 10 مواضيع تكراراً:")
        top_10_topics = get_top_topics(df, 10)
        print(top_10_topics)

        # 3. Top 5 Negative Topics
        print("\n3️⃣  أكثر 5 مواضيع مرتبطة بمشاعر سلبية:")
        top_5_neg_topics = get_top_topics_by_sentiment(df, 'negative', 5)
        print(top_5_neg_topics)

        # 4. Top 5 Positive Topics
        print("\n7️⃣  أكثر 5 مواضيع مرتبطة بمشاعر إيجابية:")
        top_5_pos_topics = get_top_topics_by_sentiment(df, 'positive', 5)
        print(top_5_pos_topics)
        
        # 5. Top Negative Entities (Products/Employees)
        print("\n🔍 أكثر الكيانات المرتبطة بالسلبية (منتجات / موظفين / خدمات):")
        top_neg_entities = get_top_negative_entities(df, 10)
        print(top_neg_entities)

        # 6. Top 10 words in negative comments
        print("\n9️⃣  الكلمات الأكثر تكراراً في التعليقات السلبية:")
        top_neg_words = get_top_words_in_text(df, 'negative', 10)
        print(top_neg_words)

        # Save to Excel
        print(f"\n💾 Saving detailed results to {output_file}...")
        try:
            with pd.ExcelWriter(output_file) as writer:
                sent_dist.to_excel(writer, sheet_name='توزيع المشاعر')
                top_10_topics.to_excel(writer, sheet_name='أهم 10 مواضيع', header=['التكرار'])
                top_5_neg_topics.to_excel(writer, sheet_name='أكثر 5 مشاكل', header=['التكرار'])
                top_5_pos_topics.to_excel(writer, sheet_name='أثر النقاط الإيجابية', header=['التكرار'])
                top_neg_entities.to_excel(writer, sheet_name='كيانات سلبية (منتجات-موظفين)', header=['التكرار'])
                top_neg_words.to_excel(writer, sheet_name='كلمات سلبية متكررة', header=['التكرار'])
            print("✅ Analysis completed successfully!")
        except Exception as e:
            print(f"⚠️ Error saving Excel file: {str(e)}")
