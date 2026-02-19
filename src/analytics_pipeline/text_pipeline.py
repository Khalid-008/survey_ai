import os
import re
import sys
import ast
import warnings
from collections import Counter

import pandas as pd
import numpy as np
from transformers import pipeline as hf_pipeline
from gliner import GLiNER
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from tqdm import tqdm

# إعداد المسار للاستيراد
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llms.models import model

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION & CONSTANTS
# إعدادات وثوابت المشروع
# ============================================================================

# أسماء النماذج المستخدمة
SENTIMENT_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment"
NER_MODEL       = "NAMAA-Space/gliner_arabic-v2.1"
TOPIC_MODEL     = "CAMeL-Lab/bert-base-arabic-camelbert-da"

# إعدادات المعالجة
SENTIMENT_BATCH_SIZE = 32
NER_BATCH_SIZE       = 16
CHUNK_SIZE           = 5000  # معالجة البيانات الكبيرة على دفعات لتجنب مشاكل الذاكرة

# إعدادات التحليل
MIN_TOPIC_SIZE  = 15
TOP_N_ITEMS     = 15
TOP_N_WORDS     = 15
SENTIMENT_FILTER = 'negative'

# تصنيفات الكيانات المستخدمة في NER
NER_LABELS = ["شخص", "مكان", "منظمة", "تاريخ", "وقت", "منتج", "حدث"]

# أنواع الكيانات المهمة للتحليل
RELEVANT_ENTITY_TYPES = ['شخص', 'منتج', 'منظمة', 'مكان']

# الكلمات الشائعة التي تُحذف من التحليل
ARABIC_STOPWORDS = {
    'في', 'من', 'إلى', 'على', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'التي',
    'الذي', 'أن', 'أو', 'كان', 'كانت', 'لكن', 'ما', 'لا', 'نعم',
    'قد', 'لم', 'لن', 'إن', 'كل', 'بعض', 'غير', 'بين', 'عند',
    'منذ', 'حتى', 'ثم', 'أم', 'إما', 'بل', 'حيث', 'كيف', 'أين',
    'متى', 'لماذا', 'ماذا', 'هل', 'له', 'لها', 'لهم', 'لنا', 'هو',
    'هي', 'هم', 'نحن', 'أنت', 'أنتم', 'انا', 'أنا', 'و', 'أو', 'ف', 'ب',
    'ك', 'ل', 'ال', 'الـ', 'ـ', 'ة', 'ه', 'ها', 'هن', 'هما', 'همـ',
    'يا', 'يكون', 'الي', 'اللي', 'ان', 'او', 'بعد', 'قبل', 'بس', 'تم',
    'تمت', 'عدم', 'اذا', 'إذا', 'لو', 'بدون', 'ممكن', 'يوجد', 'ليس',
    'كنت', 'يعني', 'عدم وجود'
}

# قيم افتراضية للبيانات الناقصة
DEFAULT_TOPIC_LABEL   = "متنوع"
UNDEFINED_TOPIC_LABEL = "غير محدد"
ERROR_TOPIC_LABEL     = "خطأ"
OUTLIER_TOPIC_ID      = -1


# ============================================================================
# GLOBAL MODEL CACHE
# تخزين النماذج في الذاكرة لتجنب إعادة تحميلها في كل مرة
# ============================================================================

# هذه المتغيرات تحفظ النماذج بعد أول تحميل
_cached_sentiment_pipeline = None
_cached_ner_model          = None
_cached_topic_extractor    = None


def get_sentiment_pipeline():
    """تحميل نموذج تحليل المشاعر أو إرجاعه من الذاكرة إن كان محملاً مسبقاً."""
    global _cached_sentiment_pipeline

    if _cached_sentiment_pipeline is None:
        print(f"🔄 Loading sentiment model: {SENTIMENT_MODEL}")
        _cached_sentiment_pipeline = hf_pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL,
            device="cpu"
        )
        print("✅ Sentiment model loaded")

    return _cached_sentiment_pipeline


def get_ner_model():
    """تحميل نموذج NER أو إرجاعه من الذاكرة إن كان محملاً مسبقاً."""
    global _cached_ner_model

    if _cached_ner_model is None:
        print(f"🔄 Loading NER model: {NER_MODEL}")
        _cached_ner_model = GLiNER.from_pretrained(NER_MODEL)
        print("✅ NER model loaded")

    return _cached_ner_model


def get_topic_extractor():
    """إنشاء أداة استخراج المواضيع أو إرجاعها من الذاكرة إن كانت موجودة مسبقاً."""
    global _cached_topic_extractor

    if _cached_topic_extractor is None:
        print("🔄 Initializing topic extractor...")
        _cached_topic_extractor = create_topic_extractor(min_topic_size=MIN_TOPIC_SIZE)
        print("✅ Topic extractor ready")

    return _cached_topic_extractor


# ============================================================================
# TEXT NORMALIZATION
# تطبيع وتنظيف النص العربي
# ============================================================================

def normalize_arabic(text):
    """تطبيع النص العربي بتوحيد الأحرف وإزالة الرموز غير المرغوبة."""
    if text is None:
        return ""

    text = str(text)

    # توحيد الحروف العربية المتشابهة
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)

    # الإبقاء فقط على الأحرف العربية والأرقام والمسافات والشرطات والنقطتين
    text = re.sub("[^\u0600-\u06FF\s0-9\-:]", "", text)

    # توحيد المسافات
    text = re.sub(r"\s+", " ", text).strip()

    return text


def convert_arabic_time_to_24h(text):
    """تحويل صيغ الوقت العربية (م/ص) إلى صيغة 24 ساعة."""
    if not isinstance(text, str):
        return text

    # نمط: رقم + م أو ص + فاصل + رقم + م أو ص
    pattern = r'(\d+)\s*([مص]?)\s*(?:الى|-|—)\s*(\d+)\s*([مص])'

    def replace_match(match):
        h1_str, p1, h2_str, p2 = match.groups()
        h1 = int(h1_str)
        h2 = int(h2_str)

        # إذا لم يُذكر وقت الساعة الأولى نفترض أنه مثل الثانية
        if not p1:
            p1 = p2

        def to_24h(hour, period):
            if period == 'م':  # مساء PM
                return hour if hour == 12 else hour + 12
            if period == 'ص':  # صباح AM
                return 0 if hour == 12 else hour
            return hour

        h1_24 = to_24h(h1, p1)
        h2_24 = to_24h(h2, p2)

        return f"{h1_24:02d}:00-{h2_24:02d}:00"

    return re.sub(pattern, replace_match, text)


def remove_stopwords(text):
    """إزالة الكلمات الشائعة من النص وإرجاع قائمة الكلمات المتبقية."""
    # إبقاء الأحرف العربية فقط
    clean_text = re.sub(r'[^\u0600-\u06FF\s]', '', text)

    words = []
    for word in clean_text.split():
        if len(word) > 2 and word not in ARABIC_STOPWORDS:
            words.append(word)

    return words


# ============================================================================
# SENTIMENT ANALYSIS
# تحليل المشاعر
# ============================================================================

def analyze_sentiment_batch(texts, default_sentiment="neutral"):
    """
    تحليل المشاعر لقائمة من النصوص.

    texts             : قائمة النصوص المراد تحليلها
    default_sentiment : القيمة الافتراضية للنصوص الفارغة أو غير الصالحة
    يُرجع             : قائمة بتصنيف المشاعر لكل نص
    """
    print(f"📊 Analyzing sentiment for {len(texts)} texts...")

    # تصفية النصوص الصالحة مع تتبع مواضعها الأصلية
    valid_indices = []
    for i, t in enumerate(texts):
        if t and str(t).strip() and str(t).lower() != "no answer":
            valid_indices.append(i)

    valid_texts = [str(texts[i]) for i in valid_indices]

    # تهيئة النتائج بالقيمة الافتراضية
    sentiments = [default_sentiment] * len(texts)

    if not valid_texts:
        print("⚠️ No valid texts to analyze")
        return sentiments

    print(f"   Valid texts: {len(valid_texts)}/{len(texts)}")

    try:
        sentiment_pipeline = get_sentiment_pipeline()
        all_results = []

        # معالجة النصوص على دفعات لتجنب مشاكل الذاكرة
        with tqdm(total=len(valid_texts), desc="   Sentiment Analysis", unit="text") as pbar:
            for i in range(0, len(valid_texts), CHUNK_SIZE):
                chunk = valid_texts[i:i + CHUNK_SIZE]
                chunk_results = sentiment_pipeline(chunk, batch_size=SENTIMENT_BATCH_SIZE)
                all_results.extend(chunk_results)
                pbar.update(len(chunk))

        # إعادة النتائج لمواضعها الأصلية
        for idx, result in zip(valid_indices, all_results):
            sentiments[idx] = result["label"]

        print("✅ Sentiment analysis complete")

    except Exception as e:
        print(f"⚠️ Error in sentiment analysis: {str(e)}")
        import traceback
        traceback.print_exc()

    return sentiments


# ============================================================================
# NAMED ENTITY RECOGNITION
# استخراج الكيانات المسماة
# ============================================================================

def convert_entities_to_simple_format(entities):
    """
    تحويل نتائج GLiNER إلى تنسيق بايثون بسيط قابل للحفظ كـ JSON.

    entities : قائمة الكيانات من GLiNER
    يُرجع   : قائمة قواميس بتنسيق بسيط
    """
    if not entities:
        return []

    simple_entities = []
    for entity in entities:
        simple_entity = {
            'text':  str(entity.get('text', '')),
            'label': str(entity.get('label', '')),
            'score': float(entity.get('score', 0.0)),
            'start': int(entity.get('start', 0)),
            'end':   int(entity.get('end', 0))
        }
        simple_entities.append(simple_entity)

    return simple_entities


def extract_entities_batch(texts, labels=None, threshold=0.3):
    """
    استخراج الكيانات المسماة من قائمة نصوص.

    texts     : قائمة النصوص
    labels    : أنواع الكيانات المراد استخراجها
    threshold : الحد الأدنى لدرجة الثقة
    يُرجع    : قائمة من قوائم الكيانات لكل نص
    """
    if labels is None:
        labels = NER_LABELS

    print(f"🏷️  Extracting entities from {len(texts)} texts...")

    # تصفية النصوص الصالحة
    valid_indices = []
    for i, t in enumerate(texts):
        if t and str(t).strip() and str(t).lower() != "no answer":
            valid_indices.append(i)

    valid_texts = [str(texts[i]) for i in valid_indices]

    # تهيئة النتائج الفارغة
    all_results = [[] for _ in range(len(texts))]

    if not valid_texts:
        print("⚠️ No valid texts to process")
        return all_results

    print(f"   Valid texts: {len(valid_texts)}/{len(texts)}")

    try:
        ner_model = get_ner_model()
        all_batched_entities = []

        with tqdm(total=len(valid_texts), desc="   NER Extraction", unit="text") as pbar:
            for i in range(0, len(valid_texts), CHUNK_SIZE):
                chunk = valid_texts[i:i + CHUNK_SIZE]
                chunk_entities = ner_model.batch_predict_entities(
                    chunk,
                    labels,
                    threshold=threshold,
                    batch_size=NER_BATCH_SIZE
                )
                all_batched_entities.extend(chunk_entities)
                pbar.update(len(chunk))

        # إعادة النتائج لمواضعها الأصلية
        for idx, entities in zip(valid_indices, all_batched_entities):
            all_results[idx] = convert_entities_to_simple_format(entities)

        print("✅ Entity extraction complete")

    except Exception as e:
        print(f"⚠️ Error in entity extraction: {str(e)}")
        import traceback
        traceback.print_exc()

    return all_results


# ============================================================================
# TOPIC EXTRACTION
# استخراج المواضيع
# ============================================================================

def create_topic_extractor(min_topic_size=10):
    """
    إنشاء نموذج BERTopic لاستخراج المواضيع من النصوص العربية.

    min_topic_size : الحد الأدنى لعدد الوثائق في الموضوع الواحد
    يُرجع         : قاموس يحتوي على النموذج ومكوناته
    """
    print(f"⏳ Loading embedding model: {TOPIC_MODEL}")
    embedding_model = SentenceTransformer(TOPIC_MODEL)

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=list(ARABIC_STOPWORDS),
        max_features=5000,
        min_df=2
    )

    ctfidf_model = ClassTfidfTransformer()

    return {
        'embedding_model':  embedding_model,
        'umap_model':       umap_model,
        'hdbscan_model':    hdbscan_model,
        'vectorizer_model': vectorizer_model,
        'ctfidf_model':     ctfidf_model,
        'topic_model':      None,   # يُنشأ عند التدريب
        'min_topic_size':   min_topic_size
    }


def train_topic_model(extractor, documents):
    """
    تدريب نموذج المواضيع على قائمة الوثائق.

    extractor : القاموس الذي أنشأه create_topic_extractor
    documents : قائمة النصوص للتدريب
    يُرجع    : نفس القاموس مع النموذج المدرّب
    """
    print(f"⏳ Training topic model on {len(documents)} documents...")
    print("   This may take several minutes for large datasets...")

    topic_model = BERTopic(
        embedding_model=extractor['embedding_model'],
        umap_model=extractor['umap_model'],
        hdbscan_model=extractor['hdbscan_model'],
        vectorizer_model=extractor['vectorizer_model'],
        ctfidf_model=extractor['ctfidf_model'],
        language="arabic",
        calculate_probabilities=True,
        verbose=True
    )

    try:
        topic_model.fit_transform(documents)
        extractor['topic_model'] = topic_model
        print("✅ Topic model trained successfully")
    except Exception as e:
        print(f"⚠️ Error training topic model: {str(e)}")
        import traceback
        traceback.print_exc()
        extractor['topic_model'] = None

    return extractor


def get_topics_from_documents(extractor, documents, top_n_words=10):
    """
    استخراج المواضيع من قائمة الوثائق.

    extractor   : القاموس الذي أنشأه create_topic_extractor
    documents   : قائمة النصوص
    top_n_words : عدد الكلمات المعبّرة عن كل موضوع
    يُرجع      : قاموس بنتائج المواضيع
    """
    # تدريب النموذج إذا لم يكن مدرّباً بعد
    if extractor['topic_model'] is None:
        extractor = train_topic_model(extractor, documents)

    if extractor['topic_model'] is None:
        return {
            'topic_info':        pd.DataFrame(),
            'topics_keywords':   {},
            'document_topics':   [OUTLIER_TOPIC_ID] * len(documents),
            'topic_probabilities': [0.0] * len(documents),
            'n_topics':          0
        }

    topic_model = extractor['topic_model']
    topic_info = topic_model.get_topic_info()

    # استخراج الكلمات المفتاحية لكل موضوع
    topics_keywords = {}
    for topic_id in topic_info['Topic']:
        if topic_id != OUTLIER_TOPIC_ID:
            topic_words = topic_model.get_topic(topic_id)
            topics_keywords[topic_id] = [word for word, score in topic_words[:top_n_words]]

    topics, probs = topic_model.transform(documents)

    unique_topic_ids = set(topics)
    n_topics = len(unique_topic_ids) - (1 if OUTLIER_TOPIC_ID in unique_topic_ids else 0)

    return {
        'topic_info':          topic_info,
        'topics_keywords':     topics_keywords,
        'document_topics':     topics,
        'topic_probabilities': probs,
        'n_topics':            n_topics
    }


def extract_topics_from_texts(texts):
    """
    الدالة الرئيسية لاستخراج المواضيع من قائمة نصوص.

    texts  : قائمة النصوص
    يُرجع : (قائمة تسميات المواضيع, قائمة معرّفات المواضيع, قاموس الكلمات المفتاحية)
    """
    if not texts or len(texts) == 0:
        return [], [], {}

    # تصفية النصوص الصالحة
    valid_texts = []
    for t in texts:
        if t and str(t).strip() and str(t).lower() != "no answer":
            valid_texts.append(str(t))

    if len(valid_texts) < MIN_TOPIC_SIZE:
        print(f"⚠️ Insufficient texts for topic modeling (need {MIN_TOPIC_SIZE}, got {len(valid_texts)})")
        return [DEFAULT_TOPIC_LABEL] * len(texts), [OUTLIER_TOPIC_ID] * len(texts), {}

    try:
        extractor = get_topic_extractor()
        print(f"🔍 Extracting topics from {len(valid_texts)} texts...")

        results = get_topics_from_documents(extractor, valid_texts)

        # إعادة النتائج لتشمل كل النصوص الأصلية
        all_topic_labels = []
        all_topic_ids    = []
        valid_idx = 0

        for text in texts:
            if text and str(text).strip() and str(text).lower() != "no answer":
                topic_id = results['document_topics'][valid_idx]
                all_topic_ids.append(topic_id)

                if topic_id == OUTLIER_TOPIC_ID:
                    all_topic_labels.append(DEFAULT_TOPIC_LABEL)
                else:
                    keywords = results['topics_keywords'].get(topic_id, [])
                    label = ", ".join(keywords) if keywords else f"موضوع {topic_id}"
                    all_topic_labels.append(label)

                valid_idx += 1
            else:
                all_topic_labels.append(UNDEFINED_TOPIC_LABEL)
                all_topic_ids.append(OUTLIER_TOPIC_ID)

        return all_topic_labels, all_topic_ids, results['topics_keywords']

    except Exception as e:
        print(f"⚠️ Error in topic extraction: {str(e)}")
        return [ERROR_TOPIC_LABEL] * len(texts), [OUTLIER_TOPIC_ID] * len(texts), {}


# ============================================================================
# ENRICHMENT PIPELINE
# مسار الإثراء الرئيسي
# ============================================================================

def enrich_survey_data(df):
    """
    تطبيق مسار الإثراء الكامل على DataFrame للاستبيان.
    يضيف أعمدة: sentiment, entities, topic_label, topic_id

    df     : DataFrame يحتوي على عمود Answer_normalized
    يُرجع : نفس الـ DataFrame مع الأعمدة المضافة
    """
    print("\n" + "="*80)
    print("🚀 Starting Survey Enrichment Pipeline")
    print("="*80)

    df = df.copy()
    answer_texts = df["Answer_normalized"].astype(str).tolist()

    # الخطوة 1: تحليل المشاعر
    print("\n📊 Step 1/3: Sentiment Analysis")
    df["sentiment"] = analyze_sentiment_batch(answer_texts)

    # الخطوة 2: استخراج الكيانات
    print("\n🏷️  Step 2/3: Named Entity Recognition")
    df["entities"] = extract_entities_batch(answer_texts)

    # الخطوة 3: استخراج المواضيع
    print("\n🔍 Step 3/3: Topic Extraction")
    topic_labels, topic_ids, _ = extract_topics_from_texts(answer_texts)
    df["topic_label"] = topic_labels
    df["topic_id"]    = topic_ids

    print("\n" + "="*80)
    print("✅ Enrichment Pipeline Complete")
    print("="*80)

    return df


# ============================================================================
# STATISTICAL ANALYSIS
# التحليل الإحصائي
# ============================================================================

def get_sentiment_distribution(df):
    """حساب توزيع المشاعر مع الأعداد والنسب المئوية."""
    if 'sentiment' not in df.columns:
        return pd.DataFrame()

    counts      = df['sentiment'].value_counts()
    percentages = df['sentiment'].value_counts(normalize=True) * 100

    return pd.DataFrame({
        'العدد':               counts,
        'النسبة المئوية (%)': percentages.round(2)
    })


def get_top_topics(df, top_n=10):
    """الحصول على أكثر المواضيع تكراراً في الإجابات."""
    if 'topic_label' not in df.columns:
        return pd.Series()

    exclude_labels = {DEFAULT_TOPIC_LABEL, UNDEFINED_TOPIC_LABEL, ERROR_TOPIC_LABEL, ''}

    topics_series = df['topic_label'].dropna().astype(str)

    # تقسيم المواضيع المفصولة بفواصل
    all_topics = []
    for topic_str in topics_series:
        parts = [p.strip() for p in topic_str.split(',')]
        for p in parts:
            if p not in exclude_labels:
                all_topics.append(p)

    return pd.Series(all_topics).value_counts().head(top_n)


def get_top_topics_by_sentiment(df, sentiment_label='negative', top_n=5):
    """الحصول على أكثر المواضيع تكراراً لمشاعر محددة."""
    if 'sentiment' not in df.columns or 'topic_label' not in df.columns:
        return pd.Series()

    filtered_df = df[df['sentiment'].str.contains(sentiment_label, case=False, na=False)]
    return get_top_topics(filtered_df, top_n)


def get_top_entities(df, sentiment_label='negative', top_n=5):
    """الحصول على أكثر الكيانات ذكراً لمشاعر محددة."""
    if 'sentiment' not in df.columns or 'entities' not in df.columns:
        return pd.Series()

    filtered_df = df[df['sentiment'].str.contains(sentiment_label, case=False, na=False)]

    entity_counts = []
    for _, row in filtered_df.iterrows():
        try:
            raw_entities = row['entities']

            if isinstance(raw_entities, str):
                entities = ast.literal_eval(raw_entities)
            elif isinstance(raw_entities, list):
                entities = raw_entities
            else:
                continue

            for ent in entities:
                if ent['label'] in RELEVANT_ENTITY_TYPES:
                    entity_counts.append(f"{ent['text']} ({ent['label']})")

        except Exception:
            continue

    return pd.Series(entity_counts).value_counts().head(top_n)


def get_top_words(df, sentiment_label='negative', top_n=5):
    """الحصول على أكثر الكلمات تكراراً في إجابات مشاعر محددة."""
    if 'sentiment' not in df.columns or 'Answer_normalized' not in df.columns:
        return pd.Series()

    filtered_df = df[df['sentiment'].str.contains(sentiment_label, case=False, na=False)]
    texts = filtered_df['Answer_normalized'].dropna().astype(str).tolist()

    all_words = []
    for text in texts:
        words = remove_stopwords(text)
        all_words.extend(words)

    return pd.Series(all_words).value_counts().head(top_n)


# ============================================================================
# CONTEXT EXTRACTION FOR LLM
# استخراج السياق للتحليل باستخدام النموذج اللغوي
# ============================================================================

def get_topics_with_answers(df, sentiment_label='negative', top_n=5):
    """
    جمع الإجابات المرتبطة بكل موضوع لتحليلها بالنموذج اللغوي.

    يُرجع: قاموس { اسم الموضوع: { count: العدد, answers: [الإجابات] } }
    """
    if 'sentiment' not in df.columns or 'topic_label' not in df.columns:
        return {}

    filtered_df = df[df['sentiment'].str.contains(sentiment_label, case=False, na=False)]
    exclude_labels = {DEFAULT_TOPIC_LABEL, UNDEFINED_TOPIC_LABEL, ERROR_TOPIC_LABEL, '', 'nan'}

    topic_to_answers = {}

    for _, row in filtered_df.iterrows():
        topic  = str(row.get('topic_label', ''))
        answer = str(row.get('Answer_normalized', ''))

        parts = [p.strip() for p in topic.split(',')]
        for topic_part in parts:
            if topic_part not in exclude_labels:
                if topic_part not in topic_to_answers:
                    topic_to_answers[topic_part] = []
                topic_to_answers[topic_part].append(answer)

    # ترتيب المواضيع حسب الأكثر تكراراً
    topic_counts = {topic: len(answers) for topic, answers in topic_to_answers.items()}
    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    result = {}
    for topic, count in sorted_topics:
        result[topic] = {'count': count, 'answers': topic_to_answers[topic]}

    return result


def get_entities_with_answers(df, sentiment_label='negative', top_n=5):
    """
    جمع الإجابات المرتبطة بكل كيان لتحليلها بالنموذج اللغوي.

    يُرجع: قاموس { اسم الكيان: { count: العدد, answers: [الإجابات] } }
    """
    if 'sentiment' not in df.columns or 'entities' not in df.columns:
        return {}

    filtered_df = df[df['sentiment'].str.contains(sentiment_label, case=False, na=False)]
    entity_to_answers = {}

    for _, row in filtered_df.iterrows():
        answer = str(row.get('Answer_normalized', ''))

        try:
            raw_entities = row.get('entities', [])

            if isinstance(raw_entities, str):
                entities = ast.literal_eval(raw_entities)
            elif isinstance(raw_entities, list):
                entities = raw_entities
            else:
                continue

            for ent in entities:
                if ent['label'] in RELEVANT_ENTITY_TYPES:
                    entity_key = f"{ent['text']} ({ent['label']})"
                    if entity_key not in entity_to_answers:
                        entity_to_answers[entity_key] = []
                    entity_to_answers[entity_key].append(answer)

        except Exception:
            continue

    # ترتيب الكيانات حسب الأكثر تكراراً
    entity_counts  = {entity: len(answers) for entity, answers in entity_to_answers.items()}
    sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    result = {}
    for entity, count in sorted_entities:
        result[entity] = {'count': count, 'answers': entity_to_answers[entity]}

    return result


def get_words_with_answers(df, sentiment_label='negative', top_n=5):
    """
    جمع الإجابات المرتبطة بأكثر الكلمات تكراراً لتحليلها بالنموذج اللغوي.

    يُرجع: قاموس { الكلمة: { count: العدد, answers: [الإجابات] } }
    """
    if 'sentiment' not in df.columns or 'Answer_normalized' not in df.columns:
        return {}

    filtered_df = df[df['sentiment'].str.contains(sentiment_label, case=False, na=False)]
    word_to_answers = {}

    for _, row in filtered_df.iterrows():
        answer = str(row.get('Answer_normalized', ''))
        words  = remove_stopwords(answer)

        for word in words:
            if word not in word_to_answers:
                word_to_answers[word] = []
            word_to_answers[word].append(answer)

    # ترتيب الكلمات حسب الأكثر تكراراً
    word_counts   = {word: len(answers) for word, answers in word_to_answers.items()}
    sorted_words  = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    result = {}
    for word, count in sorted_words:
        result[word] = {'count': count, 'answers': word_to_answers[word]}

    return result


# ============================================================================
# LLM ANALYSIS
# التحليل باستخدام النموذج اللغوي
# ============================================================================

def build_llm_prompt(data_dict, data_type, survey_title, survey_question, total_count=None):
    """
    بناء نص الطلب المُرسل للنموذج اللغوي.

    data_dict      : قاموس البيانات (مواضيع أو كيانات أو كلمات)
    data_type      : نوع البيانات 'topics' | 'entities' | 'words'
    survey_title   : عنوان الاستبيان
    survey_question: نص السؤال
    total_count    : إجمالي الإجابات (اختياري)
    يُرجع         : نص الطلب
    """
    type_labels_singular = {
        'topics':   'الموضوع',
        'entities': 'الكيان',
        'words':    'الكلمة'
    }

    type_label = type_labels_singular.get(data_type, 'العنصر')

    total_line = ""
    if total_count:
        total_line = f"- إجمالي الإجابات المحللة: {total_count}"

    prompt = f"""أنت محلل بيانات خبير متخصص في تحليل استبيانات رضا العملاء باللغة العربية.

        مهمتك: تحليل بيانات الاستبيان المرفقة واستخراج الرؤى الأساسية بطريقة واضحة، أرقام دقيقة، وبمنتهى الاختصار.

        **معلومات الاستبيان:**
        - عنوان الاستبيان: "{survey_title}"
        - السؤال المطروح: "{survey_question}"
        {total_line}

        ## خطوات التحليل:

        1. **التحليل الرقمي (الأرقام والنسب):**
        - اذكر عدد تكرار كل {type_label} والنسبة المئوية من إجمالي البيانات المزودة.

        2. **الرؤى المختصرة:**
        - لماذا يظهر هذا {type_label} بكثرة؟ وما هو سياق ذكره؟

        ## صيغة العرض المطلوبة:

        ```
        ## 📊 مؤشرات {type_label}s:
        - [ال{type_label}]: [العدد] تكرار ([النسبة]%)
        ...

        ## 📝 التقرير المختصر:
        - [ال{type_label}]: [سبب الظهور وأهم الملاحظات باختصار]
        ```

        ## ملاحظات:
        - كن مختصراً جداً.
        - ركز على الأرقام.

        ---

        **البيانات المطلوب تحليلها:**

        """

    for item, data in data_dict.items():
        prompt += f"\n{'='*80}\n"
        prompt += f"**{type_label}: {item}**\n"
        prompt += f"**عدد التكرارات: {data['count']}**\n\n"
        prompt += f"**جميع الإجابات المتعلقة بهذا {type_label}:**\n\n"

        for i, answer in enumerate(data['answers'], 1):
            prompt += f"{i}. {answer}\n"

    prompt += f"\n{'='*80}\n"

    return prompt


def analyze_with_llm(data_dict, data_type, survey_title, survey_question, total_count=None):
    """
    إرسال البيانات للنموذج اللغوي والحصول على التحليل.

    data_dict      : قاموس البيانات
    data_type      : نوع البيانات 'topics' | 'entities' | 'words'
    survey_title   : عنوان الاستبيان
    survey_question: نص السؤال
    total_count    : إجمالي الإجابات (اختياري)
    يُرجع         : نص التحليل من النموذج اللغوي
    """
    if not data_dict:
        return "لا توجد بيانات كافية للتحليل"

    type_labels = {'topics': 'المواضيع', 'entities': 'الكيانات', 'words': 'الكلمات'}
    type_label  = type_labels.get(data_type, 'العناصر')

    print(f"\n🤖 Sending {type_label} to LLM for analysis...")
    print(f"📊 Number of items: {len(data_dict)}")

    total_answers = sum(data['count'] for data in data_dict.values())
    print(f"📝 Total answers: {total_answers}")

    try:
        prompt   = build_llm_prompt(data_dict, data_type, survey_title, survey_question, total_count)
        response = model.invoke(prompt)
        return response.content

    except Exception as e:
        return f"❌ خطأ في الاتصال بـ LLM: {str(e)}"


# ============================================================================
# UNIFIED ANALYSIS PIPELINE
# مسار التحليل الموحد
# ============================================================================

def run_full_analysis_pipeline(df, survey_title, survey_question):
    """
    تشغيل مسار التحليل الكامل: إثراء → إحصاء → تحليل LLM.

    df              : DataFrame الاستبيان
    survey_title    : عنوان الاستبيان
    survey_question : نص السؤال
    يُرجع          : قاموس بجميع نتائج التحليل
    """
    print("\n" + "="*80)
    print("🚀 Starting Complete Analysis Pipeline")
    print("="*80)

    # المرحلة 1: الإثراء
    print("\n📊 Stage 1: Data Enrichment")
    enriched_df = enrich_survey_data(df.copy())

    # المرحلة 2: التحليل الإحصائي
    print("\n📈 Stage 2: Statistical Analysis")

    results = {
        'enriched_df':             enriched_df,
        'sentiment_distribution':  get_sentiment_distribution(enriched_df),
        'top_topics':              get_top_topics(enriched_df, TOP_N_ITEMS),
        'top_topics_by_sentiment': get_top_topics_by_sentiment(enriched_df, SENTIMENT_FILTER, TOP_N_ITEMS),
        'top_entities':            get_top_entities(enriched_df, SENTIMENT_FILTER, TOP_N_ITEMS),
        'top_words':               get_top_words(enriched_df, SENTIMENT_FILTER, TOP_N_ITEMS)
    }

    # حساب عدد الإجابات للمشاعر المحددة
    try:
        filtered_df    = enriched_df[enriched_df['sentiment'].str.contains(SENTIMENT_FILTER, case=False, na=False)]
        total_filtered = len(filtered_df)
    except Exception:
        total_filtered = len(enriched_df)

    # المرحلة 3: التحليل بالنموذج اللغوي
    print("\n🤖 Stage 3: LLM-Powered Insights")

    print("\n  📌 Analyzing topics...")
    topics_data = get_topics_with_answers(enriched_df, SENTIMENT_FILTER, TOP_N_ITEMS)
    results['topics_analysis'] = analyze_with_llm(
        topics_data, 'topics', survey_title, survey_question, total_filtered
    )

    print("\n  📌 Analyzing entities...")
    entities_data = get_entities_with_answers(enriched_df, SENTIMENT_FILTER, TOP_N_ITEMS)
    results['entities_analysis'] = analyze_with_llm(
        entities_data, 'entities', survey_title, survey_question, total_filtered
    )

    print("\n  📌 Analyzing words...")
    words_data = get_words_with_answers(enriched_df, SENTIMENT_FILTER, TOP_N_ITEMS)
    results['words_analysis'] = analyze_with_llm(
        words_data, 'words', survey_title, survey_question, total_filtered
    )

    print("\n" + "="*80)
    print("✅ Complete Analysis Pipeline Finished Successfully")
    print("="*80)

    return results