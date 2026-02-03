import os
import re
import ast
import datetime
import pandas as pd
import warnings
import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter
from transformers import pipeline
from gliner import GLiNER
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llms.models import model

warnings.filterwarnings('ignore')

# ============================================================================
# GLOBAL MODEL PIPELINES (Loaded once)
# ============================================================================

_sentiment_pipeline = None
_ner_pipeline = None
_topic_extractor = None

# ============================================================================
# HELPER FUNCTIONS - TEXT NORMALIZATION
# ============================================================================

def normalize_arabic(text):
    """Normalize Arabic text by standardizing characters and removing unwanted symbols."""
    if text is None:
        return ""
    text = str(text)
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    # Preserve Arabic characters, spaces, numbers, hyphens, and colons
    text = re.sub("[^\u0600-\u06FF\s0-9\-:]", "", text) 
    text = re.sub("\s+", " ", text).strip()
    return text


def convert_arabic_time_to_24h(text):
    """Convert Arabic time formats to 24-hour format."""
    if not isinstance(text, str):
        return text
    
    # Pattern to match numbers and Arabic PM/AM indicators
    pattern = r'(\d+)\s*([مص]?)\s*(?:الى|-|—)\s*(\d+)\s*([مص])'
    
    def replace_match(match):
        h1_str, p1, h2_str, p2 = match.groups()
        h1, h2 = int(h1_str), int(h2_str)
        
        if not p1:
            p1 = p2
            
        def to_24h(h, p):
            if p == 'م':  # PM
                return h if h == 12 else h + 12
            if p == 'ص':  # AM
                return 0 if h == 12 else h
            return h

        h1_24 = to_24h(h1, p1)
        h2_24 = to_24h(h2, p2)
        return f"{h1_24:02d}:00-{h2_24:02d}:00"

    return re.sub(pattern, replace_match, text)

# ============================================================================
# HELPER FUNCTIONS - MODEL LOADERS
# ============================================================================

def get_sentiment_pipeline():
    """Get or load sentiment analysis pipeline."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        print("🔄 Loading sentiment analysis model...")
        _sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model="CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment", 
            device="cpu"
        )
        print("✅ Sentiment model loaded")
    return _sentiment_pipeline


def get_ner_model():
    """Get or load GLiNER Arabic NER model."""
    global _ner_pipeline
    if _ner_pipeline is None:
        print("🔄 Loading GLiNER Arabic NER model...")
        _ner_pipeline = GLiNER.from_pretrained("NAMAA-Space/gliner_arabic-v2.1")
        print("✅ GLiNER model loaded")
    return _ner_pipeline


def get_topic_extractor(min_topic_size=2):
    """Get or create ArabicTopicExtractor."""
    global _topic_extractor
    if _topic_extractor is None:
        print("🔄 Loading ArabicTopicExtractor...")
        _topic_extractor = ArabicTopicExtractor(min_topic_size=min_topic_size)
        print("✅ ArabicTopicExtractor loaded")
    return _topic_extractor

# ============================================================================
# HELPER FUNCTIONS - NER
# ============================================================================

def convert_ner_to_native(ner_results):
    """Convert GLiNER results to native Python types for serialization."""
    if not ner_results:
        return []
    
    native_results = []
    for entity in ner_results:
        native_entity = {
            'text': str(entity.get('text', '')),
            'label': str(entity.get('label', '')),
            'score': float(entity.get('score', 0.0)),
            'start': int(entity.get('start', 0)),
            'end': int(entity.get('end', 0))
        }
        native_results.append(native_entity)
    return native_results


def extract_entities_batch(texts: List[str], model: Any, batch_size: int = 16) -> List[List[Dict[str, Any]]]:
    """Extract entities using GLiNER with batch processing."""
    if not texts:
        return []

    labels = ["شخص", "مكان", "منظمة", "تاريخ", "وقت", "منتج", "حدث"]
    
    valid_indices = [
        i for i, t in enumerate(texts) 
        if t and str(t).strip() and str(t).lower() != "no answer"
    ]
    valid_texts = [str(texts[i]) for i in valid_indices]
    
    all_results = [[] for _ in range(len(texts))]
    
    if not valid_texts:
        return all_results

    try:
        print(f"🔍 Processing NER in batches (batch_size={batch_size})...")
        batched_entities = model.batch_predict_entities(valid_texts, labels, threshold=0.3, batch_size=batch_size)
        
        for idx, entities in zip(valid_indices, batched_entities):
            all_results[idx] = convert_ner_to_native(entities)
            
    except Exception as e:
        print(f"⚠️ Error in batch entity extraction: {str(e)}")
        
    return all_results

# ============================================================================
# TOPIC EXTRACTION CLASS
# ============================================================================

class ArabicTopicExtractor:
    """Arabic Topic Extractor using CAMeLBERT."""
    
    def __init__(
        self,
        model_name: str = "CAMeL-Lab/bert-base-arabic-camelbert-da",
        language: str = "arabic",
        min_topic_size: int = 10,
        nr_topics: Any = None
    ):
        self.model_name = model_name
        self.language = language
        self.min_topic_size = min_topic_size
        self.nr_topics = nr_topics
        
        print(f"⏳ Loading Model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self._setup_components()
        self.topic_model = None
        
    def _setup_components(self):
        """Setup UMAP, HDBSCAN, and vectorizer components."""
        self.umap_model = UMAP(
            n_neighbors=15, n_components=5, min_dist=0.0, 
            metric='cosine', random_state=42
        )
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_topic_size, metric='euclidean',
            cluster_selection_method='eom', prediction_data=True
        )
        self.vectorizer_model = CountVectorizer(
            ngram_range=(1, 2), stop_words=self._get_arabic_stopwords(),
            max_features=5000, min_df=2
        )
        self.ctfidf_model = ClassTfidfTransformer()
        
    def _get_arabic_stopwords(self) -> List[str]:
        """Return common Arabic stopwords."""
        return [
            'في', 'من', 'إلى', 'على', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'التي',
            'الذي', 'أن', 'أو', 'كان', 'كانت', 'لكن', 'ما', 'لا', 'نعم',
            'قد', 'لم', 'لن', 'إن', 'كل', 'بعض', 'غير', 'بين', 'عند',
            'منذ', 'حتى', 'ثم', 'أم', 'إما', 'بل', 'حيث', 'كيف', 'أين',
            'متى', 'لماذا', 'ماذا', 'هل', 'له', 'لها', 'لهم', 'لنا', 'هو',
            'هي', 'هم', 'نحن', 'أنت', 'أنتم', 'انا', 'أنا', 'و', 'أو', 'ف', 'ب',
            'ك', 'ل', 'ال', 'الـ', 'ـ', 'ة', 'ه', 'ها', 'هن', 'هما', 'همـ'
        ]
    
    def fit(self, documents: List[str]) -> 'ArabicTopicExtractor':
        """Train the topic model on documents."""
        print(f"⏳ Training model on {len(documents)} documents...")
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
            language=self.language,
            calculate_probabilities=True,
            verbose=True,
            nr_topics=self.nr_topics
        )
        try:
            self.topic_model.fit_transform(documents)
            print(f"✅ Topic model trained successfully")
        except Exception as e:
            print(f"⚠️ Error in topic model training: {str(e)}")
            self.topic_model = None
        return self
    
    def extract_topics(self, documents: List[str], top_n_words: int = 10) -> Dict:
        """Extract topics from documents."""
        if self.topic_model is None:
            self.fit(documents)
            
        if self.topic_model is None:
            return {
                'topic_info': pd.DataFrame(),
                'topics_keywords': {},
                'document_topics': [-1] * len(documents),
                'topic_probabilities': [0.0] * len(documents),
                'n_topics': 0
            }
        
        topic_info = self.topic_model.get_topic_info()
        topics_keywords = {}
        for topic_id in topic_info['Topic']:
            if topic_id != -1:
                topic_words = self.topic_model.get_topic(topic_id)
                topics_keywords[topic_id] = [
                    word for word, score in topic_words[:top_n_words]
                ]
        
        topics, probs = self.topic_model.transform(documents)
        return {
            'topic_info': topic_info,
            'topics_keywords': topics_keywords,
            'document_topics': topics,
            'topic_probabilities': probs,
            'n_topics': len(set(topics)) - (1 if -1 in topics else 0)
        }


def extract_topics_from_texts(texts, min_topic_size=2):
    """Extract topics from a list of texts using ArabicTopicExtractor."""
    if not texts or len(texts) == 0:
        return [], [], {}
    
    valid_texts = [str(t) for t in texts if t and str(t).strip() and str(t).lower() != "no answer"]
    
    if len(valid_texts) < min_topic_size:
        print(f"⚠️ Not enough valid texts for topic modeling (minimum {min_topic_size} required)")
        return ["متنوع"] * len(texts), [-1] * len(texts), {}
    
    try:
        extractor = get_topic_extractor(min_topic_size=min_topic_size)
        print(f"🔍 Extracting topics from {len(valid_texts)} texts...")
        results = extractor.extract_topics(valid_texts)
        
        all_topic_labels = []
        all_topic_ids = []
        valid_idx = 0
        for text in texts:
            if text and str(text).strip() and str(text).lower() != "no answer":
                topic_id = results['document_topics'][valid_idx]
                all_topic_ids.append(topic_id)
                if topic_id == -1:
                    all_topic_labels.append("متنوع")
                else:
                    keywords = results['topics_keywords'].get(topic_id, [])
                    all_topic_labels.append(", ".join(keywords) if keywords else f"موضوع {topic_id}")
                valid_idx += 1
            else:
                all_topic_labels.append("غير محدد")
                all_topic_ids.append(-1)
        
        return all_topic_labels, all_topic_ids, results['topics_keywords']
    
    except Exception as e:
        print(f"⚠️ Error in topic extraction: {str(e)}")
        return ["خطأ"] * len(texts), [-1] * len(texts), {}

# ============================================================================
# PIPELINE ENTRANCE
# ============================================================================

def run_enrichment_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full analytical enrichment pipeline on a DataFrame."""
    sentiment_pipeline = get_sentiment_pipeline()
    ner_model = get_ner_model()
    
    # 1. Batched Sentiment Analysis
    print("📊 Applying batched sentiment analysis...")
    answer_texts = df["Answer_normalized"].astype(str).tolist()
    valid_sent_indices = [
        i for i, t in enumerate(answer_texts) 
        if t and t.strip() and t.lower() != "no answer"
    ]
    valid_sent_texts = [answer_texts[i] for i in valid_sent_indices]
    
    sentiments = ["neutral"] * len(answer_texts)
    if valid_sent_texts:
        sent_results = sentiment_pipeline(valid_sent_texts, batch_size=32)
        for idx, res in zip(valid_sent_indices, sent_results):
            sentiments[idx] = res["label"]
    df["sentiment"] = sentiments
    
    # 2. Batched NER
    print("🏷️  Applying batched GLiNER NER...")
    df["entities"] = extract_entities_batch(answer_texts, ner_model, batch_size=16)

    # 3. Topic Extraction
    print("🔍 Applying topic extraction...")
    topic_labels, topic_ids, _ = extract_topics_from_texts(answer_texts)
    df["topic_label"] = topic_labels
    df["topic_id"] = topic_ids
    
    return df

# ============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ============================================================================

def get_sentiment_distribution(df):
    """حساب التوزيع العام للمشاعر (إيجابي / سلبي / محايد)"""
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
    """ما أكثر المواضيع تكراراً؟"""
    if 'topic_label' not in df.columns:
        return "Topic column not found"
    
    topics_series = df['topic_label'].dropna().astype(str)
    
    all_topics = []
    for t in topics_series:
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
    
    filtered_df = df[df['sentiment'].str.contains(sentiment_label, case=False, na=False)]
    topics_series = filtered_df['topic_label'].dropna().astype(str)
    
    all_topics = []
    for t in topics_series:
        parts = [p.strip() for p in t.split(',')]
        for p in parts:
            if p not in ['متنوع', 'غير محدد', 'خطأ', '']:
                all_topics.append(p)
    
    return pd.Series(all_topics).value_counts().head(top_n)


def get_top_negative_entities(df, top_n=5):
    """ما أكثر الكيانات (منتج / خدمة / موظف) المرتبطة بالسلبية؟"""
    if 'sentiment' not in df.columns or 'entities' not in df.columns:
        return "Required columns not found"
    
    negative_df = df[df['sentiment'].str.contains('negative', case=False, na=False)]
    entity_counts = []
    
    for _, row in negative_df.iterrows():
        try:
            raw_entities = row['entities']
            if isinstance(raw_entities, str):
                entities = ast.literal_eval(raw_entities)
            elif isinstance(raw_entities, list):
                entities = raw_entities
            else:
                entities = []
                
            if isinstance(entities, list):
                for ent in entities:
                    if ent['label'] in ['شخص', 'منتج', 'منظمة', 'مكان']:
                        entity_counts.append(f"{ent['text']} ({ent['label']})")
        except:
            continue
            
    return pd.Series(entity_counts).value_counts().head(top_n)


def get_top_words_in_text(df, sentiment_label='negative', top_n=5):
    """ما الكلمات الأكثر تكراراً في التعليقات؟"""
    if 'sentiment' not in df.columns or 'Answer_normalized' not in df.columns:
        return "Required columns not found"
    
    filtered_df = df[df['sentiment'].str.contains(sentiment_label, case=False, na=False)]
    texts = filtered_df['Answer_normalized'].dropna().astype(str).tolist()
    
    stopwords = set([
        'في', 'من', 'على', 'إلى', 'عن', 'مع', 'ما', 'يا', 'ب', 'ل', 'و', 'ك', 
        'لا', 'نعم', 'كان', 'يكون', 'هذا', 'هذه', 'ذلك', 'الي', 'اللي', 'ان',
        'او', 'هل', 'كل', 'بعد', 'قبل', 'عند', 'حتى', 'بس', 'تم', 'تمت',
        'عدم', 'عدم وجود', 'التي', 'الذي', 'اذا', 'إذا', 'لو', 'غير', 'بدون',
        'ممكن', 'يوجد', 'ليس', 'لم', 'لن', 'أو', 'أن', 'كنت', 'كانت', 'يعني'
    ])
    
    words = []
    for text in texts:
        clean_text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
        for word in clean_text.split():
            if len(word) > 2 and word not in stopwords:
                words.append(word)
    
    top_words = pd.Series(words).value_counts().head(top_n)
    return top_words


# ============================================================================
# LLM CONTEXT EXTRACTION FUNCTIONS
# ============================================================================

def get_topics_with_full_context_for_llm(df, sentiment_label='negative', top_n=5):
    """
    استخراج المواضيع مع كل الإجابات الكاملة لإرسالها إلى LLM للتحليل
    Returns: dict with topic as key and list of all answers as value
    """
    if 'sentiment' not in df.columns or 'topic_label' not in df.columns:
        return "Required columns not found"
    
    filtered_df = df[df['sentiment'].str.contains(sentiment_label, case=False, na=False)]
    topic_to_answers = {}
    
    for idx, row in filtered_df.iterrows():
        topic = str(row.get('topic_label', ''))
        answer = str(row.get('Answer_normalized', ''))
        
        parts = [p.strip() for p in topic.split(',')]
        for t in parts:
            if t not in ['متنوع', 'غير محدد', 'خطأ', '', 'nan']:
                if t not in topic_to_answers:
                    topic_to_answers[t] = []
                topic_to_answers[t].append(answer)
    
    topic_counts = {topic: len(answers) for topic, answers in topic_to_answers.items()}
    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    result = {}
    for topic, count in sorted_topics:
        result[topic] = {
            'count': count,
            'answers': topic_to_answers[topic]
        }
    
    return result


def get_entities_with_full_context_for_llm(df, sentiment_label='negative', top_n=5):
    """
    استخراج الكيانات مع كل الإجابات الكاملة لإرسالها إلى LLM للتحليل
    Returns: dict with entity as key and list of all answers as value
    """
    if 'sentiment' not in df.columns or 'entities' not in df.columns:
        return "Required columns not found"
    
    filtered_df = df[df['sentiment'].str.contains(sentiment_label, case=False, na=False)]
    entity_to_answers = {}
    
    for idx, row in filtered_df.iterrows():
        answer = str(row.get('Answer_normalized', ''))
        try:
            raw_entities = row.get('entities', [])
            if isinstance(raw_entities, str):
                entities = ast.literal_eval(raw_entities)
            elif isinstance(raw_entities, list):
                entities = raw_entities
            else:
                entities = []
                
            if isinstance(entities, list):
                for ent in entities:
                    if ent['label'] in ['شخص', 'منتج', 'منظمة', 'مكان']:
                        entity_key = f"{ent['text']} ({ent['label']})"
                        if entity_key not in entity_to_answers:
                            entity_to_answers[entity_key] = []
                        entity_to_answers[entity_key].append(answer)
        except:
            continue
    
    entity_counts = {entity: len(answers) for entity, answers in entity_to_answers.items()}
    sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    result = {}
    for entity, count in sorted_entities:
        result[entity] = {
            'count': count,
            'answers': entity_to_answers[entity]
        }
    
    return result


def get_words_with_full_context_for_llm(df, sentiment_label='negative', top_n=5):
    """
    استخراج الكلمات مع كل الإجابات الكاملة لإرسالها إلى LLM للتحليل
    Returns: dict with word as key and list of all answers as value
    """
    if 'sentiment' not in df.columns or 'Answer_normalized' not in df.columns:
        return "Required columns not found"
    
    filtered_df = df[df['sentiment'].str.contains(sentiment_label, case=False, na=False)]
    
    stopwords = set([
        'في', 'من', 'على', 'إلى', 'عن', 'مع', 'ما', 'يا', 'ب', 'ل', 'و', 'ك', 
        'لا', 'نعم', 'كان', 'يكون', 'هذا', 'هذه', 'ذلك', 'الي', 'اللي', 'ان',
        'او', 'هل', 'كل', 'بعد', 'قبل', 'عند', 'حتى', 'بس', 'تم', 'تمت',
        'عدم', 'عدم وجود', 'التي', 'الذي', 'اذا', 'إذا', 'لو', 'غير', 'بدون',
        'ممكن', 'يوجد', 'ليس', 'لم', 'لن', 'أو', 'أن', 'كنت', 'كانت', 'يعني'
    ])
    
    word_to_answers = {}
    
    for idx, row in filtered_df.iterrows():
        answer = str(row.get('Answer_normalized', ''))
        clean_text = re.sub(r'[^\u0600-\u06FF\s]', '', answer)
        
        for word in clean_text.split():
            if len(word) > 2 and word not in stopwords:
                if word not in word_to_answers:
                    word_to_answers[word] = []
                word_to_answers[word].append(answer)
    
    word_counts = {word: len(answers) for word, answers in word_to_answers.items()}
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    result = {}
    for word, count in sorted_words:
        result[word] = {
            'count': count,
            'answers': word_to_answers[word]
        }
    
    return result


# ============================================================================
# LLM ANALYSIS FUNCTIONS
# ============================================================================

def analyze_topics_with_llm(topics_data, survey_title, survey_question, total_count=None):
    """
    تحليل المواضيع باستخدام LLM
    
    Args:
        topics_data: dict from get_topics_with_full_context_for_llm
        survey_title: عنوان الاستبيان
        survey_question: السؤال المطروح في الاستبيان
        total_count: إجمالي عدد الإجابات المحللة (اختياري)
    
    Returns:
        str: تحليل LLM للبيانات
    """
    prompt = f"""أنت محلل بيانات خبير متخصص في تحليل استبيانات رضا العملاء باللغة العربية.

    مهمتك: تحليل بيانات الاستبيان المرفقة واستخراج الرؤى الأساسية بطريقة واضحة، مختصرة، ومليئة بالأرقام.

    **معلومات الاستبيان:**
    - عنوان الاستبيان: "{survey_title}"
    - السؤال المطروح: "{survey_question}"
    {f"- إجمالي الإجابات المحللة: {total_count}" if total_count else ""}

    ## خطوات التحليل:

    1. **التحليل الرقمي والمؤشرات (هام جداً):**
    - استخرج جميع الأرقام الممكنة (عدد التكرارات، النسب المئوية التقريبية بناءً على الإجمالي).
    - صنف المواضيع حسب كثافتها.

    2. **تحليل تفاصيل كل موضوع:**
    - لخص النقاط الأساسية باختصار شديد.
    - اذكر أمثلة مختصرة جداً.

    3. **الخلاصة:**
    - ما الذي يجب فعله الآن؟ (Actionable insights).

    ## صيغة العرض المطلوبة (التزام تام بهذا التنسيق):

    ```
    ## 📊 الأرقام والمؤشرات الرئيسية:
    - [اسم الموضوع الأول]: [العدد] تكرار ([النسبة]%)
    - [اسم الموضوع الثاني]: [العدد] تكرار ([النسبة]%)
    ...
    - إجمالي الملاحظات المحللة: {total_count if total_count else "[المجموع]"} ملاحظة

    ## 🔍 التحليل المختصر:
    **1. [اسم الموضوع]:** [شرح في سطرين كحد أقصى مع ذكر أهم مثال]
    **2. [اسم الموضوع]:** [شرح في سطرين كحد أقصى مع ذكر أهم مثال]
    ...

    ## 💡 التوصيات السريعة:
    - [توصية 1]
    - [توصية 2]
    ```

    ## ملاحظات مهمة:
    - كن **مختصراً جداً** ومباشراً.
    - ركز على **الأرقام** في القسم الأول.
    - اجعل التحليل جاهزاً للاستخدام الفوري.

    ---

    **البيانات المطلوب تحليلها:**

    """
    
    for topic, data in topics_data.items():
        prompt += f"\n{'='*80}\n"
        prompt += f"**الموضوع: {topic}**\n"
        prompt += f"**عدد التكرارات: {data['count']}**\n\n"
        prompt += "**جميع الإجابات المتعلقة بهذا الموضوع:**\n\n"
        
        for i, answer in enumerate(data['answers'], 1):
            prompt += f"{i}. {answer}\n"
    
    prompt += f"\n{'='*80}\n"

    print("\n🤖 إرسال البيانات إلى LLM للتحليل...")
    print(f"📊 عدد المواضيع: {len(topics_data)}")
    total_answers = sum(data['count'] for data in topics_data.values())
    print(f"📝 إجمالي عدد الإجابات: {total_answers}")
    
    try:
        response = model.invoke(prompt)
        return response.content
    except Exception as e:
        return f"❌ خطأ في الاتصال بـ LLM: {str(e)}"


def analyze_data_with_llm(data_dict, data_type, survey_title, survey_question, total_count=None):
    """
    تحليل أي نوع من البيانات باستخدام LLM (مواضيع، كيانات، كلمات)
    
    Args:
        data_dict: dict from get_*_with_full_context_for_llm functions
        data_type: نوع البيانات ('topics', 'entities', 'words')
        survey_title: عنوان الاستبيان
        survey_question: السؤال المطروح في الاستبيان
        total_count: إجمالي عدد الإجابات المحللة (اختياري)
    
    Returns:
        str: تحليل LLM للبيانات
    """
    type_labels = {
        'topics': 'الموضوع',
        'entities': 'الكيان',
        'words': 'الكلمة'
    }
    
    type_label = type_labels.get(data_type, 'العنصر')
    
    prompt = f"""أنت محلل بيانات خبير متخصص في تحليل استبيانات رضا العملاء باللغة العربية.

    مهمتك: تحليل بيانات الاستبيان المرفقة واستخراج الرؤى الأساسية بطريقة واضحة، أرقام دقيقة، وبمنتهى الاختصار.

    **معلومات الاستبيان:**
    - عنوان الاستبيان: "{survey_title}"
    - السؤال المطروح: "{survey_question}"
    {f"- إجمالي الإجابات المحللة: {total_count}" if total_count else ""}

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
    
    print(f"\n🤖 إرسال بيانات {type_label} إلى LLM للتحليل...")
    print(f"📊 عدد العناصر: {len(data_dict)}")
    total_answers = sum(data['count'] for data in data_dict.values())
    print(f"📝 إجمالي عدد الإجابات: {total_answers}")
    
    try:
        response = model.invoke(prompt)
        return response.content
    except Exception as e:
        return f"❌ خطأ في الاتصال بـ LLM: {str(e)}"


# ============================================================================
# UNIFIED ANALYSIS PIPELINE
# ============================================================================

def run_full_analysis_pipeline(
    df: pd.DataFrame,
    survey_title: str,
    survey_question: str,
    sentiment_filter: str = 'negative',
    top_n: int = 5,
    include_llm_analysis: bool = True
    ) -> Dict[str, Any]:
    """
    Pipeline شامل يجمع بين إثراء البيانات والتحليل الإحصائي والتحليل الذكي
    
    Args:
        df: DataFrame يحتوي على بيانات الاستبيان
        survey_title: عنوان الاستبيان
        survey_question: السؤال المطروح
        sentiment_filter: فلتر المشاعر ('negative', 'positive', 'neutral')
        top_n: عدد العناصر الأكثر تكراراً
        include_llm_analysis: هل يتم تضمين تحليل LLM؟
    
    Returns:
        dict يحتوي على:
        - enriched_df: البيانات المُثراة
        - sentiment_distribution: توزيع المشاعر
        - top_topics: أكثر المواضيع تكراراً
        - top_entities: أكثر الكيانات تكراراً
        - top_words: أكثر الكلمات تكراراً
        - topics_analysis: تحليل LLM للمواضيع (اختياري)
        - entities_analysis: تحليل LLM للكيانات (اختياري)
        - words_analysis: تحليل LLM للكلمات (اختياري)
    """
    print("\n" + "="*80)
    print("🚀 بدء Pipeline التحليل الشامل")
    print("="*80)
    
    # Step 1: Data Enrichment
    print("\n📊 المرحلة 1: إثراء البيانات...")
    enriched_df = run_enrichment_pipeline(df.copy())
    
    # Step 2: Statistical Analysis
    print("\n📈 المرحلة 2: التحليل الإحصائي...")
    results = {
        'enriched_df': enriched_df,
        'sentiment_distribution': get_sentiment_distribution(enriched_df),
        'top_topics': get_top_topics(enriched_df, top_n=top_n),
        'top_topics_by_sentiment': get_top_topics_by_sentiment(enriched_df, sentiment_filter, top_n),
        'top_entities': get_top_negative_entities(enriched_df, top_n=top_n),
        'top_words': get_top_words_in_text(enriched_df, sentiment_filter, top_n=top_n)
    }
    
    # Step 3: LLM Analysis (Optional)
    if include_llm_analysis:
        print("\n🤖 المرحلة 3: التحليل الذكي باستخدام LLM...")
        
        # Calculate total records for this sentiment_filter to help LLM with percentages
        try:
            filtered_df = enriched_df[enriched_df['sentiment'].str.contains(sentiment_filter, case=False, na=False)]
            total_filtered = len(filtered_df)
        except:
            total_filtered = len(enriched_df)
            
        # Topics Analysis
        print("\n  📌 تحليل المواضيع...")
        topics_data = get_topics_with_full_context_for_llm(enriched_df, sentiment_filter, top_n)
        if isinstance(topics_data, dict) and topics_data:
            results['topics_analysis'] = analyze_topics_with_llm(topics_data, survey_title, survey_question, total_filtered)
        else:
            results['topics_analysis'] = "لا توجد بيانات كافية لتحليل المواضيع"
        
        # Entities Analysis
        print("\n  📌 تحليل الكيانات...")
        entities_data = get_entities_with_full_context_for_llm(enriched_df, sentiment_filter, top_n)
        if isinstance(entities_data, dict) and entities_data:
            results['entities_analysis'] = analyze_data_with_llm(entities_data, 'entities', survey_title, survey_question, total_filtered)
        else:
            results['entities_analysis'] = "لا توجد بيانات كافية لتحليل الكيانات"
        
        # Words Analysis
        print("\n  📌 تحليل الكلمات...")
        words_data = get_words_with_full_context_for_llm(enriched_df, sentiment_filter, top_n)
        if isinstance(words_data, dict) and words_data:
            results['words_analysis'] = analyze_data_with_llm(words_data, 'words', survey_title, survey_question, total_filtered)
        else:
            results['words_analysis'] = "لا توجد بيانات كافية لتحليل الكلمات"
    
    print("\n" + "="*80)
    print("✅ اكتمل Pipeline التحليل الشامل بنجاح!")
    print("="*80)
    
    return results

