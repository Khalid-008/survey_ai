import os
import re
import datetime
import pandas as pd
import warnings
import numpy as np
from typing import List, Dict, Any, Optional
from transformers import pipeline
from gliner import GLiNER
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer

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
# Analsysis Functions
# ============================================================================

