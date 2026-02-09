import os
import re
import sys
import ast
import warnings
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

import pandas as pd
import numpy as np
from transformers import pipeline
from gliner import GLiNER
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llms.models import model

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for NLP models."""
    sentiment_model: str = "CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment"
    ner_model: str = "NAMAA-Space/gliner_arabic-v2.1"
    topic_model: str = "CAMeL-Lab/bert-base-arabic-camelbert-da"
    sentiment_batch_size: int = 32
    ner_batch_size: int = 16
    chunk_size: int = 5000  # Process large datasets in chunks to avoid memory issues


@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters."""
    min_topic_size: int = 15
    top_n_items: int = 15
    top_n_words: int = 15
    sentiment_filter: str = 'negative'


# NER entity labels
NER_LABELS = ["شخص", "مكان", "منظمة", "تاريخ", "وقت", "منتج", "حدث"]

# Relevant entity types for analysis
RELEVANT_ENTITY_TYPES = ['شخص', 'منتج', 'منظمة', 'مكان']

# Arabic stopwords
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

# Default values for missing data
DEFAULT_TOPIC_LABEL = "متنوع"
UNDEFINED_TOPIC_LABEL = "غير محدد"
ERROR_TOPIC_LABEL = "خطأ"
OUTLIER_TOPIC_ID = -1


# ============================================================================
# GLOBAL MODEL CACHE (Singleton Pattern)
# ============================================================================

class ModelCache:
    """Singleton cache for heavy NLP models to avoid reloading."""
    
    _instance = None
    _sentiment_pipeline = None
    _ner_model = None
    _topic_extractor = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_sentiment_pipeline(self, config: ModelConfig = None) -> Any:
        """Get or load sentiment analysis pipeline."""
        if self._sentiment_pipeline is None:
            config = config or ModelConfig()
            print(f"🔄 Loading sentiment model: {config.sentiment_model}")
            self._sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=config.sentiment_model,
                device="cpu"
            )
            print("✅ Sentiment model loaded")
        return self._sentiment_pipeline
    
    def get_ner_model(self, config: ModelConfig = None) -> GLiNER:
        """Get or load GLiNER Arabic NER model."""
        if self._ner_model is None:
            config = config or ModelConfig()
            print(f"🔄 Loading NER model: {config.ner_model}")
            self._ner_model = GLiNER.from_pretrained(config.ner_model)
            print("✅ NER model loaded")
        return self._ner_model
    
    def get_topic_extractor(self, min_topic_size: int = 2) -> 'ArabicTopicExtractor':
        """Get or create ArabicTopicExtractor."""
        if self._topic_extractor is None:
            print("🔄 Initializing ArabicTopicExtractor...")
            self._topic_extractor = ArabicTopicExtractor(min_topic_size=min_topic_size)
            print("✅ ArabicTopicExtractor initialized")
        return self._topic_extractor


# Global cache instance
_model_cache = ModelCache()


# ============================================================================
# TEXT NORMALIZATION UTILITIES
# ============================================================================

class ArabicTextNormalizer:
    """Handles Arabic text normalization and cleaning."""
    
    @staticmethod
    def normalize_arabic(text: str) -> str:
        """
        Normalize Arabic text by standardizing characters.
        
        Args:
            text: Input Arabic text
            
        Returns:
            Normalized text
        """
        if text is None:
            return ""
        
        text = str(text)
        
        # Normalize Arabic letters
        text = re.sub("[إأآا]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ة", "ه", text)
        
        # Keep only Arabic characters, spaces, numbers, hyphens, and colons
        text = re.sub("[^\u0600-\u06FF\s0-9\-:]", "", text)
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    @staticmethod
    def convert_arabic_time_to_24h(text: str) -> str:
        """
        Convert Arabic time formats (with م/ص indicators) to 24-hour format.
        
        Args:
            text: Text containing Arabic time expressions
            
        Returns:
            Text with time converted to 24-hour format
        """
        if not isinstance(text, str):
            return text
        
        # Pattern: number + optional م/ص + range indicator + number + م/ص
        pattern = r'(\d+)\s*([مص]?)\s*(?:الى|-|—)\s*(\d+)\s*([مص])'
        
        def replace_match(match):
            h1_str, p1, h2_str, p2 = match.groups()
            h1, h2 = int(h1_str), int(h2_str)
            
            # If first period is missing, assume it's the same as second
            if not p1:
                p1 = p2
            
            def to_24h(hour: int, period: str) -> int:
                """Convert 12-hour format to 24-hour based on period."""
                if period == 'م':  # PM (مساء)
                    return hour if hour == 12 else hour + 12
                if period == 'ص':  # AM (صباح)
                    return 0 if hour == 12 else hour
                return hour
            
            h1_24 = to_24h(h1, p1)
            h2_24 = to_24h(h2, p2)
            
            return f"{h1_24:02d}:00-{h2_24:02d}:00"
        
        return re.sub(pattern, replace_match, text)
    
    @staticmethod
    def remove_stopwords(text: str, stopwords: set = None) -> List[str]:
        """
        Remove stopwords from Arabic text and return word list.
        
        Args:
            text: Input text
            stopwords: Set of stopwords to remove
            
        Returns:
            List of non-stopword tokens
        """
        stopwords = stopwords or ARABIC_STOPWORDS
        
        # Keep only Arabic characters
        clean_text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
        
        words = []
        for word in clean_text.split():
            if len(word) > 2 and word not in stopwords:
                words.append(word)
        
        return words


# ============================================================================
# SENTIMENT ANALYSIS
# ============================================================================

class SentimentAnalyzer:
    """Handles batch sentiment analysis of Arabic text."""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.pipeline = _model_cache.get_sentiment_pipeline(self.config)
    
    def analyze_batch(self, texts: List[str], default_sentiment: str = "neutral") -> List[str]:
        """
        Analyze sentiment for a list of texts with progress tracking and chunked processing.
        
        Args:
            texts: List of text strings to analyze
            default_sentiment: Default sentiment for invalid/empty texts
            
        Returns:
            List of sentiment labels
        """
        print(f"📊 Analyzing sentiment for {len(texts)} texts...")
        
        # Filter valid texts and track their indices
        valid_indices = [
            i for i, t in enumerate(texts)
            if t and str(t).strip() and str(t).lower() != "no answer"
        ]
        valid_texts = [str(texts[i]) for i in valid_indices]
        
        # Initialize results with default sentiment
        sentiments = [default_sentiment] * len(texts)
        
        if not valid_texts:
            print("⚠️ No valid texts to analyze")
            return sentiments
        
        print(f"   Valid texts: {len(valid_texts)}/{len(texts)}")
        
        try:
            # Process in chunks to avoid memory issues with large datasets
            chunk_size = self.config.chunk_size
            all_results = []
            
            # Create progress bar
            num_chunks = (len(valid_texts) + chunk_size - 1) // chunk_size
            
            with tqdm(total=len(valid_texts), desc="   Sentiment Analysis", unit="text") as pbar:
                for i in range(0, len(valid_texts), chunk_size):
                    chunk = valid_texts[i:i + chunk_size]
                    
                    # Process chunk with batch_size
                    chunk_results = self.pipeline(chunk, batch_size=self.config.sentiment_batch_size)
                    all_results.extend(chunk_results)
                    
                    # Update progress bar
                    pbar.update(len(chunk))
            
            # Map results back to original indices
            for idx, result in zip(valid_indices, all_results):
                sentiments[idx] = result["label"]
            
            print(f"✅ Sentiment analysis complete")
            
        except Exception as e:
            print(f"⚠️ Error in sentiment analysis: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return sentiments


# ============================================================================
# NAMED ENTITY RECOGNITION
# ============================================================================

class NamedEntityRecognizer:
    """Handles NER extraction from Arabic text."""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model = _model_cache.get_ner_model(self.config)
    
    @staticmethod
    def convert_to_native_types(entities: List[Dict]) -> List[Dict]:
        """
        Convert GLiNER results to native Python types for JSON serialization.
        
        Args:
            entities: List of entity dictionaries from GLiNER
            
        Returns:
            List of serializable entity dictionaries
        """
        if not entities:
            return []
        
        native_entities = []
        for entity in entities:
            native_entity = {
                'text': str(entity.get('text', '')),
                'label': str(entity.get('label', '')),
                'score': float(entity.get('score', 0.0)),
                'start': int(entity.get('start', 0)),
                'end': int(entity.get('end', 0))
            }
            native_entities.append(native_entity)
        
        return native_entities
    
    def extract_batch(self, texts: List[str], labels: List[str] = None, threshold: float = 0.3) -> List[List[Dict[str, Any]]]:
        """
        Extract named entities from texts with progress tracking and chunked processing.
        
        Args:
            texts: List of text strings to process
            labels: Entity labels to extract
            threshold: Confidence threshold for entity extraction
            
        Returns:
            List of entity lists for each text
        """
        labels = labels or NER_LABELS
        print(f"🏷️  Extracting entities from {len(texts)} texts...")
        
        # Filter valid texts
        valid_indices = [
            i for i, t in enumerate(texts)
            if t and str(t).strip() and str(t).lower() != "no answer"
        ]
        valid_texts = [str(texts[i]) for i in valid_indices]
        
        # Initialize empty results
        all_results = [[] for _ in range(len(texts))]
        
        if not valid_texts:
            print("⚠️ No valid texts to process")
            return all_results
        
        print(f"   Valid texts: {len(valid_texts)}/{len(texts)}")
        
        try:
            # Process in chunks
            chunk_size = self.config.chunk_size
            all_batched_entities = []
            
            with tqdm(total=len(valid_texts), desc="   NER Extraction", unit="text") as pbar:
                for i in range(0, len(valid_texts), chunk_size):
                    chunk = valid_texts[i:i + chunk_size]
                    
                    # Batch prediction for chunk
                    chunk_entities = self.model.batch_predict_entities(
                        chunk,
                        labels,
                        threshold=threshold,
                        batch_size=self.config.ner_batch_size
                    )
                    all_batched_entities.extend(chunk_entities)
                    
                    # Update progress bar
                    pbar.update(len(chunk))
            
            # Map results back to original indices
            for idx, entities in zip(valid_indices, all_batched_entities):
                all_results[idx] = self.convert_to_native_types(entities)
            
            print(f"✅ Entity extraction complete")
            
        except Exception as e:
            print(f"⚠️ Error in entity extraction: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return all_results


# ============================================================================
# TOPIC EXTRACTION
# ============================================================================

class ArabicTopicExtractor:
    """Topic modeling for Arabic text using BERTopic with CAMeLBERT."""
    
    def __init__(
        self,
        model_name: str = None,
        language: str = "arabic",
        min_topic_size: int = 10,
        nr_topics: Optional[int] = None
    ):
        config = ModelConfig()
        self.model_name = model_name or config.topic_model
        self.language = language
        self.min_topic_size = min_topic_size
        self.nr_topics = nr_topics
        
        print(f"⏳ Loading embedding model: {self.model_name}")
        self.embedding_model = SentenceTransformer(self.model_name)
        
        self._setup_components()
        self.topic_model = None
    
    def _setup_components(self):
        """Initialize UMAP, HDBSCAN, and vectorizer components."""
        self.umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_topic_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        self.vectorizer_model = CountVectorizer(
            ngram_range=(1, 2),
            stop_words=list(ARABIC_STOPWORDS),
            max_features=5000,
            min_df=2
        )
        
        self.ctfidf_model = ClassTfidfTransformer()
    
    def fit(self, documents: List[str]) -> 'ArabicTopicExtractor':
        """
        Train the topic model on documents.
        
        Args:
            documents: List of documents to train on
            
        Returns:
            Self for method chaining
        """
        print(f"⏳ Training topic model on {len(documents)} documents...")
        print("   This may take several minutes for large datasets...")
        
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
            with tqdm(total=3, desc="   Topic Modeling", unit="stage") as pbar:
                pbar.set_description("   Generating embeddings")
                self.topic_model.fit_transform(documents)
                pbar.update(3)
            print("✅ Topic model trained successfully")
        except Exception as e:
            print(f"⚠️ Error training topic model: {str(e)}")
            import traceback
            traceback.print_exc()
            self.topic_model = None
        
        return self
    
    def extract_topics(self,documents: List[str],top_n_words: int = 10) -> Dict[str, Any]:
        if self.topic_model is None:
            self.fit(documents)
        
        if self.topic_model is None:
            return {
                'topic_info': pd.DataFrame(),
                'topics_keywords': {},
                'document_topics': [OUTLIER_TOPIC_ID] * len(documents),
                'topic_probabilities': [0.0] * len(documents),
                'n_topics': 0
            }
        
        # Get topic information
        topic_info = self.topic_model.get_topic_info()
        
        # Extract keywords for each topic
        topics_keywords = {}
        for topic_id in topic_info['Topic']:
            if topic_id != OUTLIER_TOPIC_ID:
                topic_words = self.topic_model.get_topic(topic_id)
                topics_keywords[topic_id] = [
                    word for word, score in topic_words[:top_n_words]
                ]
        
        # Transform documents to get topic assignments
        topics, probs = self.topic_model.transform(documents)
        
        return {
            'topic_info': topic_info,
            'topics_keywords': topics_keywords,
            'document_topics': topics,
            'topic_probabilities': probs,
            'n_topics': len(set(topics)) - (1 if OUTLIER_TOPIC_ID in topics else 0)
        }


class TopicExtractionService:
    """Service for extracting topics from text collections."""
    
    @staticmethod
    def extract_topics(texts: List[str],min_topic_size: int = 2) -> Tuple[List[str], List[int], Dict[int, List[str]]]:
        if not texts or len(texts) == 0:
            return [], [], {}
        
        # Filter valid texts
        valid_texts = [
            str(t) for t in texts
            if t and str(t).strip() and str(t).lower() != "no answer"
        ]
        
        if len(valid_texts) < min_topic_size:
            print(f"⚠️ Insufficient texts for topic modeling (need {min_topic_size}, got {len(valid_texts)})")
            return [DEFAULT_TOPIC_LABEL] * len(texts), [OUTLIER_TOPIC_ID] * len(texts), {}
        
        try:
            extractor = _model_cache.get_topic_extractor(min_topic_size=min_topic_size)
            print(f"🔍 Extracting topics from {len(valid_texts)} texts...")
            
            results = extractor.extract_topics(valid_texts)
            
            # Map results back to all texts
            all_topic_labels = []
            all_topic_ids = []
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
# ============================================================================

class SurveyEnrichmentPipeline:
    """Main pipeline for enriching survey data with NLP analytics."""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.sentiment_analyzer = SentimentAnalyzer(self.config)
        self.ner = NamedEntityRecognizer(self.config)
    
    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply full enrichment pipeline to survey DataFrame.
        
        Args:
            df: DataFrame with survey responses
            
        Returns:
            Enriched DataFrame with sentiment, entities, and topics
        """
        print("\n" + "="*80)
        print("🚀 Starting Survey Enrichment Pipeline")
        print("="*80)
        
        df = df.copy()
        answer_texts = df["Answer_normalized"].astype(str).tolist()
        
        # Step 1: Sentiment Analysis
        print("\n📊 Step 1/3: Sentiment Analysis")
        df["sentiment"] = self.sentiment_analyzer.analyze_batch(answer_texts)
        
        # Step 2: Named Entity Recognition
        print("\n🏷️  Step 2/3: Named Entity Recognition")
        df["entities"] = self.ner.extract_batch(answer_texts)
        
        # Step 3: Topic Extraction
        print("\n🔍 Step 3/3: Topic Extraction")
        topic_labels, topic_ids, _ = TopicExtractionService.extract_topics(answer_texts)
        df["topic_label"] = topic_labels
        df["topic_id"] = topic_ids
        
        print("\n" + "="*80)
        print("✅ Enrichment Pipeline Complete")
        print("="*80)
        
        return df


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

class StatisticalAnalyzer:
    """Performs statistical analysis on enriched survey data."""
    
    @staticmethod
    def get_sentiment_distribution(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate sentiment distribution with counts and percentages."""
        if 'sentiment' not in df.columns:
            return pd.DataFrame()
        
        counts = df['sentiment'].value_counts()
        percentages = df['sentiment'].value_counts(normalize=True) * 100
        
        return pd.DataFrame({
            'العدد': counts,
            'النسبة المئوية (%)': percentages.round(2)
        })
    
    @staticmethod
    def get_top_topics(
        df: pd.DataFrame,
        top_n: int = 10,
        exclude_labels: set = None
    ) -> pd.Series:
        """Get most frequent topics from survey responses."""
        if 'topic_label' not in df.columns:
            return pd.Series()
        
        exclude_labels = exclude_labels or {DEFAULT_TOPIC_LABEL, UNDEFINED_TOPIC_LABEL, ERROR_TOPIC_LABEL, ''}
        
        topics_series = df['topic_label'].dropna().astype(str)
        
        # Split comma-separated topics
        all_topics = []
        for topic_str in topics_series:
            parts = [p.strip() for p in topic_str.split(',')]
            all_topics.extend([p for p in parts if p not in exclude_labels])
        
        return pd.Series(all_topics).value_counts().head(top_n)
    
    @staticmethod
    def get_top_topics_by_sentiment(
        df: pd.DataFrame,
        sentiment_label: str = 'negative',
        top_n: int = 5
    ) -> pd.Series:
        """Get most frequent topics for a specific sentiment."""
        if 'sentiment' not in df.columns or 'topic_label' not in df.columns:
            return pd.Series()
        
        filtered_df = df[df['sentiment'].str.contains(sentiment_label, case=False, na=False)]
        return StatisticalAnalyzer.get_top_topics(filtered_df, top_n)
    
    @staticmethod
    def get_top_entities(
        df: pd.DataFrame,
        sentiment_label: str = 'negative',
        top_n: int = 5,
        entity_types: List[str] = None
    ) -> pd.Series:
        """Get most frequent entities for a specific sentiment."""
        if 'sentiment' not in df.columns or 'entities' not in df.columns:
            return pd.Series()
        
        entity_types = entity_types or RELEVANT_ENTITY_TYPES
        filtered_df = df[df['sentiment'].str.contains(sentiment_label, case=False, na=False)]
        
        entity_counts = []
        for _, row in filtered_df.iterrows():
            try:
                raw_entities = row['entities']
                
                # Handle different entity formats
                if isinstance(raw_entities, str):
                    entities = ast.literal_eval(raw_entities)
                elif isinstance(raw_entities, list):
                    entities = raw_entities
                else:
                    continue
                
                # Extract relevant entities
                for ent in entities:
                    if ent['label'] in entity_types:
                        entity_counts.append(f"{ent['text']} ({ent['label']})")
            
            except Exception:
                continue
        
        return pd.Series(entity_counts).value_counts().head(top_n)
    
    @staticmethod
    def get_top_words(
        df: pd.DataFrame,
        sentiment_label: str = 'negative',
        top_n: int = 5
    ) -> pd.Series:
        """Get most frequent words in responses for a specific sentiment."""
        if 'sentiment' not in df.columns or 'Answer_normalized' not in df.columns:
            return pd.Series()
        
        filtered_df = df[df['sentiment'].str.contains(sentiment_label, case=False, na=False)]
        texts = filtered_df['Answer_normalized'].dropna().astype(str).tolist()
        
        normalizer = ArabicTextNormalizer()
        all_words = []
        
        for text in texts:
            words = normalizer.remove_stopwords(text)
            all_words.extend(words)
        
        return pd.Series(all_words).value_counts().head(top_n)


# ============================================================================
# CONTEXT EXTRACTION FOR LLM
# ============================================================================

class ContextExtractor:
    """Extracts contextual data with full answers for LLM analysis."""
    
    @staticmethod
    def extract_topics_with_context(
        df: pd.DataFrame,
        sentiment_label: str = 'negative',
        top_n: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """Extract topics with all associated answers."""
        if 'sentiment' not in df.columns or 'topic_label' not in df.columns:
            return {}
        
        filtered_df = df[df['sentiment'].str.contains(sentiment_label, case=False, na=False)]
        topic_to_answers = {}
        
        exclude_labels = {DEFAULT_TOPIC_LABEL, UNDEFINED_TOPIC_LABEL, ERROR_TOPIC_LABEL, '', 'nan'}
        
        for _, row in filtered_df.iterrows():
            topic = str(row.get('topic_label', ''))
            answer = str(row.get('Answer_normalized', ''))
            
            # Split comma-separated topics
            parts = [p.strip() for p in topic.split(',')]
            for topic_part in parts:
                if topic_part not in exclude_labels:
                    if topic_part not in topic_to_answers:
                        topic_to_answers[topic_part] = []
                    topic_to_answers[topic_part].append(answer)
        
        # Get top N topics by frequency
        topic_counts = {topic: len(answers) for topic, answers in topic_to_answers.items()}
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return {
            topic: {'count': count, 'answers': topic_to_answers[topic]}
            for topic, count in sorted_topics
        }
    
    @staticmethod
    def extract_entities_with_context(
        df: pd.DataFrame,
        sentiment_label: str = 'negative',
        top_n: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """Extract entities with all associated answers."""
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
        
        # Get top N entities by frequency
        entity_counts = {entity: len(answers) for entity, answers in entity_to_answers.items()}
        sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return {
            entity: {'count': count, 'answers': entity_to_answers[entity]}
            for entity, count in sorted_entities
        }
    
    @staticmethod
    def extract_words_with_context(
        df: pd.DataFrame,
        sentiment_label: str = 'negative',
        top_n: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """Extract frequent words with all associated answers."""
        if 'sentiment' not in df.columns or 'Answer_normalized' not in df.columns:
            return {}
        
        filtered_df = df[df['sentiment'].str.contains(sentiment_label, case=False, na=False)]
        word_to_answers = {}
        normalizer = ArabicTextNormalizer()
        
        for _, row in filtered_df.iterrows():
            answer = str(row.get('Answer_normalized', ''))
            words = normalizer.remove_stopwords(answer)
            
            for word in words:
                if word not in word_to_answers:
                    word_to_answers[word] = []
                word_to_answers[word].append(answer)
        
        # Get top N words by frequency
        word_counts = {word: len(answers) for word, answers in word_to_answers.items()}
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return {
            word: {'count': count, 'answers': word_to_answers[word]}
            for word, count in sorted_words
        }


# ============================================================================
# LLM ANALYSIS
# ============================================================================

class LLMAnalyzer:
    """Generates insights using LLM analysis."""
    
    @staticmethod
    def _build_analysis_prompt(
        data_dict: Dict[str, Dict[str, Any]],
        data_type: str,
        survey_title: str,
        survey_question: str,
        total_count: Optional[int] = None
    ) -> str:
        """Build comprehensive analysis prompt for LLM."""
        
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
        
        return prompt
    
    @staticmethod
    def analyze_data(
        data_dict: Dict[str, Dict[str, Any]],
        data_type: str,
        survey_title: str,
        survey_question: str,
        total_count: Optional[int] = None
    ) -> str:
        """
        Analyze data using LLM.
        
        Args:
            data_dict: Dictionary with items and their associated answers
            data_type: Type of data ('topics', 'entities', 'words')
            survey_title: Survey title
            survey_question: Survey question
            total_count: Total number of responses analyzed
            
        Returns:
            LLM analysis text
        """
        if not data_dict:
            return "لا توجد بيانات كافية للتحليل"
        
        type_labels = {'topics': 'المواضيع', 'entities': 'الكيانات', 'words': 'الكلمات'}
        type_label = type_labels.get(data_type, 'العناصر')
        
        print(f"\n🤖 Sending {type_label} to LLM for analysis...")
        print(f"📊 Number of items: {len(data_dict)}")
        total_answers = sum(data['count'] for data in data_dict.values())
        print(f"📝 Total answers: {total_answers}")
        
        try:
            prompt = LLMAnalyzer._build_analysis_prompt(
                data_dict, data_type, survey_title, survey_question, total_count
            )
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
    config: AnalysisConfig = None
) -> Dict[str, Any]:
    """
    Run complete analysis pipeline: enrichment → statistics → LLM insights.
    
    Args:
        df: Survey DataFrame
        survey_title: Survey title
        survey_question: Survey question text
        config: Analysis configuration
        
    Returns:
        Dictionary containing all analysis results
    """
    config = config or AnalysisConfig()
    
    print("\n" + "="*80)
    print("🚀 Starting Complete Analysis Pipeline")
    print("="*80)
    
    # Stage 1: Enrichment
    print("\n📊 Stage 1: Data Enrichment")
    enrichment_pipeline = SurveyEnrichmentPipeline()
    enriched_df = enrichment_pipeline.enrich(df.copy())
    
    # Stage 2: Statistical Analysis
    print("\n📈 Stage 2: Statistical Analysis")
    analyzer = StatisticalAnalyzer()
    
    results = {
        'enriched_df': enriched_df,
        'sentiment_distribution': analyzer.get_sentiment_distribution(enriched_df),
        'top_topics': analyzer.get_top_topics(enriched_df, config.top_n_items),
        'top_topics_by_sentiment': analyzer.get_top_topics_by_sentiment(
            enriched_df, config.sentiment_filter, config.top_n_items
        ),
        'top_entities': analyzer.get_top_entities(
            enriched_df, config.sentiment_filter, config.top_n_items
        ),
        'top_words': analyzer.get_top_words(
            enriched_df, config.sentiment_filter, config.top_n_items
        )
    }
    
    # Stage 3: LLM Analysis
    print("\n🤖 Stage 3: LLM-Powered Insights")
    
    # Calculate total filtered responses
    try:
        filtered_df = enriched_df[
            enriched_df['sentiment'].str.contains(config.sentiment_filter, case=False, na=False)
        ]
        total_filtered = len(filtered_df)
    except Exception:
        total_filtered = len(enriched_df)
    
    extractor = ContextExtractor()
    llm_analyzer = LLMAnalyzer()
    
    # Topics analysis
    print("\n  📌 Analyzing topics...")
    topics_data = extractor.extract_topics_with_context(
        enriched_df, config.sentiment_filter, config.top_n_items
    )
    results['topics_analysis'] = llm_analyzer.analyze_data(
        topics_data, 'topics', survey_title, survey_question, total_filtered
    )
    
    # Entities analysis
    print("\n  📌 Analyzing entities...")
    entities_data = extractor.extract_entities_with_context(
        enriched_df, config.sentiment_filter, config.top_n_items
    )
    results['entities_analysis'] = llm_analyzer.analyze_data(
        entities_data, 'entities', survey_title, survey_question, total_filtered
    )
    
    # Words analysis
    print("\n  📌 Analyzing words...")
    words_data = extractor.extract_words_with_context(
        enriched_df, config.sentiment_filter, config.top_n_items
    )
    results['words_analysis'] = llm_analyzer.analyze_data(
        words_data, 'words', survey_title, survey_question, total_filtered
    )
    
    print("\n" + "="*80)
    print("✅ Complete Analysis Pipeline Finished Successfully")
    print("="*80)
    
    return results


# ============================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# ============================================================================

def normalize_arabic(text: str) -> str:
    """Legacy compatibility wrapper."""
    return ArabicTextNormalizer.normalize_arabic(text)


def convert_arabic_time_to_24h(text: str) -> str:
    """Legacy compatibility wrapper."""
    return ArabicTextNormalizer.convert_arabic_time_to_24h(text)


def run_enrichment_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy compatibility wrapper."""
    pipeline = SurveyEnrichmentPipeline()
    return pipeline.enrich(df)