"""
Arabic Topic Extraction using BERTopic with CAMeLBERT
"""

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import warnings

warnings.filterwarnings('ignore')

class ArabicTopicExtractor:
    """
    Arabic Topic Extractor using CAMeLBERT
    """
    
    def __init__(
        self,
        model_name: str = "CAMeL-Lab/bert-base-arabic-camelbert-da",
        language: str = "arabic",
        min_topic_size: int = 10,
        nr_topics: Any = None
    ):
        """
        Initialization
        
        Args:
            model_name: CAMeLBERT model name
            language: Language (arabic)
            min_topic_size: Minimum topic size
            nr_topics: Number of topics after reduction
        """
        self.model_name = model_name
        self.language = language
        self.min_topic_size = min_topic_size
        self.nr_topics = nr_topics
        
        # Load embedding model
        print(f"⏳ Loading Model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Setup BERTopic components
        self._setup_components()
        
        # Initialize BERTopic
        self.topic_model = None
        
    def _setup_components(self):
        """Setup model components"""
        
        # UMAP for dimensionality reduction
        self.umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        
        # HDBSCAN for clustering
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_topic_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # CountVectorizer for Arabic words
        self.vectorizer_model = CountVectorizer(
            ngram_range=(1, 2),
            stop_words=self._get_arabic_stopwords(),
            max_features=5000,
            min_df=2
        )
        
        # ClassTfidfTransformer
        self.ctfidf_model = ClassTfidfTransformer()
        
    def _get_arabic_stopwords(self) -> List[str]:
        """Get Arabic stopwords"""
        arabic_stopwords = [
            'في', 'من', 'إلى', 'على', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'التي',
            'الذي', 'أن', 'أو', 'كان', 'كانت', 'لكن', 'ما', 'لا', 'نعم',
            'قد', 'لم', 'لن', 'إن', 'كل', 'بعض', 'غير', 'بين', 'عند',
            'منذ', 'حتى', 'ثم', 'أم', 'إما', 'بل', 'حيث', 'كيف', 'أين',
            'متى', 'لماذا', 'ماذا', 'هل', 'له', 'لها', 'لهم', 'لنا', 'هو',
            'هي', 'هم', 'نحن', 'أنت', 'أنتم', 'انا', 'أنا', 'و', 'أو', 'ف', 'ب',
            'ك', 'ل', 'ال', 'الـ', 'ـ', 'ة', 'ه', 'ها', 'هن', 'هما', 'همـ'
        ]
        return arabic_stopwords
    
    def fit(self, documents: List[str]) -> 'ArabicTopicExtractor':
        """
        Train the model on documents
        
        Args:
            documents: List of Arabic texts
            
        Returns:
            self
        """
        print(f"⏳ Training model on {len(documents)} documents...")
        
        # Initialize BERTopic
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
        
        # Train
        try:
            self.topic_model.fit_transform(documents)
            print(f"✅ Topic model trained successfully")
        except Exception as e:
            print(f"⚠️ Error in topic model training: {str(e)}")
            self.topic_model = None
            
        return self
    
    def extract_topics(
        self,
        documents: List[str],
        top_n_words: int = 10
    ) -> Dict:
        """
        Extract topics from documents
        
        Args:
            documents: List of Arabic texts
            top_n_words: Number of keywords per topic
            
        Returns:
            Dictionary with topic info
        """
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
        
        # Get topic info
        topic_info = self.topic_model.get_topic_info()
        
        # Extract keywords
        topics_keywords = {}
        for topic_id in topic_info['Topic']:
            if topic_id != -1:
                topic_words = self.topic_model.get_topic(topic_id)
                topics_keywords[topic_id] = [
                    word for word, score in topic_words[:top_n_words]
                ]
        
        # Transform documents
        topics, probs = self.topic_model.transform(documents)
        
        return {
            'topic_info': topic_info,
            'topics_keywords': topics_keywords,
            'document_topics': topics,
            'topic_probabilities': probs,
            'n_topics': len(set(topics)) - (1 if -1 in topics else 0)
        }
    
    def get_topic_info(self) -> pd.DataFrame:
        """Get topic info"""
        if self.topic_model is None:
            return pd.DataFrame()
        return self.topic_model.get_topic_info()
    
    def save_model(self, path: str):
        """Save model"""
        if self.topic_model is None:
            raise ValueError("Model must be trained first")
        
        self.topic_model.save(path, serialization="pytorch")
        print(f"✅ Model saved to: {path}")
    
    @classmethod
    def load_model(cls, path: str) -> 'ArabicTopicExtractor':
        """Load saved model"""
        instance = cls()
        instance.topic_model = BERTopic.load(path)
        print(f"✅ Model loaded from: {path}")
        return instance
