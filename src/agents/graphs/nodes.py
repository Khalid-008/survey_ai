import sys
import os
import re
import datetime
import traceback
import pandas as pd
import warnings
import numpy as np
from typing import Literal, List, Tuple, Dict, Any

# Add the 'src' directory to sys.path to resolve absolute imports when running directly
current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from langchain_core.documents import Document
from langchain_core.messages import trim_messages, AIMessage
from llms.models import model, google_embeddings_model
from agents.prompt.get_relevant_question_node_prompt import get_relevant_question_prompt
from langgraph.types import Command
from agents.graphs.setup import State
from data.operations import get_survey_df
from agents.prompt.generate_analysis_from_rag_prompt import generate_analysis_from_qualitative_data_prompt
from agents.prompt.generate_analysis_from_sql_prompt import generate_analysis_from_quantitative_data_prompt
from agents.prompt.generate_correlation_analysis_prompt import generate_correlation_analysis_prompt
from agents.prompt.visualization_generator_prompt import (
    visualization_generator_from_quantitative_data_prompt, 
    visualization_generator_from_qualitative_data_prompt
)
from agents.prompt.visualization_correlation_prompt import visualization_generator_for_correlation_prompt
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
    # Match ranges: (num) (optional م/ص) (الى/ - ) (num) (م/ص)
    pattern = r'(\d+)\s*([مص]?)\s*(?:الى|-|—)\s*(\d+)\s*([مص])'
    
    def replace_match(match):
        h1_str, p1, h2_str, p2 = match.groups()
        h1, h2 = int(h1_str), int(h2_str)
        
        # If p1 (AM/PM) is missing for the first number, assume it matches the second
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
        # Format as HH:00-HH:00
        return f"{h1_24:02d}:00-{h2_24:02d}:00"

    # Use regex sub with the callback function
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

    # Define entity labels in Arabic
    labels = [
        "شخص",      # Person
        "مكان",     # Location
        "منظمة",    # Organization
        "تاريخ",    # Date
        "وقت",      # Time
        "منتج",     # Product
        "حدث"       # Event
    ]
    
    # Filter indices for valid texts
    valid_indices = [
        i for i, t in enumerate(texts) 
        if t and str(t).strip() and str(t).lower() != "no answer"
    ]
    valid_texts = [str(texts[i]) for i in valid_indices]
    
    # Initialize results with empty lists
    all_results = [[] for _ in range(len(texts))]
    
    if not valid_texts:
        return all_results

    try:
        print(f"🔍 Processing NER in batches (batch_size={batch_size})...")
        # GLiNER predict_entities supports lists
        batched_entities = model.predict_entities(valid_texts, labels, threshold=0.3)
        
        # GLiNER returns a list of lists of dictionaries
        for idx, entities in zip(valid_indices, batched_entities):
            all_results[idx] = convert_ner_to_native(entities)
            
    except Exception as e:
        print(f"⚠️ Error in batch entity extraction: {str(e)}")
        # Fallback to empty results already initialized
        
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
            stop_words=self._get_arabic_stopwords(),
            max_features=5000,
            min_df=2
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
    
    # Filter out empty texts
    valid_texts = [str(t) for t in texts if t and str(t).strip() and str(t).lower() != "no answer"]
    
    if len(valid_texts) < min_topic_size:
        print(f"⚠️ Not enough valid texts for topic modeling (minimum {min_topic_size} required)")
        return ["متنوع"] * len(texts), [-1] * len(texts), {}
    
    try:
        extractor = get_topic_extractor(min_topic_size=min_topic_size)
        print(f"🔍 Extracting topics from {len(valid_texts)} texts...")
        results = extractor.extract_topics(valid_texts)
        
        # Map results back to original list length
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
# MAIN WORKFLOW NODES
# ============================================================================

def retrieve_survey_question(state: State):
    """
    CLEANING LAYER: Retrieve and clean survey data.
    Data is NOT saved here - only cleaned and returned.
    """
    try:
        survey_df = get_survey_df(state["survey_id"])
        print(f"Total rows before cleaning: {len(survey_df)}")
        
        # Apply 24h conversion BEFORE stripping AM/PM in normalize_arabic
        survey_df["Answer_normalized"] = survey_df["Answer"].apply(convert_arabic_time_to_24h)
        
        # Apply general normalization
        survey_df["Answer_normalized"] = survey_df["Answer_normalized"].apply(normalize_arabic)

        # Handle null values
        survey_df["Answer_normalized"] = survey_df["Answer_normalized"].fillna("NO ANSWER")

        # Lowercase & trim
        survey_df["Answer_normalized"] = (
            survey_df["Answer_normalized"]
            .str.lower()
            .str.strip()
        )

        # Map answers
        mapping = {
            "نوعا ما": "محايد",
            "الى حد ما": "محايد",
            "نعم": "1",
            "لا": "0"
        }
        survey_df["Answer_normalized"] = survey_df["Answer_normalized"].replace(mapping)

        # Clean time ranges and specific letters
        survey_df["Answer_normalized"] = (
            survey_df["Answer_normalized"]
            .str.replace(r"\bمن\b", "", regex=True)
            .str.replace(r"\bالى\b", "-", regex=True)
            .str.replace(r"\bم\b", "", regex=True)
            .str.replace(r"\bص\b", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        
        # Remove duplicates
        survey_df = survey_df.drop_duplicates(
            subset=["QuestionID", "Answer_normalized"]
        )

        print(f"Total rows after cleaning: {len(survey_df)}")

        # Check if empty
        if survey_df.empty:
            print(f"No data found for survey ID {state['survey_id']}")
            return {
                "survey_data": [],
                "messages": [
                    AIMessage(content=f"No data found for survey ID {state['survey_id']}. Analysis skipped.")
                ]
            }

        # Convert to dict (NOT SAVED YET)
        survey_data = survey_df.to_dict(orient="records")

        return {
            "survey_data": survey_data,
            "messages": [
                AIMessage(
                    content=f"Survey data retrieved and cleaned successfully for survey ID {state['survey_id']}. Ready for analysis."
                )
            ]
        }

    except Exception as e:
        raise RuntimeError(f"Server error: {str(e)}")


def enrich_data(state: State):
    """
    ENRICHMENT LAYER: Apply sentiment analysis, NER, and topic extraction.
    Data is SAVED ONLY AFTER enrichment is complete.
    """
    try:
        # Convert list of dicts to DataFrame
        if not state.get("survey_data"):
            print("No survey data to enrich.")
            return {
                "survey_data": [],
                "messages": [AIMessage(content="No survey data available for enrichment.")]
            }
            
        survey_df = pd.DataFrame(state["survey_data"])
        
        if survey_df.empty:
            print("Survey DataFrame is empty.")
            return {
                "survey_data": [],
                "messages": [AIMessage(content="Survey data is empty, skipping enrichment.")]
            }
        
        # Get models (loaded only once globally)
        sentiment_pipeline = get_sentiment_pipeline()
        ner_model = get_ner_model()
        
        # Apply Sentiment Analysis in Batches
        print("📊 Applying batched sentiment analysis...")
        answer_texts = survey_df["Answer_normalized"].astype(str).tolist()
        
        # Filter valid texts for sentiment
        valid_sent_indices = [
            i for i, t in enumerate(answer_texts) 
            if t and t.strip() and t.lower() != "no answer"
        ]
        valid_sent_texts = [answer_texts[i] for i in valid_sent_indices]
        
        sentiments = ["neutral"] * len(answer_texts)
        if valid_sent_texts:
            print(f"🧪 Processing {len(valid_sent_texts)} sentiments in batches...")
            sent_results = sentiment_pipeline(valid_sent_texts, batch_size=32)
            for idx, res in zip(valid_sent_indices, sent_results):
                sentiments[idx] = res["label"]
        
        survey_df["sentiment"] = sentiments
        
        # Apply GLiNER NER in Batches
        print("🏷️  Applying batched GLiNER NER...")
        survey_df["entities"] = extract_entities_batch(answer_texts, ner_model, batch_size=16)

        # Apply Topic Extraction
        print("🔍 Applying topic extraction...")
        topic_labels, topic_ids, keywords_map = extract_topics_from_texts(
            answer_texts
        )
        survey_df["topic_label"] = topic_labels
        survey_df["topic_id"] = topic_ids
        
        print(f"✅ Enrichment completed successfully!")
        
        # ====================================================================
        # SAVE ENRICHED DATA (ONLY AFTER ENRICHMENT)
        # ====================================================================
        if not os.path.exists("exports"):
            os.makedirs("exports")
            
        csv_path = f"exports/survey_data_enriched_{state['survey_id']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            # Convert entities to string for CSV export
            survey_df_export = survey_df.copy()
            survey_df_export["entities"] = survey_df_export["entities"].apply(str)
            survey_df_export.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"💾 Enriched data exported to: {csv_path}")
        except Exception as e:
            print(f"⚠️ Could not export CSV: {str(e)}")
        
        # Convert back to list of dicts for state
        enriched_data = survey_df.to_dict(orient="records")
        
        return {
            "survey_data": enriched_data,
            "messages": [
                AIMessage(
                    content=f"Data enriched successfully with sentiment analysis, NER, and topic extraction for {len(enriched_data)} records. Exported to {csv_path}"
                )
            ]
        }
        
    except Exception as e:
        print(f"⚠️ Non-fatal ERROR in enrich_data: {str(e)}")
        print(traceback.format_exc())
        return {
            "survey_data": state.get("survey_data", []),
            "messages": [
                AIMessage(
                    content=f"Warning: Data enrichment failed ({str(e)}), but proceeding with workflow using basic data."
                )
            ]
        }