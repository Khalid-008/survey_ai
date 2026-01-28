import re
import datetime
import traceback
import pandas as pd
import os
from langchain_core.documents import Document
from langchain_core.messages import trim_messages, AIMessage
from llms.models import model, google_embeddings_model
from agents.prompt.get_relevant_question_node_prompt import get_relevant_question_prompt
from langgraph.types import Command
from agents.graphs.setup import State
from typing import Literal
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
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Adding layers
# Cleaning layer
# retrive all survey question with answers from DB
def normalize_arabic(text):
    if text is None:
        return ""
    text = str(text)
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    # preserve Arabic characters, spaces, numbers, hyphens, and colons
    text = re.sub("[^\u0600-\u06FF\s0-9\-:]", "", text) 
    text = re.sub("\s+", " ", text).strip()
    return text

def convert_arabic_time_to_24h(text):
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
            if p == 'م': # PM
                return h if h == 12 else h + 12
            if p == 'ص': # AM
                return 0 if h == 12 else h
            return h

        h1_24 = to_24h(h1, p1)
        h2_24 = to_24h(h2, p2)
        # Format as HH:00-HH:00
        return f"{h1_24:02d}:00-{h2_24:02d}:00"

    # Use regex sub with the callback function
    return re.sub(pattern, replace_match, text)

mapping = {
    "نوعا ما": "محايد",
    "الى حد ما": "محايد",
    "نعم": "1",
    "لا": "0"
}

def retrieve_survey_question(state: State):
    try:
        survey_df = get_survey_df(state["survey_id"])

        print(f"Total rows before cleaning: {len(survey_df)}")
        
        # Apply 24h conversion BEFORE stripping AM/PM in normalize_arabic
        survey_df["Answer_normalized"] = survey_df["Answer"].apply(convert_arabic_time_to_24h)
        
        # Now apply general normalization
        survey_df["Answer_normalized"] = survey_df["Answer_normalized"].apply(normalize_arabic)

        # 1.1 Handle null values
        survey_df["Answer_normalized"] = survey_df["Answer_normalized"].fillna("NO ANSWER")

        # 3. Lowercase & trim
        survey_df["Answer_normalized"] = (
            survey_df["Answer_normalized"]
            .str.lower()
            .str.strip()
        )

        # 4. Map answers
        survey_df["Answer_normalized"] = survey_df["Answer_normalized"].replace(mapping)

        # 5. Clean time ranges and specific letters
        survey_df["Answer_normalized"] = (
            survey_df["Answer_normalized"]
            .str.replace(r"\bمن\b", "", regex=True)
            .str.replace(r"\bالى\b", "-", regex=True)
            .str.replace(r"\bم\b", "", regex=True)
            .str.replace(r"\bص\b", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        
        # 6. Remove duplicates
        survey_df = survey_df.drop_duplicates(
            subset=["QuestionID", "Answer_normalized"]
        )

        print(f"Total rows after cleaning: {len(survey_df)}")

        # 7. Export cleaned CSV
        if not os.path.exists("exports"):
            os.makedirs("exports")
        csv_path = f"exports/survey_data_{state['survey_id']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            survey_df.head(1000).to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"Cleaned data exported to: {csv_path}")
        except Exception as e:
            print(f"Could not save CSV to {csv_path}: {str(e)}")

        # --------------------
        if survey_df.empty:
            print(f"No data found for survey ID {state['survey_id']}")
            return {
                "survey_data": [],
                "messages": [
                    AIMessage(content=f"No data found for survey ID {state['survey_id']}. Analysis skipped.")
                ]
            }

        # 8. Convert to dict
        # --------------------
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
# add tem in data frame to use them 
# clean each type and normlize if needed 
# you should endup with a clean data frame to use it in the next layer

# Enrichment layer
# Initialize pipelines globally to avoid reloading models on every call
_sentiment_pipeline = None
_ner_pipeline = None
_topic_model = None
_embedding_model = None

def get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        print("🔄 Loading sentiment analysis model...")
        _sentiment_pipeline = pipeline("sentiment-analysis", model="CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment", device="cpu")
        print("✅ Sentiment model loaded")
    return _sentiment_pipeline

def get_ner_model():
    global _ner_pipeline
    if _ner_pipeline is None:
        print("🔄 Loading GLiNER Arabic NER model...")
        _ner_pipeline = GLiNER.from_pretrained("NAMAA-Space/gliner_arabic-v2.1")
        print("✅ GLiNER model loaded")
    return _ner_pipeline

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

def extract_entities_with_gliner(text, model):
    """Extract entities using GLiNER with predefined Arabic labels."""
    if not text or str(text).strip() == "" or str(text).lower() == "no answer":
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
    
    try:
        entities = model.predict_entities(str(text), labels, threshold=0.3)
        return convert_ner_to_native(entities)
    except Exception as e:
        print(f"⚠️ Error extracting entities: {str(e)}")
        return []

def get_topic_model():
    """Get or create BERTopic model with CAMeLBERT embeddings."""
    global _topic_model, _embedding_model
    if _topic_model is None:
        print("🔄 Loading CAMeLBERT embedding model for topic modeling...")
        # Use CAMeLBERT for Arabic embeddings
        _embedding_model = SentenceTransformer("CAMeL-Lab/bert-base-arabic-camelbert-msa")
        print("✅ Embedding model loaded")
        
        print("🔄 Initializing BERTopic...")
        _topic_model = BERTopic(
            embedding_model=_embedding_model,
            language="arabic",
            calculate_probabilities=False,  # Faster
            verbose=False
        )
        print("✅ BERTopic initialized")
    return _topic_model

def extract_topics_from_texts(texts):
    """Extract topics from a list of texts using BERTopic."""
    if not texts or len(texts) == 0:
        return [], []
    
    # Filter out empty texts
    valid_texts = [str(t) for t in texts if t and str(t).strip() and str(t).lower() != "no answer"]
    
    if len(valid_texts) < 5:  # BERTopic needs minimum documents
        print("⚠️ Not enough valid texts for topic modeling (minimum 5 required)")
        return ["غير محدد"] * len(texts), [0.0] * len(texts)
    
    try:
        topic_model = get_topic_model()
        print(f"🔍 Extracting topics from {len(valid_texts)} texts...")
        
        # Fit and transform
        topics, probs = topic_model.fit_transform(valid_texts)
        
        # Get topic labels
        topic_labels = []
        for topic_id in topics:
            if topic_id == -1:
                topic_labels.append("متنوع")  # Outlier
            else:
                # Get top words for this topic
                topic_words = topic_model.get_topic(topic_id)
                if topic_words:
                    # Take top 3 words
                    top_words = [word for word, _ in topic_words[:3]]
                    topic_labels.append(", ".join(top_words))
                else:
                    topic_labels.append(f"موضوع {topic_id}")
        
        return topic_labels, topics
    
    except Exception as e:
        print(f"⚠️ Error in topic extraction: {str(e)}")
        return ["خطأ"] * len(texts), [-1] * len(texts)


def enrich_data(state: State):
    try:
        # Convert list of dicts to DataFrame
        import pandas as pd
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
        
        # Apply Sentiment Analysis (only on non-empty answers)
        print("📊 Applying sentiment analysis...")
        survey_df["sentiment"] = survey_df["Answer_normalized"].apply(
            lambda x: sentiment_pipeline(str(x))[0]["label"] if x and str(x).strip() and str(x).lower() != "no answer" else "neutral"
        )
        
        # Apply GLiNER NER to each answer
        print("🏷️  Applying GLiNER NER (Named Entity Recognition)...")
        survey_df["entities"] = survey_df["Answer_normalized"].apply(
            lambda x: extract_entities_with_gliner(x, ner_model)
        )
        
        print(f"✅ Enrichment completed successfully!")
        
        # Export enriched CSV
        csv_path = f"exports/survey_data_enriched_{state['survey_id']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            # Convert entities to string for CSV export
            survey_df_export = survey_df.copy()
            survey_df_export["entities"] = survey_df_export["entities"].apply(str)
            survey_df_export.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"💾 Enriched data exported to: {csv_path}")
        except Exception as e:
            print(f"Could not export CSV: {str(e)}")
        
        # Convert back to list of dicts for state
        enriched_data = survey_df.to_dict(orient="records")
        
        return {
            "survey_data": enriched_data,
            "messages": [
                AIMessage(
                    content=f"Data enriched successfully with sentiment analysis and NER for {len(enriched_data)} records. Exported to {csv_path}"
                )
            ]
        }
    except Exception as e:
        print(f"⚠️  Non-fatal ERROR in enrich_data: {str(e)}")
        print(traceback.format_exc())
        return {
            "survey_data": state.get("survey_data", []),
            "messages": [
                AIMessage(
                    content=f"Warning: Data enrichment failed ({str(e)}), but proceeding with workflow using basic data."
                )
            ]
        }