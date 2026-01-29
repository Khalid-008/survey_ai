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
from analytics_pipeline.pipeline import (
    normalize_arabic, 
    convert_arabic_time_to_24h,
    run_enrichment_pipeline
)

warnings.filterwarnings('ignore')

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
        
        # Run the full analytics pipeline
        survey_df = run_enrichment_pipeline(survey_df)
        
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