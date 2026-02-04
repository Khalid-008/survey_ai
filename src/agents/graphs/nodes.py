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
    run_full_analysis_pipeline
)
from agents.prompt.synthesis_agent_prompt import synthesis_agent_prompt

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
                "analysis_results": {},
                "messages": [
                    AIMessage(content=f"No data found for survey ID {state['survey_id']}. Analysis skipped.")
                ]
            }

        # Convert to dict (NOT SAVED YET)
        survey_data = survey_df.to_dict(orient="records")

        return {
            "survey_data": survey_data,
            "analysis_results": {},
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
        
        # Get survey title (fallback to ID if SurveyTitle column is missing)
        survey_title = survey_df["SurveyTitle"].iloc[0] if "SurveyTitle" in survey_df.columns else f"Survey {state['survey_id']}"
        
        # We will process TEXT_INPUT questions one by one
        enriched_groups = []
        all_pipeline_results = {}
        
        # Group by QuestionID to process each question separately
        for question_id, group_df in survey_df.groupby("QuestionID"):
            question_type = group_df["QuestionType"].iloc[0]
            question_text = group_df["Questions"].iloc[0]
            
            if question_type == "TEXT_INPUT":
                print(f"🔍 Analyzing TEXT_INPUT question [{question_id}]: {question_text[:50]}...")
                try:
                    # Run the full analytics pipeline for this specific question
                    pipeline_results = run_full_analysis_pipeline(group_df, survey_title, question_text)
                    pipeline_results['survey_question'] = question_text
                    pipeline_results['survey_title'] = survey_title
                    enriched_groups.append(pipeline_results['enriched_df'])
                    all_pipeline_results[question_id] = pipeline_results
                except Exception as e:
                    print(f"⚠️ Error analyzing question {question_id}: {str(e)}")
                    enriched_groups.append(group_df)
            else:
                # For non-TEXT_INPUT, we just keep the data as is (or could add default columns)
                # But to maintain consistency, we should ensure columns exist
                temp_df = group_df.copy()
                if "sentiment" not in temp_df.columns: temp_df["sentiment"] = "neutral"
                if "entities" not in temp_df.columns: temp_df["entities"] = "[]"
                if "topic_label" not in temp_df.columns: temp_df["topic_label"] = "متنوع"
                if "topic_id" not in temp_df.columns: temp_df["topic_id"] = -1
                enriched_groups.append(temp_df)
        
        # Reconstruct the full DataFrame
        survey_df = pd.concat(enriched_groups, ignore_index=True)
        
        print(f"✅ Enrichment completed successfully for {len(all_pipeline_results)} TEXT_INPUT questions!")
        
        # Convert back to list of dicts for state
        enriched_data = survey_df.to_dict(orient="records")
        
        # Prepare serializable analysis results
        serializable_analysis = {}
        for q_id, results in all_pipeline_results.items():
            q_results = results.copy()
            # Remove DataFrame to avoid redundancy and serialization issues
            if 'enriched_df' in q_results:
                del q_results['enriched_df']
            
            # Convert pandas Series/DataFrames to dicts
            for key, value in q_results.items():
                if isinstance(value, (pd.DataFrame, pd.Series)):
                    q_results[key] = value.to_dict()
            
            serializable_analysis[str(q_id)] = q_results

        return {
            "survey_data": enriched_data,
            "analysis_results": serializable_analysis,
            "messages": [
                AIMessage(
                    content=f"Data enriched successfully for {len(all_pipeline_results)} TEXT_INPUT questions. Proceeding to synthesis."
                )
            ]
        }
        
    except Exception as e:
        print(f"⚠️ Non-fatal ERROR in enrich_data: {str(e)}")
        print(traceback.format_exc())
        return {
            "survey_data": state.get("survey_data", []),
            "analysis_results": state.get("analysis_results", {}),
            "messages": [
                AIMessage(
                    content=f"Warning: Data enrichment failed ({str(e)}), but proceeding with workflow using basic data."
                )
            ]
        }

def synthesis_agent(state: State) -> Command[Literal["__end__"]]:
    """
    SYNTHESIS LAYER: Process all analysis results into a single executive report.
    """
    try:
        print(f"=== SYNTHESIS AGENT ===")
        print(f"Processing final synthesis for survey ID: {state['survey_id']}")

        analysis_results = state.get("analysis_results", {})
        if not analysis_results:
            return Command(
                update={"messages": [AIMessage(content="No analysis results found to synthesize.", name="synthesis_agent")]},
                goto="__end__"
            )

        # Get survey subject/title from the first result
        first_result = next(iter(analysis_results.values()))
        survey_title = first_result.get("survey_title", f"Survey {state['survey_id']}")

        # Format analysis data for the LLM
        analytics_messages = []
        for q_id, results in analysis_results.items():
            q_text = results.get("survey_question", f"Question {q_id}")
            sentiment_dist = results.get("sentiment_distribution", "N/A")
            topics_analysis = results.get("topics_analysis", "")
            entities_analysis = results.get("entities_analysis", "")
            
            msg = f"Question: {q_text}\n"
            msg += f"Sentiment Distribution: {sentiment_dist}\n"
            if topics_analysis:
                msg += f"Topics Analysis: {topics_analysis}\n"
            if entities_analysis:
                msg += f"Entities Analysis: {entities_analysis}\n"
            
            analytics_messages.append(msg)

        # Invoke the synthesis prompt
        prompt_messages = synthesis_agent_prompt.invoke({
            "survey_subject": survey_title,
            "analytics_messages": analytics_messages
        })

        response = model.invoke(prompt_messages)
        synthesis_content = response.content

        # Save the executive report as a text file
        if not os.path.exists("exports"):
            os.makedirs("exports")
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"exports/executive_summary_{state['survey_id']}_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(synthesis_content)
        
        print(f"💾 Executive summary saved to: {report_path}")
        
        return Command(
            update={
                "messages": [
                    AIMessage(content=synthesis_content, name="synthesis_agent")
                ],
            },
            goto="__end__"
        )

    except Exception as e:
        print(f"⚠️ Error in synthesis_agent: {str(e)}")
        print(traceback.format_exc())
        return Command(
            update={"messages": [AIMessage(content=f"Synthesis failed: {str(e)}", name="synthesis_agent")]},
            goto="__end__"
        )