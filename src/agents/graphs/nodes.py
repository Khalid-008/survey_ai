import os
import sys
import datetime
import traceback
import warnings
import json
import re
from typing import Literal, Dict, Any, List, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np
from langchain_core.messages import AIMessage
from langgraph.types import Command

# Configure path for imports
current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from llms.models import model
from agents.graphs.setup import State
from data.operations import get_survey_df
from analytics_pipeline.pipeline import (
    normalize_arabic,
    convert_arabic_time_to_24h,
    run_full_analysis_pipeline
)
from agents.prompt.synthesis_agent_prompt import synthesis_agent_prompt
from agents.prompt.chart_generation_prompt import chart_generation_prompt

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

ANSWER_MAPPING = {
    "نوعا ما": "محايد",
    "الى حد ما": "محايد",
    "نعم": "1",
    "لا": "0"
}

CLEANING_PATTERNS = [
    (r"\bمن\b", ""),
    (r"\bالى\b", "-"),
    (r"\bم\b", ""),
    (r"\bص\b", ""),
    (r"\s+", " ")
]

EXPORTS_DIR = "exports"
DEFAULT_SENTIMENT = "neutral"
DEFAULT_TOPIC = "متنوع"
DEFAULT_TOPIC_ID = -1


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SurveyMetrics:
    """Metrics for tracking survey data processing."""
    rows_before_cleaning: int
    rows_after_cleaning: int
    text_input_questions: int
    enriched_questions: int
    failed_questions: int


# ============================================================================
# DATA CLEANING UTILITIES
# ============================================================================

class SurveyDataCleaner:
    """Handles all survey data cleaning operations."""
    
    @staticmethod
    def normalize_answers(df: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization pipeline to survey answers."""
        df = df.copy()
        
        # Time conversion must happen before general normalization
        df["Answer_normalized"] = df["Answer"].apply(convert_arabic_time_to_24h)
        df["Answer_normalized"] = df["Answer_normalized"].apply(normalize_arabic)
        
        # Handle missing values
        df["Answer_normalized"] = df["Answer_normalized"].fillna("NO ANSWER")
        
        # Case normalization
        df["Answer_normalized"] = (
            df["Answer_normalized"]
            .str.lower()
            .str.strip()
        )
        
        return df
    
    @staticmethod
    def apply_answer_mapping(df: pd.DataFrame) -> pd.DataFrame:
        """Map common Arabic responses to standardized values."""
        df = df.copy()
        df["Answer_normalized"] = df["Answer_normalized"].replace(ANSWER_MAPPING)
        return df
    
    @staticmethod
    def clean_text_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Remove unwanted patterns from text."""
        df = df.copy()
        
        for pattern, replacement in CLEANING_PATTERNS:
            df["Answer_normalized"] = df["Answer_normalized"].str.replace(
                pattern, replacement, regex=True
            )
        
        df["Answer_normalized"] = df["Answer_normalized"].str.strip()
        return df
    
    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate responses for the same question."""
        return df.drop_duplicates(subset=["QuestionID", "Answer_normalized"])
    
    @classmethod
    def clean_survey_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Execute full cleaning pipeline."""
        df = cls.normalize_answers(df)
        df = cls.apply_answer_mapping(df)
        df = cls.clean_text_patterns(df)
        df = cls.remove_duplicates(df)
        return df


# ============================================================================
# DATA ENRICHMENT UTILITIES
# ============================================================================

class SurveyDataEnricher:
    """Handles NLP enrichment of survey data."""
    
    @staticmethod
    def add_default_enrichment_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Add default enrichment columns for non-TEXT_INPUT questions."""
        df = df.copy()
        
        if "sentiment" not in df.columns:
            df["sentiment"] = DEFAULT_SENTIMENT
        if "entities" not in df.columns:
            df["entities"] = "[]"
        if "topic_label" not in df.columns:
            df["topic_label"] = DEFAULT_TOPIC
        if "topic_id" not in df.columns:
            df["topic_id"] = DEFAULT_TOPIC_ID
            
        return df
    
    @staticmethod
    def enrich_text_question(
        df: pd.DataFrame,
        survey_title: str,
        question_text: str,
        question_id: str
    ) -> Dict[str, Any]:
        """
        Run full analytics pipeline on a TEXT_INPUT question.
        
        Returns:
            Dict containing enriched_df and analysis results
        """
        print(f"🔍 Analyzing TEXT_INPUT question [{question_id}]: {question_text[:50]}...")
        
        try:
            results = run_full_analysis_pipeline(df, survey_title, question_text)
            results['survey_question'] = question_text
            results['survey_title'] = survey_title
            results['question_id'] = question_id
            
            print(f"✅ Question {question_id} analyzed successfully")
            return results
            
        except Exception as e:
            print(f"⚠️ Error analyzing question {question_id}: {str(e)}")
            raise
    
    @classmethod
    def process_survey_questions(
        cls,
        survey_df: pd.DataFrame,
        survey_title: str
    ) -> tuple[pd.DataFrame, Dict[str, Any], SurveyMetrics]:
        """
        Process all survey questions, enriching TEXT_INPUT types.
        
        Returns:
            Tuple of (enriched_df, analysis_results, metrics)
        """
        enriched_groups = []
        all_analysis_results = {}
        text_input_count = 0
        enriched_count = 0
        failed_count = 0
        
        for question_id, group_df in survey_df.groupby("QuestionID"):
            question_type = group_df["QuestionType"].iloc[0]
            question_text = group_df["Questions"].iloc[0]
            
            if question_type == "TEXT_INPUT":
                text_input_count += 1
                try:
                    results = cls.enrich_text_question(
                        group_df, survey_title, question_text, str(question_id)
                    )
                    enriched_groups.append(results['enriched_df'])
                    all_analysis_results[str(question_id)] = results
                    enriched_count += 1
                    
                except Exception as e:
                    print(f"⚠️ Failed to enrich question {question_id}, using raw data")
                    enriched_groups.append(cls.add_default_enrichment_columns(group_df))
                    failed_count += 1
            else:
                # Non-TEXT_INPUT questions get default enrichment columns
                enriched_groups.append(cls.add_default_enrichment_columns(group_df))
        
        enriched_df = pd.concat(enriched_groups, ignore_index=True)
        
        metrics = SurveyMetrics(
            rows_before_cleaning=0,  # Set by caller
            rows_after_cleaning=0,   # Set by caller
            text_input_questions=text_input_count,
            enriched_questions=enriched_count,
            failed_questions=failed_count
        )
        
        return enriched_df, all_analysis_results, metrics


# ============================================================================
# RESULT SERIALIZATION
# ============================================================================

class ResultSerializer:
    """Handles serialization of analysis results."""
    
    @staticmethod
    def make_serializable(results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert analysis results to JSON-serializable format."""
        serialized = {}
        
        for q_id, q_results in results.items():
            q_results = q_results.copy()
            
            # Remove DataFrame to avoid redundancy
            if 'enriched_df' in q_results:
                del q_results['enriched_df']
            
            # Convert pandas objects to dicts
            for key, value in q_results.items():
                if isinstance(value, (pd.DataFrame, pd.Series)):
                    q_results[key] = value.to_dict()
            
            serialized[str(q_id)] = q_results
        
        return serialized


# ============================================================================
# SYNTHESIS UTILITIES
# ============================================================================

class SynthesisReportGenerator:
    """Generates executive summary reports."""
    
    @staticmethod
    def format_analytics_messages(analysis_results: Dict[str, Any]) -> List[str]:
        """Format analysis results into messages for synthesis."""
        messages = []
        
        for q_id, results in analysis_results.items():
            q_text = results.get("survey_question", f"Question {q_id}")
            sentiment_dist = results.get("sentiment_distribution", "N/A")
            topics_analysis = results.get("topics_analysis", "")
            entities_analysis = results.get("entities_analysis", "")
            
            msg_parts = [f"Question: {q_text}"]
            msg_parts.append(f"Sentiment Distribution: {sentiment_dist}")
            
            if topics_analysis:
                msg_parts.append(f"Topics Analysis: {topics_analysis}")
            if entities_analysis:
                msg_parts.append(f"Entities Analysis: {entities_analysis}")
            
            messages.append("\n".join(msg_parts))
        
        return messages
    
    @staticmethod
    def save_report(content: str, survey_id: str) -> str:
        """Save executive summary to file."""
        os.makedirs(EXPORTS_DIR, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"executive_summary_{survey_id}_{timestamp}.txt"
        filepath = os.path.join(EXPORTS_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"💾 Executive summary saved to: {filepath}")
        return filepath


# ============================================================================
# WORKFLOW NODES
# ============================================================================

def retrieve_survey_question(state: State) -> Dict[str, Any]:
    """
    STAGE 1: Retrieve and clean survey data.
    
    This is the data cleaning layer - no enrichment happens here.
    """
    try:
        survey_id = state["survey_id"]
        survey_df = get_survey_df(survey_id)
        
        rows_before = len(survey_df)
        print(f"📊 Loaded {rows_before} rows for survey {survey_id}")
        
        # Clean the data
        survey_df = SurveyDataCleaner.clean_survey_data(survey_df)
        
        rows_after = len(survey_df)
        print(f"✨ Cleaned data: {rows_after} rows ({rows_before - rows_after} duplicates removed)")
        
        # Validate data
        if survey_df.empty:
            return {
                "survey_data": [],
                "analysis_results": {},
                "messages": [
                    AIMessage(content=f"No data found for survey {survey_id}. Analysis skipped.")
                ]
            }
        
        # Convert to dict (not saved yet - just prepared)
        survey_data = survey_df.to_dict(orient="records")
        
        return {
            "survey_data": survey_data,
            "analysis_results": {},
            "messages": [
                AIMessage(
                    content=f"Survey data retrieved and cleaned: {rows_after} responses ready for analysis."
                )
            ]
        }
        
    except Exception as e:
        error_msg = f"Failed to retrieve survey data: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        raise RuntimeError(error_msg)


def enrich_data(state: State) -> Dict[str, Any]:
    """
    STAGE 2: Apply NLP enrichment to survey data.
    
    Processes TEXT_INPUT questions with sentiment analysis, NER, and topic extraction.
    Data is saved only after enrichment is complete.
    """
    try:
        # Validate input
        if not state.get("survey_data"):
            return {
                "survey_data": [],
                "messages": [AIMessage(content="No survey data available for enrichment.")]
            }
        
        survey_df = pd.DataFrame(state["survey_data"])
        
        if survey_df.empty:
            return {
                "survey_data": [],
                "messages": [AIMessage(content="Survey data is empty, skipping enrichment.")]
            }
        
        # Get survey metadata
        survey_title = (
            survey_df["SurveyTitle"].iloc[0] 
            if "SurveyTitle" in survey_df.columns 
            else f"Survey {state['survey_id']}"
        )
        
        print(f"🔬 Starting enrichment for: {survey_title}")
        
        # Process all questions
        enriched_df, analysis_results, metrics = SurveyDataEnricher.process_survey_questions(
            survey_df, survey_title
        )
        
        # Report metrics
        print(f"📈 Enrichment complete:")
        print(f"   • TEXT_INPUT questions: {metrics.text_input_questions}")
        print(f"   • Successfully enriched: {metrics.enriched_questions}")
        print(f"   • Failed: {metrics.failed_questions}")
        
        # Serialize results
        enriched_data = enriched_df.to_dict(orient="records")
        serializable_analysis = ResultSerializer.make_serializable(analysis_results)
        
        return {
            "survey_data": enriched_data,
            "analysis_results": serializable_analysis,
            "messages": [
                AIMessage(
                    content=f"Data enriched: {metrics.enriched_questions}/{metrics.text_input_questions} "
                            f"TEXT_INPUT questions analyzed successfully."
                )
            ]
        }
        
    except Exception as e:
        error_msg = f"Enrichment failed: {str(e)}"
        print(f"⚠️ {error_msg}")
        print(traceback.format_exc())
        
        # Non-fatal error: proceed with basic data
        return {
            "survey_data": state.get("survey_data", []),
            "analysis_results": state.get("analysis_results", {}),
            "messages": [
                AIMessage(
                    content=f"Warning: {error_msg}. Proceeding with basic data."
                )
            ]
        }


def synthesis_agent(state: State) -> Command[Literal["__end__"]]:
    """
    STAGE 3: Generate executive summary report.
    
    Synthesizes all analysis results into a cohesive executive summary.
    """
    try:
        print("=" * 60)
        print("SYNTHESIS AGENT")
        print("=" * 60)
        print(f"Survey ID: {state['survey_id']}")
        
        analysis_results = state.get("analysis_results", {})
        
        if not analysis_results:
            return Command(
                update={
                    "messages": [
                        AIMessage(
                            content="No analysis results to synthesize.",
                            name="synthesis_agent"
                        )
                    ]
                },
                goto="__end__"
            )
        
        # Extract survey metadata
        first_result = next(iter(analysis_results.values()))
        survey_title = first_result.get("survey_title", f"Survey {state['survey_id']}")
        
        print(f"Survey: {survey_title}")
        print(f"Questions analyzed: {len(analysis_results)}")
        
        # Format analytics for LLM
        analytics_messages = SynthesisReportGenerator.format_analytics_messages(
            analysis_results
        )
        
        # Generate synthesis
        prompt_messages = synthesis_agent_prompt.invoke({
            "survey_subject": survey_title,
            "analytics_messages": analytics_messages
        })
        
        response = model.invoke(prompt_messages)
        synthesis_content = response.content
        
        # Save report
        report_path = SynthesisReportGenerator.save_report(
            synthesis_content,
            state['survey_id']
        )
        
        print("✅ Synthesis complete")
        
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=synthesis_content,
                        name="synthesis_agent"
                    )
                ]
            },
            goto="__end__"
        )
        
    except Exception as e:
        error_msg = f"Synthesis failed: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=error_msg,
                        name="synthesis_agent"
                    )
                ]
            },
            goto="__end__"
        )


# ============================================================================
# CHART GENERATION UTILITIES
# ============================================================================

def extract_json_from_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Extract JSON array from LLM response, handling markdown code blocks and extra text.
    
    Args:
        response_text: Raw response from LLM
        
    Returns:
        Parsed JSON array of chart configurations
    """
    # Remove markdown code blocks
    response_text = re.sub(r'```json\s*', '', response_text)
    response_text = re.sub(r'```\s*', '', response_text)
    
    # Try to find JSON array pattern
    json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
    else:
        json_str = response_text.strip()
    
    try:
        charts = json.loads(json_str)
        if isinstance(charts, list):
            return charts
        elif isinstance(charts, dict):
            return [charts]
        else:
            raise ValueError("Parsed JSON is neither list nor dict")
    except json.JSONDecodeError as e:
        print(f"⚠️ Failed to parse JSON: {e}")
        print(f"Response text: {response_text[:500]}")
        return []


def format_analysis_summary(analysis_results: Dict[str, Any]) -> str:
    """
    Format analysis results into a concise summary for chart generation.
    
    Args:
        analysis_results: Dictionary of analysis results from enrichment
        
    Returns:
        Formatted string summary
    """
    summary_parts = []
    
    for q_id, results in analysis_results.items():
        q_text = results.get("survey_question", f"Question {q_id}")
        
        # Sentiment distribution
        sentiment_dist = results.get("sentiment_distribution", {})
        if sentiment_dist:
            summary_parts.append(f"**السؤال**: {q_text}")
            summary_parts.append(f"**توزيع المشاعر**: {sentiment_dist}")
        
        # Topic analysis
        topics_analysis = results.get("topics_analysis", "")
        if topics_analysis:
            summary_parts.append(f"**تحليل المواضيع**: {topics_analysis}")
        
        # Entity analysis
        entities_analysis = results.get("entities_analysis", "")
        if entities_analysis:
            summary_parts.append(f"**الكيانات المذكورة**: {entities_analysis}")
        
        # Top topics with counts
        top_topics = results.get("top_topics", [])
        if top_topics:
            # Handle both list and dict formats
            if isinstance(top_topics, dict):
                topics_str = ", ".join([f"{topic} ({count})" for topic, count in list(top_topics.items())[:5]])
            elif isinstance(top_topics, list):
                topics_str = ", ".join([f"{t.get('topic', t.get('label', 'N/A'))} ({t.get('count', 0)})" for t in top_topics[:5]])
            else:
                topics_str = str(top_topics)
            summary_parts.append(f"**أبرز المواضيع**: {topics_str}")
        
        # Top entities with counts
        top_entities = results.get("top_entities", [])
        if top_entities:
            # Handle both list and dict formats
            if isinstance(top_entities, dict):
                entities_str = ", ".join([f"{entity} ({count})" for entity, count in list(top_entities.items())[:5]])
            elif isinstance(top_entities, list):
                entities_str = ", ".join([f"{e.get('entity', e.get('label', 'N/A'))} ({e.get('count', 0)})" for e in top_entities[:5]])
            else:
                entities_str = str(top_entities)
            summary_parts.append(f"**أبرز الكيانات**: {entities_str}")
        
        summary_parts.append("---")
    
    return "\n".join(summary_parts)


# ============================================================================
# CHART GENERATION NODE
# ============================================================================

def generate_charts_agent(state: State) -> Command[Literal["__end__"]]:
    """
    STAGE 4: Generate chart configurations based on analysis results.
    
    Uses an LLM to intelligently create Chart.js configurations that match
    the frontend Vue.js components.
    """
    try:
        print("=" * 60)
        print("CHART GENERATION AGENT")
        print("=" * 60)
        print(f"Survey ID: {state['survey_id']}")
        
        analysis_results = state.get("analysis_results", {})
        
        if not analysis_results:
            print("⚠️ No analysis results available for chart generation")
            return Command(
                update={
                    "chart_configs": [],
                    "messages": [
                        AIMessage(
                            content="No analysis data available to generate charts.",
                            name="chart_generation_agent"
                        )
                    ]
                },
                goto="__end__"
            )
        
        # Extract survey metadata
        first_result = next(iter(analysis_results.values()))
        survey_title = first_result.get("survey_title", f"Survey {state['survey_id']}")
        
        print(f"Survey: {survey_title}")
        print(f"Questions analyzed: {len(analysis_results)}")
        
        # Format analysis summary for LLM
        analytics_summary = format_analysis_summary(analysis_results)
        
        print(f"Analytics summary length: {len(analytics_summary)} chars")
        
        # Generate chart configurations using LLM
        prompt_messages = chart_generation_prompt.invoke({
            "survey_subject": survey_title,
            "analytics_summary": analytics_summary
        })
        
        print("🤖 Invoking LLM for chart generation...")
        response = model.invoke(prompt_messages)
        response_text = response.content
        
        print(f"📥 LLM response length: {len(response_text)} chars")
        
        # Parse JSON response
        chart_configs = extract_json_from_response(response_text)
        
        # Validate chart count (must be 4-6)
        if len(chart_configs) < 4:
            print(f"⚠️ Only {len(chart_configs)} charts generated, expected 4-6. Using what we have.")
        elif len(chart_configs) > 6:
            print(f"⚠️ {len(chart_configs)} charts generated, trimming to 6.")
            chart_configs = chart_configs[:6]
        
        print(f"✅ Generated {len(chart_configs)} chart configurations")
        
        # Log chart types
        for i, chart in enumerate(chart_configs):
            chart_type = chart.get("type", "unknown")
            chart_title = chart.get("title", "Untitled")
            print(f"   {i+1}. {chart_type}: {chart_title}")
        
        return Command(
            update={
                "chart_configs": chart_configs,
                "messages": [
                    AIMessage(
                        content=f"Generated {len(chart_configs)} chart configurations successfully.",
                        name="chart_generation_agent"
                    )
                ]
            },
            goto="__end__"
        )
        
    except Exception as e:
        error_msg = f"Chart generation failed: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        
        return Command(
            update={
                "chart_configs": [],
                "messages": [
                    AIMessage(
                        content=error_msg,
                        name="chart_generation_agent"
                    )
                ]
            },
            goto="__end__"
        )
