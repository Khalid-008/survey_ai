import os
import sys
import json
import re
from typing import Dict, Any, List

# Configure path for imports
current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from langchain_core.messages import AIMessage
from langgraph.types import Command
from typing import Literal

from llms.models import model
from agents.graphs.setup import State
from agents.prompt.chart_generation_prompt import chart_generation_prompt


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
            topics_str = ", ".join([f"{t['topic']} ({t['count']})" for t in top_topics[:5]])
            summary_parts.append(f"**أبرز المواضيع**: {topics_str}")
        
        # Top entities with counts
        top_entities = results.get("top_entities", [])
        if top_entities:
            entities_str = ", ".join([f"{e['entity']} ({e['count']})" for e in top_entities[:5]])
            summary_parts.append(f"**أبرز الكيانات**: {entities_str}")
        
        summary_parts.append("---")
    
    return "\n".join(summary_parts)


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
        import traceback
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


if __name__ == "__main__":
    # Test the chart generation function
    print("Testing chart generation agent...")
    
    test_state = {
        "survey_id": 1,
        "analysis_results": {
            "1": {
                "survey_title": "استطلاع رضا التجار",
                "survey_question": "ما رأيك في الخدمة؟",
                "sentiment_distribution": {
                    "positive": 65.4,
                    "negative": 17.66,
                    "neutral": 16.91
                },
                "topics_analysis": "المواضيع الرئيسية: نقص الشرائح (31%), مشكلات التطبيق (16%), استجابة المندوبين (15%)",
                "entities_analysis": "الشركات المذكورة: زين، STC، موبايلي",
                "top_topics": [
                    {"topic": "نقص الشرائح", "count": 31},
                    {"topic": "مشكلات التطبيق", "count": 16},
                    {"topic": "استجابة المندوبين", "count": 15}
                ],
                "top_entities": [
                    {"entity": "زين", "count": 20},
                    {"entity": "STC", "count": 15}
                ]
            }
        }
    }
    
    result = generate_charts_agent(test_state)
    print("\n" + "=" * 60)
    print("TEST RESULT:")
    print("=" * 60)
    print(f"Chart configs count: {len(result.update.get('chart_configs', []))}")
    print(json.dumps(result.update.get('chart_configs', []), ensure_ascii=False, indent=2))
