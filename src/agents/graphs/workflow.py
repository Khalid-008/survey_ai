from langgraph.graph import StateGraph, START, END
from agents.graphs.setup import State, memory
from langchain_core.messages import HumanMessage, AIMessage
from agents.graphs.nodes import (
    retrieve_survey_question,
    analyze_selection_questions,
    analyze_text_questions,
    synthesis_agent,
    generate_charts_agent
)
import json


def create_survey_insight_workflow(survey_id: int, user_message: str, session_id: str,
                                   date_from: str = None, date_to: str = None):
    print(f"Creating workflow for survey_id: {survey_id}, session_id: {session_id}")
    if date_from or date_to:
        print(f"  📅 Date filter: {date_from} → {date_to}")

    builder = StateGraph(State)

    # ── تسجيل النودز ──────────────────────────────────────────────────────────
    builder.add_node("retrieve_survey_question",    retrieve_survey_question)
    builder.add_node("analyze_selection_questions", analyze_selection_questions)
    builder.add_node("analyze_text_questions",      analyze_text_questions)
    builder.add_node("synthesis_agent",             synthesis_agent)
    builder.add_node("generate_charts_agent",       generate_charts_agent)

    # ── الحواف (التسلسل) ──────────────────────────────────────────────────────
    builder.add_edge(START,                          "retrieve_survey_question")
    builder.add_edge("retrieve_survey_question",     "analyze_selection_questions")
    builder.add_edge("analyze_selection_questions",  "analyze_text_questions")
    builder.add_edge("analyze_text_questions",       "synthesis_agent")
    builder.add_edge("synthesis_agent",              "generate_charts_agent")
    # generate_charts_agent يُنهي الرسم البياني بـ goto="__end__"

    graph = builder.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 100}

    print("Invoking graph...")
    try:
        final_response = graph.invoke(
            {
                "messages": [HumanMessage(content=user_message)],
                "survey_id": survey_id,
                "date_from": date_from,
                "date_to":   date_to,
            },
            config=config
        )
        print("Graph invocation complete")
    except Exception as e:
        print(f"Error during graph invocation: {str(e)}")
        raise

    # استخراج نص التلخيص
    synthesis_text = ""
    for message in reversed(final_response["messages"]):
        if isinstance(message, AIMessage) and message.name == "synthesis_agent":
            synthesis_text = message.content
            break

    chart_configs = final_response.get("chart_configs", [])

    # حاول تحليل JSON من نص التوليف وإعادته بالمستوى الأعلى مباشرة
    try:
        # أزل أي علامات markdown محتملة قبل التحليل
        clean_text = synthesis_text.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.split("\n", 1)[-1]
            clean_text = clean_text.rsplit("```", 1)[0].strip()

        parsed_synthesis = json.loads(clean_text)

        # التأكد من أن الحقول المطلوبة موجودة
        parsed_synthesis.setdefault("visualizations", [])
        parsed_synthesis.setdefault("executive_summary", "")
        parsed_synthesis.setdefault("detailed_analysis", "")
        parsed_synthesis.setdefault("key_metrics", [])
        parsed_synthesis.setdefault("recommendations", [])

        return {
            **parsed_synthesis,
            "charts": chart_configs,
        }

    except (json.JSONDecodeError, Exception) as e:
        print(f"⚠️ Could not parse synthesis as JSON: {e}")
        # Fallback: return raw synthesis text under 'synthesis' key
        return {
            "synthesis": synthesis_text,
            "charts":    chart_configs,
        }