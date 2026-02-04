from langgraph.graph import StateGraph, START, END
from agents.graphs.setup import State, memory
from langchain_core.messages import HumanMessage, AIMessage
from helper.tracer import tracer_provider
from agents.graphs.nodes import (
    retrieve_survey_question,
    enrich_data,
    synthesis_agent
)

langfuse = tracer_provider()

def create_survey_insight_workflow(survey_id: int, user_message: str, session_id: str):
    print(f"Creating workflow for survey_id: {survey_id}, session_id: {session_id}")
    
    builder = StateGraph(State)

    builder.add_node("retrieve_survey_question", retrieve_survey_question)
    builder.add_node("enrich_data", enrich_data)
    builder.add_node("synthesis_agent", synthesis_agent)

    builder.add_edge(START, "retrieve_survey_question")
    builder.add_edge("retrieve_survey_question", "enrich_data")
    builder.add_edge("enrich_data", "synthesis_agent")

    graph = builder.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 100}
    if langfuse:
        config["callbacks"] = [langfuse]

    print("Invoking graph...")
    try:
        final_response = graph.invoke(
            {
                "messages": [HumanMessage(content=user_message)],
                "survey_id": survey_id
            },
            config=config
        )
        print("Graph invocation complete")
    except Exception as e:
        print(f"Error during graph invocation: {str(e)}")
        raise

    # Return the last AI message content
    for message in reversed(final_response["messages"]):
        if isinstance(message, AIMessage):
            return message.content

    print("Workflow completed but no AI response was found in messages")
    return "Workflow completed but no response was generated."