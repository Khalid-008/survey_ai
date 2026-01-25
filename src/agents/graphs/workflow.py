from langgraph.graph import StateGraph, START, END
from agents.graphs.setup import State, memory
from helper.utils import remove_think_blocks
from langchain_core.messages import HumanMessage, AIMessage
from helper.tracer import tracer_provider
from agents.graphs.nodes import (
    retrieve_survey_question
)

langfuse = tracer_provider()

def create_survey_insight_workflow(survey_id: int, user_message: str, session_id: str):
    builder = StateGraph(State)

    builder.add_node("retrieve_survey_question", retrieve_survey_question)
    # builder.add_node("get_relevant_question", get_relevant_question)
    # builder.add_node("get_answers", get_answers)

    builder.add_edge(START, "retrieve_survey_question")
    builder.add_edge("retrieve_survey_question", END)

    graph = builder.compile(checkpointer=memory)

    final_response = graph.invoke(
        {
            "messages": [HumanMessage(content=user_message)],
            "survey_id": survey_id
        },
        config={"configurable": {"thread_id": session_id}, "callbacks": [langfuse], "recursion_limit": 100}
    )

    # Return the last AI message content
    for message in reversed(final_response["messages"]):
        if isinstance(message, AIMessage):
            return message.content

    return "Workflow completed but no response was generated."