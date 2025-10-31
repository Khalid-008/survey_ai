from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.redis import RedisSaver
import os
from agents.graphs.setup import State
from helper.utils import remove_think_blocks
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage,AIMessage
from helper.tracer import tracer_provider
from agents.graphs.nodes import (
    selected_messages_node,
    retrieve_survey_question,
    get_relevant_question,
    get_answers,
    generate_analysis_from_sql,
    upload_rag,
    generate_analysis_from_rag,
    synthesis_agent
)

langfuse = tracer_provider()

def create_survey_insight_workflow(survey_id: int, user_message: str, session_id: str):
    REDIS_URI = os.getenv("REDIS_URI")
    with (
        RedisSaver.from_conn_string(REDIS_URI) as checkpointer
    ):
        checkpointer.setup()
        builder = StateGraph(State)

        builder.add_node("selected_messages_node", selected_messages_node)
        builder.add_node("retrieve_survey_question", retrieve_survey_question)
        builder.add_node("get_relevant_question", get_relevant_question)
        builder.add_node("get_answers", get_answers)
        builder.add_node("generate_analysis_from_sql", generate_analysis_from_sql)
        builder.add_node("upload_rag", upload_rag)
        builder.add_node("generate_analysis_from_rag", generate_analysis_from_rag)
        builder.add_node("synthesis_agent", synthesis_agent)

        # 1
        # retrieve survey's questions WO LLM  DONE
        # get the most relevant question to the user question W LLM
        # retrieve question's answers and put them in RAG system W LLM
        # ask the RAG to retrieve the answer based on user question W LLM
        # get the answer WO LLM
        # 2
        # retrieve survey's questions WO LLM 
        # get the most relevant question to the user question W LLM
        # retrieve question's answers WO LLM
        # share the answer to more than one agent to balance the context model W LLM
        # get the answers from the agent and feed them to the main agent W LLM 
        # ask the main agent user question W LLM
        # get the answer WO LLM


        builder.add_edge(START, "selected_messages_node")
        builder.add_edge("selected_messages_node", "retrieve_survey_question")

        graph = builder.compile(checkpointer=checkpointer)

        final_response = graph.invoke(
            {
                "messages": [HumanMessage(content=user_message)],
                "survey_id": survey_id,
                "next_question": 0,
                "charts": [],
                "iteration_count": 0
            },
            config={"configurable": {"thread_id": session_id}, "callbacks": [langfuse], "recursion_limit": 100}
        )

    # Return the last AI message content
    for message in reversed(final_response["messages"]):
        if isinstance(message, AIMessage):
            # Charts are now stored as list of JSON objects
            charts_content = final_response.get("charts", [])
            
            return {
                "content": message.content,
                "charts": charts_content
            }
                
