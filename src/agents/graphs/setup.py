from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from typing import Literal
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
from typing import Annotated
from operator import add

class State(MessagesState):
    survey_id: int
    selected_messages: list[str]
    questions : list[str]
    qualitative_questions: list[dict]
    quantitative_questions: list[dict]
    current_question: dict
    quantitative_processed: bool
    next_qualitative_question: int
    previous_query: str
    quantitative_answers: str
    qualitative_answers: str
    correlation_answers: str
    correlation_query: str
    context : list[str]
    data_type: str
    next_question: int
    dataset_sample: str
    charts: list
    iteration_count: int

# Memory
memory = MemorySaver()
