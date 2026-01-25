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
    questions : list[str]

# Memory
memory = MemorySaver()
