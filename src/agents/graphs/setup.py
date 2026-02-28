from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from typing import Literal, Any
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
from typing import Annotated
from operator import add

class State(MessagesState):
    survey_id: Any
    questions : list[dict]
    survey_data: list[dict]
    text_questions_result: dict[str, Any]       # نتائج تحليل أسئلة النصوص
    selection_questions_result: dict[str, Any]   # نتائج تحليل أسئلة الخيارات
    chart_configs: list[dict]                    # إعدادات الرسوم البيانية المولّدة

# Memory
memory = MemorySaver()
