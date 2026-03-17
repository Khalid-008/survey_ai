from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from typing import Literal, Any, Optional
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
from typing import Annotated
from operator import add

class State(MessagesState):
    survey_id: Any
    questions : list[dict]
    survey_data: list[dict]
    text_questions_data: list[dict]              # بيانات أسئلة النصوص (TEXT_INPUT) الخام
    selection_questions_data: list[dict]         # بيانات أسئلة الخيارات (غير TEXT_INPUT) الخام
    text_questions_result: dict[str, Any]       # نتائج تحليل أسئلة النصوص
    selection_questions_result: dict[str, Any]   # نتائج تحليل أسئلة الخيارات
    selection_prepared: dict[str, Any]           # مخرجات prepare_selection_data (grouped, distinct, samples, questions_block)
    chart_configs: list[dict]                    # إعدادات الرسوم البيانية المولّدة
    date_from: Optional[str]                     # فلتر التاريخ — بداية (YYYY-MM-DD)
    date_to: Optional[str]                       # فلتر التاريخ — نهاية (YYYY-MM-DD)

# Memory
memory = MemorySaver()
