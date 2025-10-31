from langgraph.prebuilt import create_react_agent
from llms.models import model
from tools.sql_agent_tools import get_survey_questions
from agents.prompt.get_relevant_question_node_prompt import get_relevant_question_prompt
from agents.prompt.data_type_agent_prompt import data_type_agent_prompt

# sql_agent = create_react_agent(
#     model=model,
#     name="sql_agent",
#     tools = [ get_survey_questions],
#     prompt = get_relevant_question_prompt.invoke({
#         "question": "",
#     }).to_string()
# )

# def data_type_agent(INSERT_DATASET_SAMPLE_HERE: str, INSERT_USER_QUESTION_HERE: str):
#     response = create_react_agent(
#         model=model,
#         name="data_type_agent",
#         prompt = data_type_agent_prompt.invoke({
#             "INSERT_DATASET_SAMPLE_HERE": INSERT_DATASET_SAMPLE_HERE,
#             "INSERT_USER_QUESTION_HERE": INSERT_USER_QUESTION_HERE,
#         }).to_string()
#     )
#     return response

