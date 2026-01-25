import json
from langchain_core.documents import Document
from langchain_core.messages import trim_messages
from llms.models import model, google_embeddings_model
from agents.prompt.get_relevant_question_node_prompt import get_relevant_question_prompt
from langgraph.types import Command
from agents.graphs.setup import State
from typing import Literal
from data.operations import get_survey_questions, run_query
from data.sql_operations import get_quantitative_answers, get_qualitative_answers, get_correlation_query
from langchain_core.messages import AIMessage
from agents.prompt.generate_analysis_from_rag_prompt import generate_analysis_from_qualitative_data_prompt
from agents.prompt.generate_analysis_from_sql_prompt import generate_analysis_from_quantitative_data_prompt
from agents.prompt.generate_correlation_analysis_prompt import generate_correlation_analysis_prompt
from agents.prompt.visualization_generator_prompt import visualization_generator_from_quantitative_data_prompt, visualization_generator_from_qualitative_data_prompt
from agents.prompt.visualization_correlation_prompt import visualization_generator_for_correlation_prompt

# Adding layers
# Cleaning layer
# retrive all survey question with answers from DB
def retrieve_survey_question(state: State) -> Command[Literal["__end__"]]:
    try:
        questions = get_survey_questions(state["survey_id"])
        return Command(
            goto="__end__",
            update={
                "questions": questions,
                "messages": [AIMessage(content=f"Survey data retrieved successfully for survey ID {state['survey_id']}. I am ready to analyze it.")]
            }
        )
    except Exception as e:
        raise RuntimeError(f"Server error: {str(e)}")
# add tem in data frame to use them 
# clean each type and normlize if needed 
# you should endup with a clean data frame to use it in the next layer

# Enrichment layer
# adding ML algortihms 
# like extract the entities from the data 
# like determine the sentiment of the data 
# like determine the topic of the data 

# Routing Layer 
# based on user question the AI shoudl retrive the relevent question from the data frame 
# the AI should give you the type of the opration to do 

# Execution layer
# based on the type of the opration the AI should execute the opration 
# like count the number of answers 
# like calculate the average of the answers 
# like calculate the standard deviation of the answers 
# like calculate the correlation between the answers 
# like calculate the correlation between the answers and the question 

# Output layer
# give the AI the analysis to generate the insights for the user 

# def get_relevant_question(state: State) -> Command[Literal["get_answers"]]:
#     try:
#         user_query = state["selected_messages"][-1].content
#         print(f"\n=== GETTING RELEVANT QUESTIONS ===")
#         print(f"User Query: {user_query}")
        
#         # Parse the questions if it's a JSON string
#         if isinstance(state["questions"], str):
#             questions_dict = json.loads(state["questions"])
#         else:
#             questions_dict = state["questions"]
        
#         print(f"Total Questions Available: {len(questions_dict.get('Questions', {}))}")
        
#         # Format the dataset for the prompt
#         dataset_str = json.dumps(questions_dict, ensure_ascii=False)
        
#         prompt = get_relevant_question_prompt.invoke({"dataset": dataset_str, "message": user_query})
#         result = model.invoke(prompt)
#         content = result.content
        
#         json_start = content.find('{')
#         json_end = content.rfind('}') + 1
#         json_str = content[json_start:json_end]
#         data = json.loads(json_str)
        
#         selected_questions = data.get('selected_questions', [])
#         print(f"Selected Questions Count: {len(selected_questions)}")
        
#         # If no questions selected, log the issue
#         if len(selected_questions) == 0:
#             print(f"⚠️ WARNING: No questions selected for query: {user_query}")
#             print(f"Available questions preview: {dataset_str[:300]}...")
        
#         # Classify questions by type
#         qualitative_questions = []
#         quantitative_questions = []
        
#         for question in selected_questions:
#             question_type = question.get('question_type')
#             # Types 1 and 2 are qualitative (open-ended, text input)
#             # Other types (14, etc.) are quantitative/rating
#             if question_type in [1, 2]:
#                 qualitative_questions.append(question)
#             else:
#                 quantitative_questions.append(question)
        
#         print(f"📊 Quantitative Questions: {len(quantitative_questions)}")
#         print(f"📝 Qualitative Questions: {len(qualitative_questions)}")
        
#         return Command(
#             goto="get_answers",
#             update={
#                 "questions": selected_questions,
#                 "qualitative_questions": qualitative_questions,
#                 "quantitative_questions": quantitative_questions,
#                 "quantitative_processed": False,
#                 "next_qualitative_question": 0
#             }
#         )
#     except Exception as e:
#         print(f"❌ ERROR in get_relevant_question: {str(e)}")
#         print(f"User Query: {state['selected_messages'][-1].content if state.get('selected_messages') else 'No messages'}")
#         raise RuntimeError(f"Server error: {str(e)}")

# def get_answers(state: State) -> Command[Literal["generate_analysis_from_sql", "upload_rag", "synthesis_agent"]]:
#     try:
#         quantitative_questions = state.get("quantitative_questions", [])
#         qualitative_questions = state.get("qualitative_questions", [])
#         next_qualitative = state.get("next_qualitative_question", 0)
#         quantitative_processed = state.get("quantitative_processed", False)
#         iteration_count = state.get("iteration_count", 0) + 1
        
#         total_questions = len(quantitative_questions) + len(qualitative_questions)
        
#         print(f"=== GET ANSWERS DEBUG ===")
#         print(f"iteration_count: {iteration_count}")
#         print(f"quantitative questions: {len(quantitative_questions)}, processed: {quantitative_processed}")
#         print(f"qualitative questions: {len(qualitative_questions)}, next: {next_qualitative}")
        
#         # Safety check: prevent infinite loops
#         max_iterations = total_questions * 3
#         if iteration_count > max_iterations:
#             print(f"⚠️ Safety limit reached: {iteration_count} iterations, max is {max_iterations}")
#             print(f"✓ Forcing transition to synthesis_agent")
#             return Command(goto="synthesis_agent")
        
#         # Process ALL quantitative questions in one batch (SQL analysis)
#         if not quantitative_processed and len(quantitative_questions) > 0:
#             print(f"📊 Processing ALL {len(quantitative_questions)} quantitative questions together")
#             for i, q in enumerate(quantitative_questions):
#                 print(f"  {i+1}. {q.get('question_text', 'N/A')[:80]}...")
            
#             # Generate query for ALL quantitative questions at once
#             query = get_quantitative_answers(quantitative_questions)
#             print(f"📝 Generated SQL Query:\n{query}")
#             quantitative_answers = run_query(query)

#             # Only perform correlation analysis if we have at least 2 quantitative questions
#             if len(quantitative_questions) >= 2:
#                 correlation_query = get_correlation_query(quantitative_questions)
#                 print(f"📝 Generated Correlation Query:\n{correlation_query}")
#                 correlation_answers = run_query(correlation_query)
#             else:
#                 print(f"⏭️ Skipping correlation analysis (requires at least 2 quantitative questions, found {len(quantitative_questions)})")
#                 correlation_query = None
#                 correlation_answers = None
            
#             return Command(
#                 goto="generate_analysis_from_sql",
#                 update={
#                     "quantitative_answers": quantitative_answers,
#                     "correlation_answers": correlation_answers,
#                     "previous_query": query,
#                     "correlation_query": correlation_query,
#                     "quantitative_processed": True,
#                     "iteration_count": iteration_count
#                 }
#             )
        
#         # Then process qualitative questions one by one (RAG analysis)
#         elif next_qualitative < len(qualitative_questions):
#             current_question = qualitative_questions[next_qualitative]
#             print(f"📝 Processing qualitative question {next_qualitative + 1}/{len(qualitative_questions)}")
#             print(f"Question: {current_question.get('question_text', 'N/A')[:100]}...")
            
#             query = get_qualitative_answers(current_question)
#             qualitative_answers = run_query(query)
            
#             return Command(
#                 goto="upload_rag",
#                 update={
#                     "qualitative_answers": qualitative_answers,
#                     "next_qualitative_question": next_qualitative + 1,
#                     "iteration_count": iteration_count,
#                     "current_question": current_question
#                 }
#             )
        
#         # All questions processed
#         else:
#             print(f"✓ All questions processed, going to synthesis_agent")
#             return Command(goto="synthesis_agent")
            
#     except Exception as e:
#         print(f"❌ ERROR in get_answers: {str(e)}")
#         raise RuntimeError(f"Server error: {str(e)}")

