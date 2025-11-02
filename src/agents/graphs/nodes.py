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
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from agents.prompt.generate_analysis_from_rag_prompt import generate_analysis_from_qualitative_data_prompt
from agents.prompt.generate_analysis_from_sql_prompt import generate_analysis_from_quantitative_data_prompt
from agents.prompt.generate_correlation_analysis_prompt import generate_correlation_analysis_prompt
from agents.prompt.synthesis_agent_prompt import synthesis_agent_prompt
from agents.prompt.visualization_generator_prompt import visualization_generator_from_quantitative_data_prompt, visualization_generator_from_qualitative_data_prompt
from agents.prompt.visualization_correlation_prompt import visualization_generator_for_correlation_prompt
#######################################################################
embeddings = google_embeddings_model
vector_store = InMemoryVectorStore(embeddings)
#######################################################################

def selected_messages_node(state: State):
    selected_messages = trim_messages(
        state["messages"],
        token_counter=len,
        max_tokens=10,
        strategy="last",
        start_on="human",
        include_system=False,  
        allow_partial=True,
    )
    return {"selected_messages": selected_messages}

def retrieve_survey_question(state: State) -> Command[Literal["get_relevant_question"]]:
    try:
        questions = get_survey_questions(state["survey_id"])
        return Command(
            goto="get_relevant_question",
            update={"questions": questions}
        )
    except Exception as e:
        raise RuntimeError(f"Server error: {str(e)}")

def get_relevant_question(state: State) -> Command[Literal["get_answers"]]:
    try:
        user_query = state["selected_messages"][-1].content
        print(f"\n=== GETTING RELEVANT QUESTIONS ===")
        print(f"User Query: {user_query}")
        
        # Parse the questions if it's a JSON string
        if isinstance(state["questions"], str):
            questions_dict = json.loads(state["questions"])
        else:
            questions_dict = state["questions"]
        
        print(f"Total Questions Available: {len(questions_dict.get('Questions', {}))}")
        
        # Format the dataset for the prompt
        dataset_str = json.dumps(questions_dict, ensure_ascii=False)
        
        prompt = get_relevant_question_prompt.invoke({"dataset": dataset_str, "message": user_query})
        result = model.invoke(prompt)
        content = result.content
        
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        json_str = content[json_start:json_end]
        data = json.loads(json_str)
        
        selected_questions = data.get('selected_questions', [])
        print(f"Selected Questions Count: {len(selected_questions)}")
        
        # If no questions selected, log the issue
        if len(selected_questions) == 0:
            print(f"⚠️ WARNING: No questions selected for query: {user_query}")
            print(f"Available questions preview: {dataset_str[:300]}...")
        
        # Classify questions by type
        qualitative_questions = []
        quantitative_questions = []
        
        for question in selected_questions:
            question_type = question.get('question_type')
            # Types 1 and 2 are qualitative (open-ended, text input)
            # Other types (14, etc.) are quantitative/rating
            if question_type in [1, 2]:
                qualitative_questions.append(question)
            else:
                quantitative_questions.append(question)
        
        print(f"📊 Quantitative Questions: {len(quantitative_questions)}")
        print(f"📝 Qualitative Questions: {len(qualitative_questions)}")
        
        return Command(
            goto="get_answers",
            update={
                "questions": selected_questions,
                "qualitative_questions": qualitative_questions,
                "quantitative_questions": quantitative_questions,
                "quantitative_processed": False,
                "next_qualitative_question": 0
            }
        )
    except Exception as e:
        print(f"❌ ERROR in get_relevant_question: {str(e)}")
        print(f"User Query: {state['selected_messages'][-1].content if state.get('selected_messages') else 'No messages'}")
        raise RuntimeError(f"Server error: {str(e)}")

def get_answers(state: State) -> Command[Literal["generate_analysis_from_sql", "upload_rag", "synthesis_agent"]]:
    try:
        quantitative_questions = state.get("quantitative_questions", [])
        qualitative_questions = state.get("qualitative_questions", [])
        next_qualitative = state.get("next_qualitative_question", 0)
        quantitative_processed = state.get("quantitative_processed", False)
        iteration_count = state.get("iteration_count", 0) + 1
        
        total_questions = len(quantitative_questions) + len(qualitative_questions)
        
        print(f"=== GET ANSWERS DEBUG ===")
        print(f"iteration_count: {iteration_count}")
        print(f"quantitative questions: {len(quantitative_questions)}, processed: {quantitative_processed}")
        print(f"qualitative questions: {len(qualitative_questions)}, next: {next_qualitative}")
        
        # Safety check: prevent infinite loops
        max_iterations = total_questions * 3
        if iteration_count > max_iterations:
            print(f"⚠️ Safety limit reached: {iteration_count} iterations, max is {max_iterations}")
            print(f"✓ Forcing transition to synthesis_agent")
            return Command(goto="synthesis_agent")
        
        # Process ALL quantitative questions in one batch (SQL analysis)
        if not quantitative_processed and len(quantitative_questions) > 0:
            print(f"📊 Processing ALL {len(quantitative_questions)} quantitative questions together")
            for i, q in enumerate(quantitative_questions):
                print(f"  {i+1}. {q.get('question_text', 'N/A')[:80]}...")
            
            # Generate query for ALL quantitative questions at once
            query = get_quantitative_answers(quantitative_questions)
            print(f"📝 Generated SQL Query:\n{query}")
            quantitative_answers = run_query(query)

            # Only perform correlation analysis if we have at least 2 quantitative questions
            if len(quantitative_questions) >= 2:
                correlation_query = get_correlation_query(quantitative_questions)
                print(f"📝 Generated Correlation Query:\n{correlation_query}")
                correlation_answers = run_query(correlation_query)
            else:
                print(f"⏭️ Skipping correlation analysis (requires at least 2 quantitative questions, found {len(quantitative_questions)})")
                correlation_query = None
                correlation_answers = None
            
            return Command(
                goto="generate_analysis_from_sql",
                update={
                    "quantitative_answers": quantitative_answers,
                    "correlation_answers": correlation_answers,
                    "previous_query": query,
                    "correlation_query": correlation_query,
                    "quantitative_processed": True,
                    "iteration_count": iteration_count
                }
            )
        
        # Then process qualitative questions one by one (RAG analysis)
        elif next_qualitative < len(qualitative_questions):
            current_question = qualitative_questions[next_qualitative]
            print(f"📝 Processing qualitative question {next_qualitative + 1}/{len(qualitative_questions)}")
            print(f"Question: {current_question.get('question_text', 'N/A')[:100]}...")
            
            query = get_qualitative_answers(current_question)
            qualitative_answers = run_query(query)
            
            return Command(
                goto="upload_rag",
                update={
                    "qualitative_answers": qualitative_answers,
                    "next_qualitative_question": next_qualitative + 1,
                    "iteration_count": iteration_count,
                    "current_question": current_question
                }
            )
        
        # All questions processed
        else:
            print(f"✓ All questions processed, going to synthesis_agent")
            return Command(goto="synthesis_agent")
            
    except Exception as e:
        print(f"❌ ERROR in get_answers: {str(e)}")
        raise RuntimeError(f"Server error: {str(e)}")

def upload_rag(state: State) -> Command[Literal["generate_analysis_from_rag"]]:
    try:
        vector_store.delete()
        docs = [Document(page_content=str(state["qualitative_answers"]))]  # qualitative answers stored as doc
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=240)
        all_splits = text_splitter.split_documents(docs)
        vector_store.add_documents(all_splits)

        # Convert messages into a single search query string
        search_query = state["selected_messages"][-1].content
        context = retrieve(search_query)

        return Command(
            goto="generate_analysis_from_rag",
            update={"context": context}
        )
    except Exception as e:
        raise RuntimeError(f"Server error: {str(e)}")

def retrieve(search: str):
    try:
        retrieved_docs = vector_store.similarity_search(str(search),20)
        return retrieved_docs
    except Exception as e:
        raise RuntimeError(f"Server error: {str(e)}")
    
def generate_analysis_from_rag(state: State) -> Command[Literal["get_answers"]]:
    try:
        # Combine all documents content
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])

        # Get current question from state
        question = state.get("current_question")
        if not question:
            raise ValueError("current_question not found in state")

        # Invoke prompt with all questions included
        formatted_prompt = generate_analysis_from_qualitative_data_prompt.invoke(
            {
            "DATASET": docs_content, 
            "USER_QUESTION": question['question_text'],
            "QUESTION_ID": question['question_id']
            }
        )
        
        analysis_response = model.invoke(formatted_prompt)

        # Generate charts for this question's analysis
        charts = state.get("charts", [])
        try:
            print(f"🎨 Generating charts for RAG analysis...")
            
            charts_prompt = visualization_generator_from_qualitative_data_prompt.invoke(
                {
                    "ANALYSIS_RESULT": analysis_response.content,
                    "USER_QUESTION": state["selected_messages"][-1].content,
                    "DATA_CONTENT": docs_content
                }
            )
            
            charts_response = model.invoke(charts_prompt)
            
            # Parse and log chart types
            try:
                chart_json = json.loads(charts_response.content)
                if isinstance(chart_json, list) and len(chart_json) > 0:
                    chart_types = [chart.get('chart_type', 'UNKNOWN') for chart in chart_json]
                    print(f"📊 RAG: Generated {len(chart_json)} charts with types: {chart_types}")
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse chart JSON: {e}")
            
            charts.append(charts_response.content)
            
        except Exception as e:
            print(f"⚠️ Chart generation failed: {e}")
        
        return Command(
            update={
                "messages": [
                    AIMessage(content=analysis_response.content, name="generate_analysis_from_rag")
                    ],
                    "charts": charts,
                    "iteration_count": state.get("iteration_count", 0)
                },
                goto="get_answers"
        )

    except Exception as e:
        raise RuntimeError(f"Server error: {str(e)}")

def generate_analysis_from_sql(state: State) -> Command[Literal["get_answers"]]:
    """Analyze all quantitative questions together using the previous query results"""
    try:
        print(f"=== GENERATE ANALYSIS FROM SQL ===")
        
        quantitative_questions = state.get("quantitative_questions", [])
        user_question = state["selected_messages"][-1].content
        previous_query = state.get("previous_query", "")
        quantitative_answers = state.get("quantitative_answers", "")
        correlation_answers = state.get("correlation_answers")
        correlation_query = state.get("correlation_query", "")
        
        print(f"Analyzing {len(quantitative_questions)} quantitative questions")
        
        # Format all question IDs for the prompt
        question_ids = [q.get('question_id') for q in quantitative_questions]
        
        # Create questions summary
        questions_summary = "\n".join([
            f"{i+1}. [ID: {q['question_id']}] {q['question_text']}" 
            for i, q in enumerate(quantitative_questions)
        ])
        
        # Parse quantitative answers
        if isinstance(quantitative_answers, str):
            data = json.loads(quantitative_answers) if quantitative_answers else {}
        else:
            data = quantitative_answers
        
        # Parse correlation answers
        if isinstance(correlation_answers, str):
            correlation_data = json.loads(correlation_answers) if correlation_answers else {}
        else:
            correlation_data = correlation_answers
        
        print(f"📊 Quantitative data size: {len(str(data))}")
        print(f"🔗 Correlation data size: {len(str(correlation_data))}")

        all_analyses = []
        charts = state.get("charts", [])
        
        # 1. Generate regular quantitative analysis
        print(f"📊 Generating regular quantitative analysis...")
        try:
            quant_prompt = generate_analysis_from_quantitative_data_prompt.invoke(
                {
                    "USER_QUESTION": user_question,
                    "QUANTITATIVE_QUESTIONS": questions_summary,
                    "QUESTION_IDS": ', '.join([f"'{qid}'" for qid in question_ids]),
                    "DATASET": json.dumps(data, ensure_ascii=False),
                    "PREVIOUS_QUERY": previous_query,
                }
            )
            
            quant_analysis_response = model.invoke(quant_prompt)
            all_analyses.append(quant_analysis_response.content)
            print(f"✅ Generated quantitative analysis")
            
            # Generate charts for quantitative data
            print(f"🎨 Generating charts for quantitative data...")
            # Format questions text for chart titles
            questions_text_for_charts = "\n".join([
                f"السؤال {i+1}: {q['question_text']}" 
                for i, q in enumerate(quantitative_questions)
            ])
            quant_charts_prompt = visualization_generator_from_quantitative_data_prompt.invoke(
                {
                    "DATA_CONTENT": json.dumps(data, ensure_ascii=False),
                    "QUESTIONS_TEXT": questions_text_for_charts
                }
            )
            
            quant_charts_response = model.invoke(quant_charts_prompt)
            
            try:
                chart_json = json.loads(quant_charts_response.content)
                if isinstance(chart_json, list) and len(chart_json) > 0:
                    chart_types = [chart.get('chart_type', 'UNKNOWN') for chart in chart_json]
                    print(f"📊 Quantitative: Generated {len(chart_json)} charts with types: {chart_types}")
                charts.append(quant_charts_response.content)
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse quantitative chart JSON: {e}")  
        except Exception as e:
            print(f"⚠️ Quantitative analysis failed: {e}")
        
        # 2. Generate correlation analysis (only if we have correlation data)
        has_correlation_data = correlation_data is not None and (
            (isinstance(correlation_data, dict) and len(correlation_data) > 0) or
            (isinstance(correlation_data, list) and len(correlation_data) > 0) or
            (isinstance(correlation_data, str) and correlation_data.strip())
        )
        
        if has_correlation_data:
            print(f"🔗 Generating correlation analysis...")
            try:
                
                corr_prompt = generate_correlation_analysis_prompt.invoke(
                    {
                        "USER_QUESTION": user_question,
                        "QUANTITATIVE_QUESTIONS": questions_summary,
                        "QUESTION_IDS": ', '.join([f"'{qid}'" for qid in question_ids]),
                        "CORRELATION_DATA": json.dumps(correlation_data, ensure_ascii=False),
                        "PREVIOUS_QUERY": correlation_query,
                    }
                )
                
                corr_analysis_response = model.invoke(corr_prompt)
                all_analyses.append(corr_analysis_response.content)
                print(f"✅ Generated correlation analysis")
                
                # Generate charts for correlation data
                print(f"🎨 Generating charts for correlation data...")
                # Format questions text for correlation chart titles
                questions_text_for_corr_charts = "\n".join([
                    f"السؤال {i+1}: {q['question_text']}" 
                    for i, q in enumerate(quantitative_questions)
                ])
                corr_charts_prompt = visualization_generator_for_correlation_prompt.invoke(
                    {
                        "CORRELATION_DATA": json.dumps(correlation_data, ensure_ascii=False),
                        "QUESTIONS_TEXT": questions_text_for_corr_charts
                    }
                )
                
                corr_charts_response = model.invoke(corr_charts_prompt)
                
                try:
                    chart_json = json.loads(corr_charts_response.content)
                    if isinstance(chart_json, list) and len(chart_json) > 0:
                        chart_types = [chart.get('chart_type', 'UNKNOWN') for chart in chart_json]
                        print(f"🔗 Correlation: Generated {len(chart_json)} charts with types: {chart_types}")
                    charts.append(corr_charts_response.content)
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse correlation chart JSON: {e}")
                    
            except Exception as e:
                print(f"⚠️ Correlation analysis failed: {e}")
        
        # Combine all analyses
        combined_analysis = "\n\n---\n\n".join(all_analyses)
        
        return Command(
            update={
                "messages": [
                    AIMessage(content=combined_analysis, name="generate_analysis_from_sql")
                ],
                "charts": charts,
                "iteration_count": state.get("iteration_count", 0)
            },
            goto="get_answers"
        )

    except Exception as e:
        print(f"❌ ERROR in generate_analysis_from_sql: {str(e)}")
        raise RuntimeError(f"Server error: {str(e)}")
    
def synthesis_agent(state: State) -> Command[Literal["__end__"]]:
    try:
        iteration_count = state.get("iteration_count", 0)
        print(f"=== SYNTHESIS AGENT ===")
        print(f"Total iterations: {iteration_count}")
        print(f"Processing final synthesis")

        prompt = synthesis_agent_prompt.invoke(
            {
            "user_question": state["selected_messages"][-1].content,
            "messages": selected_messages_node(state)["selected_messages"]
            }
        )

        response = model.invoke(prompt)

        # Charts are already generated in generate_analysis_from_rag/sql
        # Just return them without modification
        charts = state.get("charts", [])
        
        return Command(
            update={
                "messages": [
                    AIMessage(content=response.content , name="synthesis_agent")
                    ],
                    "charts": charts
                },
                goto="__end__"
        )

    except Exception as e:
        raise RuntimeError(f"Server error: {str(e)}")

