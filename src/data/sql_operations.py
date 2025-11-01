from langchain_community.utilities import SQLDatabase
import urllib
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import Annotated
from typing_extensions import TypedDict
from llms.models import google_model
import json

# Import prompt templates from their dedicated files
from agents.prompt.sql_qualitative_prompt import qualitative_prompt_template
from agents.prompt.sql_quantitative_prompt import quantitative_prompt_template
from agents.prompt.sql_correlation_prompt import correlation_prompt_template

# Import prompt builder functions
from agents.prompt.sql_prompt_builders import (
    build_quantitative_prompt,
    build_correlation_prompt,
    build_qualitative_prompt
)

# Build a URL-encoded connection string
connection_str = urllib.parse.quote_plus(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=KHALID;"
    "DATABASE=VMS;"
    "Trusted_Connection=yes;"
)



# Use SQLAlchemy-style URI with mssql+pyodbc
db = SQLDatabase.from_uri(f"mssql+pyodbc:///?odbc_connect={connection_str}")

print(db.dialect)


#===========================================================

class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]


def get_quantitative_answers(questions: list[dict]):
    """Generate SQL query to fetch information for ALL quantitative questions together.
    
    Args:
        questions: List of question dictionaries containing question_id, question_text, question_type
        
    Returns:
        SQL query that retrieves answers for ALL questions with AnswerUniqueID for correlation analysis
    """
    # Import here to avoid circular dependency
    from data.operations import get_distinct_answers_for_questions
    
    # Extract all question IDs
    question_ids = [q.get('question_id') for q in questions]
    
    # FIRST: Get distinct answers for all questions
    print(f"📊 Fetching distinct answers for {len(question_ids)} questions...")
    distinct_answers_map = get_distinct_answers_for_questions(question_ids)
    
    # Format distinct answers for the prompt
    distinct_answers_text = "\n\n".join([
        f"Question ID: {qid}\n"
        f"Question Text: {data['question_text']}\n"
        f"Distinct Answers ({len(data['distinct_answers'])}):\n" + 
        "\n".join([f"  - {answer}" for answer in data['distinct_answers'][:20]]) +
        (f"\n  ... and {len(data['distinct_answers']) - 20} more" if len(data['distinct_answers']) > 20 else "")
        for qid, data in distinct_answers_map.items()
    ])
    
    # Format questions for the prompt
    formatted_questions = "\n".join([
        f"Question {i+1}:\n- ID: {q.get('question_id')}\n- Text: {q.get('question_text')}\n- Type: {q.get('question_type')}"
        for i, q in enumerate(questions)
    ])
    
    # Build the dynamic prompt using the builder function
    prompt_text = build_quantitative_prompt(
        questions=questions,
        question_ids=question_ids,
        formatted_questions=formatted_questions,
        distinct_answers_text=distinct_answers_text
    )
    
    try:
        prompt = quantitative_prompt_template.invoke(
            {
                "dialect": db.dialect,
                "top_k": 5000,
                "table_info": db.get_table_info(),
                "input": prompt_text,
            }
        )
        structured_llm = google_model.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        query = result["query"]
        
        # Validate the generated query
        if not query or len(query.strip()) < 50:
            raise ValueError(f"Generated query is too short or empty: {query}")
        
        # Check if CTE query is complete (has both WITH clause and main SELECT)
        query_upper = query.upper().strip()
        if query_upper.startswith("WITH"):
            # Count SELECT statements - should have at least 2 (one in CTE, one main)
            select_count = query_upper.count("SELECT")
            if select_count < 2:
                raise ValueError(f"Incomplete CTE query - found only {select_count} SELECT statement(s)")
            
            # Check that closing parenthesis for CTE exists
            if query.count("(") != query.count(")"):
                raise ValueError(f"Unbalanced parentheses in CTE query")
        
        print(f"✅ Generated valid query ({len(query)} characters)")
        return query
        
    except Exception as e:
        print(f"❌ Error generating CTE query: {str(e)}")
        print(f"🔄 Falling back to simpler query structure...")
        
        # Fallback: Generate a simpler query without CTE
        placeholders = ', '.join([f"N'{qid}'" for qid in question_ids])
        
        # Build a simple aggregation query as fallback
        fallback_query = f"""
        SELECT 
            T1.ID as QuestionID,
            T1.Question as QuestionText,
            T2.Answer,
            T2.AnswerUniqueID,
            COUNT(*) as ResponseCount
        FROM 
            SurveyQuestion AS T1
        JOIN 
            SurveyAnswer AS T2 ON T1.ID = T2.SQID
        WHERE 
            T1.ID IN ({placeholders})
        GROUP BY 
            T1.ID, T1.Question, T2.Answer, T2.AnswerUniqueID
        ORDER BY 
            T1.ID, ResponseCount DESC
        """
        print(f"✅ Using fallback query")
        return fallback_query

def get_correlation_query(questions: list[dict]):
    """Generate SQL query to analyze correlation between quantitative questions based on AnswerUniqueID.
    
    Args:
        questions: List of question dictionaries containing question_id, question_text, question_type
        
    Returns:
        SQL query that analyzes correlation patterns between questions for the same respondents
    """
    # Import here to avoid circular dependency
    from data.operations import get_distinct_answers_for_questions
    
    # Extract all question IDs and texts
    question_ids = [q.get('question_id') for q in questions]
    question_texts = [q.get('question_text') for q in questions]
    
    if len(question_ids) < 2:
        raise ValueError("Correlation analysis requires at least 2 questions")
    
    # Get distinct answers for all questions
    print(f"🔗 Fetching distinct answers for correlation analysis of {len(question_ids)} questions...")
    distinct_answers_map = get_distinct_answers_for_questions(question_ids)
    
    # Format distinct answers for the prompt
    distinct_answers_text = "\n\n".join([
        f"Question ID: {qid}\n"
        f"Question Text: {data['question_text']}\n"
        f"Distinct Answers ({len(data['distinct_answers'])}):\n" + 
        "\n".join([f"  - {answer}" for answer in data['distinct_answers'][:20]]) +
        (f"\n  ... and {len(data['distinct_answers']) - 20} more" if len(data['distinct_answers']) > 20 else "")
        for qid, data in distinct_answers_map.items()
    ])
    
    # Format questions for the prompt
    formatted_questions = "\n".join([
        f"Question {i+1}:\n- ID: {q.get('question_id')}\n- Text: {q.get('question_text')}\n- Type: {q.get('question_type')}"
        for i, q in enumerate(questions)
    ])
    
    # Build the dynamic prompt using the builder function
    prompt_text = build_correlation_prompt(
        questions=questions,
        question_ids=question_ids,
        formatted_questions=formatted_questions,
        distinct_answers_text=distinct_answers_text
    )
    
    try:
        prompt = correlation_prompt_template.invoke(
            {
                "dialect": db.dialect,
                "top_k": 5000,
                "table_info": db.get_table_info(),
                "input": prompt_text,
            }
        )
        structured_llm = google_model.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        query = result["query"]
        
        # Validate the generated query
        if not query or len(query.strip()) < 50:
            raise ValueError(f"Generated correlation query is too short or empty: {query}")
        
        # Check if CTE query is complete
        query_upper = query.upper().strip()
        if query_upper.startswith("WITH"):
            select_count = query_upper.count("SELECT")
            if select_count < 2:
                raise ValueError(f"Incomplete CTE query - found only {select_count} SELECT statement(s)")
            
            if query.count("(") != query.count(")"):
                raise ValueError(f"Unbalanced parentheses in CTE query")
        
        print(f"✅ Generated valid correlation query ({len(query)} characters)")
        return query
        
    except Exception as e:
        print(f"❌ Error generating correlation query: {str(e)}")
        print(f"🔄 Falling back to simpler correlation query structure...")
        
        # Fallback: Generate a simpler correlation query with actual question text as column aliases
        placeholders = ', '.join([f"N'{qid}'" for qid in question_ids])
        
        # Build column definitions for PIVOT (only responses, no separate question text columns)
        pivot_parts = []
        for i, qid in enumerate(question_ids):
            pivot_parts.append(f"MAX(CASE WHEN T1.ID = N'{qid}' THEN T2.Answer END) AS Q{i+1}_Response")
        pivot_columns = ',\n        '.join(pivot_parts)
        
        # Build SELECT columns using actual question text as aliases
        select_columns = ',\n    '.join([
            f"Q{i+1}_Response AS [{question_texts[i]}]"
            for i in range(len(question_ids))
        ])
        
        # Build GROUP BY clause (only response columns)
        group_columns = ', '.join([f"Q{i+1}_Response" for i in range(len(question_ids))])
        
        # Build WHERE clause
        where_conditions = ' AND '.join([f"Q{i+1}_Response IS NOT NULL" for i in range(len(question_ids))])
        
        # Add percentage calculations based on first question
        fallback_query = f"""
        WITH ResponsePivot AS (
            SELECT 
                T2.AnswerUniqueID,
                {pivot_columns}
            FROM 
                SurveyQuestion AS T1
            JOIN 
                SurveyAnswer AS T2 ON T1.ID = T2.SQID
            WHERE 
                T1.ID IN ({placeholders})
            GROUP BY 
                T2.AnswerUniqueID
        )
        SELECT 
            {select_columns},
            COUNT(*) AS ResponseCount,
            CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS DECIMAL(5,2)) AS PercentageOfTotal,
            CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(PARTITION BY Q1_Response) AS DECIMAL(5,2)) AS PercentageWithinQ1
        FROM 
            ResponsePivot
        WHERE 
            {where_conditions}
        GROUP BY 
            {group_columns}
        ORDER BY 
            ResponseCount DESC
        """
        print(f"✅ Using fallback correlation query")
        return fallback_query

def get_qualitative_answers(question: dict):
    """Generate SQL query to fetch information for a qualitative question.
    
    Args:
        question: Dictionary containing question_id, question_text, question_type
        
    Returns:
        SQL query string to retrieve answers for the specified question
    """
    # Format the question properly for the LLM
    # The prompt prefers question_id over question_text (more reliable, avoids Unicode issues)
    question_id = question.get('question_id')
    question_text = question.get('question_text', 'N/A')
    
    # Build the dynamic prompt using the builder function
    formatted_input = build_qualitative_prompt(
        question_id=question_id,
        question_text=question_text
    )
    
    try:
        prompt = qualitative_prompt_template.invoke(
            {
                "dialect": db.dialect,
                "top_k": 5000,
                "table_info": db.get_table_info(),
                "input": formatted_input,
            }
        )
        structured_llm = google_model.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        query = result["query"]
        
        # Validate the generated query
        if not query or len(query.strip()) < 30:
            raise ValueError(f"Generated query is too short or empty: {query}")
        
        # Basic validation for unclosed quotes
        single_quotes = query.count("'")
        if single_quotes % 2 != 0:
            print(f"⚠️ WARNING: Query may have unclosed quotes: {query[:200]}...")
            raise ValueError("Query has unclosed quotes")
        
        print(f"✅ Generated qualitative query ({len(query)} characters)")
        print(f"Query preview: {query[:150]}...")
        
        return query
        
    except Exception as e:
        print(f"❌ Error generating qualitative query: {str(e)}")
        print(f"🔄 Using fallback query for question_id: {question_id}")
        
        # Fallback: Generate a simple, reliable query
        fallback_query = f"""
        SELECT TOP 5000
            T1.Question,
            '[' + STRING_AGG('"' + STRING_ESCAPE(T2.Answer, 'json') + '"', ',') + ']' AS Answers
        FROM SurveyQuestion AS T1
        JOIN SurveyAnswer AS T2 ON T1.ID = T2.SQID
        WHERE T1.ID = N'{question_id}'
        GROUP BY T1.Question
        """
        print(f"✅ Using fallback query")
        return fallback_query

