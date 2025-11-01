from langchain_community.utilities import SQLDatabase
import urllib
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import Annotated
from typing_extensions import TypedDict
from llms.models import llm
import json

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
    
    prompt_text = f"""
IMPORTANT: You must generate a SINGLE COMPLETE query that retrieves answers for ALL {len(questions)} questions.

All Question IDs that MUST be included:
{', '.join([f"N'{qid}'" for qid in question_ids])}

Questions to analyze:
{formatted_questions}

CRITICAL - ACTUAL DISTINCT ANSWERS FOUND IN DATABASE:
{distinct_answers_text}

⚠️ Use the EXACT answer values shown above in your CASE statements!

REQUIREMENTS:
1. Include ALL {len(questions)} question IDs using WHERE T1.ID IN (...)
2. MUST include AnswerUniqueID to correlate answers
3. Use the EXACT distinct answer values provided above
4. Your query MUST be COMPLETE with both CTE and main SELECT
5. Use N prefix for all string literals (Unicode support)
"""
    
    try:
        prompt = quantitative_prompt_template.invoke(
            {
                "dialect": db.dialect,
                "top_k": 5000,
                "table_info": db.get_table_info(),
                "input": prompt_text,
            }
        )
        structured_llm = llm.with_structured_output(QueryOutput)
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
    
    formatted_input = f"""
Question ID: {question_id}
Question Text: {question_text}

Please generate a query that:
1. Uses the question_id ('{question_id}') for filtering (WHERE T1.ID = N'{question_id}')
2. Returns the question text and all answers in a grouped format
3. Uses STRING_AGG to combine answers into a JSON-like array format
"""
    
    try:
        prompt = qualitative_prompt_template.invoke(
            {
                "dialect": db.dialect,
                "top_k": 5000,
                "table_info": db.get_table_info(),
                "input": formatted_input,
            }
        )
        structured_llm = llm.with_structured_output(QueryOutput)
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

#===========================================================
# For SQL Server (mssql):
# - Use SELECT TOP {top_k} ... instead of LIMIT.

# ask for a few relevant columns given the question or query all all the column if needed.

qualitative_system_message = """
You are an AI assistant that generates syntactically correct {dialect} queries to answer user questions based on database schema information.

## Core Instructions:
- Create a syntactically correct {dialect} query to help find the answer to the input question
- Return only relevant columns needed to answer the question
- **Never query for all columns** from any table - be selective
- **Always exclude the AnswerDate column** from your queries
- **Do not return SQID column** in results
- Unless specified by the user, limit results to at most {top_k} records

## Database-Specific Syntax:
### For SQL Server (mssql):
- Use `SELECT TOP {top_k} ...` instead of `LIMIT`
- Follow SQL Server syntax conventions
- **Use STRING_AGG('"' + STRING_ESCAPE(...)) instead of STRING_AGG(QUOTENAME(...))**
- **For Unicode strings (Arabic, Chinese, etc.): Always prefix string literals with N**
  - Example: `WHERE T1.Question = N'ماهي اكثر شركة'` instead of `WHERE T1.Question = 'ماهي اكثر شركة'`
- **For STRING_ESCAPE with Unicode: Use STRING_ESCAPE(T2.Answer, 'json')**

### For Other Databases:
- Use standard SQL LIMIT clause: `LIMIT {top_k}`
- Handle Unicode strings according to database-specific requirements

## Schema Validation Rules:
- **Only use column names that exist in the provided schema**
- **Carefully verify which columns belong to which tables**
- **Do not query for non-existent columns**
- Cross-reference all column names with the schema before finalizing the query

## Question Filtering Best Practices:
- **ALWAYS prefer filtering by question ID when available** instead of question text
- **Use WHERE T1.ID = 'question_id'** rather than WHERE T1.Question = N'question_text'
- Question IDs are more reliable and avoid Unicode matching issues
- Only use question text in WHERE clause if the question ID is not provided
- **When question_id is provided in the input, you MUST use it for filtering**

## Output Requirements:
- Return the SQL query and the original question
- **Exclude SQID from the output**
- **Structure results with each question appearing once, followed by its answers**
- **Group answers by question** to avoid repetition
- Focus on relevant data that directly answers the user's question

## Available Tables:
{table_info}

## Query Guidelines:
1. Analyze the user's question to identify required information
2. **Check if question_id is provided - if yes, use it for filtering instead of question text**
3. **For survey data: Group answers by question** to create a clean structure where each question appears once
4. **Use aggregation functions** (like STRING_AGG for SQL Server) to combine multiple answers into a single result per question
5. Select only the necessary columns to answer the question
6. Use appropriate JOIN operations when data spans multiple tables
7. Apply WHERE clauses for filtering (prefer ID over text when available)
8. Use ORDER BY for meaningful result ordering when appropriate
9. Ensure the query syntax matches the specified database dialect

## Example Structure for Survey Data:

```sql
-- PREFERRED: Using question ID (when available)
SELECT TOP 1
    T2.Answer AS MostFrequentAnswer,
    COUNT(T2.Answer) AS Count
FROM SurveyQuestion AS T1
JOIN SurveyAnswer AS T2 ON T1.ID = T2.SQID
WHERE T1.ID = 'C878DF73-7D1D-4504-9302-FC7264B10533'
GROUP BY T2.Answer
ORDER BY Count DESC;

-- FALLBACK: Using question text (only when ID not available)
SELECT TOP 1
    T2.Answer AS MostFrequentAnswer,
    COUNT(T2.Answer) AS Count
FROM SurveyQuestion AS T1
JOIN SurveyAnswer AS T2 ON T1.ID = T2.SQID
WHERE T1.Question = N'ماهي اكثر شركة يتم بيع شحن لها في متجرك ؟'
GROUP BY T2.Answer
ORDER BY Count DESC;

-- For grouping multiple answers per question (with ID)
SELECT
    T1.Question,
    '[' + STRING_AGG('"' + STRING_ESCAPE(T2.Answer, 'json') + '"', ',') + ']' AS Answers
FROM SurveyQuestion AS T1
JOIN SurveyAnswer T2 ON T2.SQID = T1.ID
WHERE T1.ID = 'question-id-here'
GROUP BY T1.Question;
```

## Important Reminders:
- **Question ID filtering is the preferred method** - it's more reliable and avoids encoding issues
- Only fall back to question text filtering when the ID is unavailable
- When using text filtering, always use N prefix for Unicode strings
- The question_id field will be provided in the input data when available - always check for it first
"""

user_prompt = "Question: {input}"

qualitative_prompt_template = ChatPromptTemplate(
    [("system", qualitative_system_message), ("user", user_prompt)]
)

# Specific prompt for quantitative questions (multiple questions, correlation analysis)
quantitative_system_message = """
You are an AI assistant that generates syntactically correct {dialect} queries for analyzing multiple survey questions together.

## CRITICAL REQUIREMENTS:
1. **ALWAYS use CTE (WITH clause) with PIVOT structure**
2. **GROUP BY AnswerUniqueID** in the CTE (each row = one respondent)
3. **Use MAX(CASE WHEN T1.ID = N'question-id' THEN T2.Answer END)** for each question
4. **Include ALL question IDs** using WHERE T1.ID IN (...)
5. **Main SELECT must analyze the pivoted data** showing distributions and percentages
6. **Prefix ALL string literals with N** for Unicode support (Arabic text)
7. **Use ONLY the actual distinct answer values** provided in the input - DO NOT GUESS!

## Available Tables:
{table_info}

## MANDATORY Query Structure:

```sql
WITH SurveyPivot AS (
    SELECT 
        T2.AnswerUniqueID,
        MAX(CASE WHEN T1.ID = N'question-id-1' THEN T2.Answer END) AS [Q1_Name],
        MAX(CASE WHEN T1.ID = N'question-id-2' THEN T2.Answer END) AS [Q2_Name]
    FROM 
        SurveyQuestion AS T1
    JOIN 
        SurveyAnswer AS T2 ON T1.ID = T2.SQID
    WHERE 
        T1.ID IN (N'question-id-1', N'question-id-2')
    GROUP BY 
        T2.AnswerUniqueID
)
SELECT 
    COUNT(*) AS Total_Responses,
    COUNT([Q1_Name]) AS Q1_Count,
    COUNT([Q2_Name]) AS Q2_Count,
    
    -- Use EXACT distinct answer values from input, NOT assumptions!
    -- Example: If you see 'موبايلي' in distinct answers, use N'%موبايلي%'
    SUM(CASE WHEN [Q1_Name] LIKE N'%actual-answer-value%' THEN 1 ELSE 0 END) AS Q1_Answer1,
    SUM(CASE WHEN [Q2_Name] = N'actual-answer-value' THEN 1 ELSE 0 END) AS Q2_Answer1,
    
    -- Percentages
    CAST(SUM(CASE WHEN [Q1_Name] LIKE N'%actual-value%' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT([Q1_Name]), 0) AS DECIMAL(5,2)) AS Q1_Percent
FROM SurveyPivot;
```

## SQL Server Syntax:
- Use `SELECT TOP {top_k} ...` (not LIMIT)
- Use `N'text'` for Unicode strings
- Never query AnswerDate or return SQID

## CRITICAL: Answer Values
You will receive actual distinct answer values for each question. Use them EXACTLY in your CASE statements:
- If distinct answers show 'موبайلي', use N'%موبايلي%', NOT 'Mobily'
- If distinct answers show '1 - ممتاز', use N'1%', NOT guesses
- Match the EXACT format provided in the input

Your query MUST:
✓ Have complete CTE with GROUP BY AnswerUniqueID
✓ Have complete main SELECT with aggregations
✓ Include ALL provided question IDs
✓ Use N prefix for all string literals
✓ Use actual distinct answer values from input
"""

quantitative_prompt_template = ChatPromptTemplate(
    [("system", quantitative_system_message), ("user", user_prompt)]
)
