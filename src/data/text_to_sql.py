from langchain_community.utilities import SQLDatabase
import urllib
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import Annotated
from typing_extensions import TypedDict
from llms.models import llm

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
    # Extract all question IDs
    question_ids = [q.get('question_id') for q in questions]
    
    # Format questions for the prompt
    formatted_questions = "\n".join([
        f"Question {i+1}:\n- ID: {q.get('question_id')}\n- Text: {q.get('question_text')}\n- Type: {q.get('question_type')}"
        for i, q in enumerate(questions)
    ])
    
    prompt_text = f"""
IMPORTANT: You must generate a SINGLE query that retrieves answers for ALL {len(questions)} questions listed below.

All Question IDs that MUST be included in your query:
{', '.join([f"'{qid}'" for qid in question_ids])}

Questions to analyze:
{formatted_questions}

CRITICAL REQUIREMENTS:
1. Your query MUST include ALL {len(questions)} question IDs using IN clause or UNION
2. MUST include AnswerUniqueID to correlate answers across questions
3. Structure the result so we can see how each user (AnswerUniqueID) answered each question
4. This enables finding relationships between user answers across different questions

Example approach for multiple questions:
- Use WHERE T1.ID IN ('question-id-1', 'question-id-2', ...)
- Group by AnswerUniqueID to see each user's responses
- Include question text or ID to identify which question each answer belongs to
"""
    
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
    return result["query"]

def get_qualitative_answers(questions: list[str]):
    """Generate SQL query to fetch information."""
    prompt = qualitative_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 5000,
            "table_info": db.get_table_info(),
            "input": questions,
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return result["query"]

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

## CRITICAL: Multiple Questions Handling - PIVOT QUERY REQUIRED
When multiple questions are provided, you MUST:
1. **USE A CTE (WITH clause) with PIVOT structure** - This is MANDATORY
2. **GROUP BY AnswerUniqueID** in the CTE - Each row = one respondent
3. **Use MAX(CASE WHEN T1.ID = 'question-id' THEN T2.Answer END)** for each question
4. **Include ALL question IDs** using WHERE T1.ID IN ('id1', 'id2', 'id3', ...)
5. **Then analyze the pivoted data** in the main SELECT to show distributions and correlations

**WHY PIVOT STRUCTURE IS MANDATORY:**
- Enables correlation analysis (see how same users answered different questions)
- Perfect for generating insights and reports
- Makes it easy to calculate distributions and percentages
- Allows finding patterns between questions

## Core Instructions:
- Create ONE query that retrieves answers for ALL questions provided
- **ALWAYS include AnswerUniqueID** to track individual user responses
- Return columns: Question (or QuestionID), AnswerUniqueID, Answer (and any aggregate info)
- Enable correlation analysis by grouping with AnswerUniqueID
- Never query for all columns - be selective
- Always exclude AnswerDate column
- Do not return SQID column
- Use SELECT TOP {top_k} for SQL Server

## Database-Specific Syntax:
### For SQL Server (mssql):
- Use `SELECT TOP {top_k} ...` instead of `LIMIT`
- Follow SQL Server syntax conventions
- Use STRING_AGG for concatenation if needed
- For Unicode strings: Always prefix string literals with N
- For STRING_ESCAPE with Unicode: Use STRING_ESCAPE(T2.Answer, 'json')

## Available Tables:
{table_info}

## REQUIRED Query Structure for Multiple Quantitative Questions:

**YOU MUST USE THIS PIVOT-STYLE CTE APPROACH:**

This structure is MANDATORY because it:
- Groups by AnswerUniqueID (each row = one respondent)
- Enables correlation analysis between questions
- Makes it easy to find patterns in how users answered different questions
- Perfect for generating insights and reports

### REQUIRED Structure:

```sql
WITH SurveyPivot AS (
    SELECT 
        T2.AnswerUniqueID,
        MAX(CASE WHEN T1.ID = N'question-id-1' THEN T2.Answer END) AS [Question1_Name],
        MAX(CASE WHEN T1.ID = N'question-id-2' THEN T2.Answer END) AS [Question2_Name],
        MAX(CASE WHEN T1.ID = N'question-id-3' THEN T2.Answer END) AS [Question3_Name]
    FROM 
        SurveyQuestion AS T1
    JOIN 
        SurveyAnswer AS T2 ON T1.ID = T2.SQID
    WHERE 
        T1.ID IN (N'question-id-1', N'question-id-2', N'question-id-3')
    GROUP BY 
        T2.AnswerUniqueID
)
-- Now analyze the pivoted results
SELECT 
    COUNT(*) AS [Total_Responses],
    
    -- Count responses for each question
    COUNT([Question1_Name]) AS [Question1_Count],
    COUNT([Question2_Name]) AS [Question2_Count],
    
    -- Breakdown by answer categories
    SUM(CASE WHEN [Question1_Name] LIKE N'1%' THEN 1 ELSE 0 END) AS [Q1_Category1],
    SUM(CASE WHEN [Question1_Name] LIKE N'2%' THEN 1 ELSE 0 END) AS [Q1_Category2],
    
    SUM(CASE WHEN [Question2_Name] LIKE N'1%' THEN 1 ELSE 0 END) AS [Q2_Category1],
    SUM(CASE WHEN [Question2_Name] LIKE N'2%' THEN 1 ELSE 0 END) AS [Q2_Category2],
    
    -- Percentages
    CAST(SUM(CASE WHEN [Question1_Name] LIKE N'1%' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT([Question1_Name]), 0) AS DECIMAL(5,2)) AS [Q1_Category1_Percent],
    CAST(SUM(CASE WHEN [Question2_Name] LIKE N'1%' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT([Question2_Name]), 0) AS DECIMAL(5,2)) AS [Q2_Category1_Percent]
FROM SurveyPivot;
```

### Example with Real Question IDs:

```sql
WITH SurveyPivot AS (
    SELECT 
        T2.AnswerUniqueID,
        MAX(CASE WHEN T1.ID = N'90973574-FA33-479B-A74F-DE3C31AE1A0E' 
                 THEN T2.Answer END) AS [ServiceTime],
        MAX(CASE WHEN T1.ID = N'3B6F2CE9-4C1E-4A26-914A-944140610DF9' 
                 THEN T2.Answer END) AS [DeliveryAgent]
    FROM 
        SurveyQuestion AS T1
    JOIN 
        SurveyAnswer AS T2 ON T1.ID = T2.SQID
    WHERE 
        T1.ID IN (
            N'90973574-FA33-479B-A74F-DE3C31AE1A0E',
            N'3B6F2CE9-4C1E-4A26-914A-944140610DF9'
        )
    GROUP BY 
        T2.AnswerUniqueID
)
-- Analyze correlations and distributions
SELECT 
    COUNT(*) AS [Total_Responses],
    COUNT(ServiceTime) AS [ServiceTime_Responses],
    COUNT(DeliveryAgent) AS [DeliveryAgent_Responses],
    
    -- Distribution for Service Time
    SUM(CASE WHEN ServiceTime LIKE N'1%' THEN 1 ELSE 0 END) AS [Excellent_ServiceTime],
    SUM(CASE WHEN ServiceTime LIKE N'2%' THEN 1 ELSE 0 END) AS [Good_ServiceTime],
    SUM(CASE WHEN ServiceTime LIKE N'3%' THEN 1 ELSE 0 END) AS [Average_ServiceTime],
    SUM(CASE WHEN ServiceTime LIKE N'4%' THEN 1 ELSE 0 END) AS [Poor_ServiceTime],
    
    -- Distribution for Delivery Agent
    SUM(CASE WHEN DeliveryAgent LIKE N'1%' THEN 1 ELSE 0 END) AS [Excellent_Agent],
    SUM(CASE WHEN DeliveryAgent LIKE N'2%' THEN 1 ELSE 0 END) AS [Good_Agent],
    SUM(CASE WHEN DeliveryAgent LIKE N'3%' THEN 1 ELSE 0 END) AS [Average_Agent],
    SUM(CASE WHEN DeliveryAgent LIKE N'4%' THEN 1 ELSE 0 END) AS [Poor_Agent],
    
    -- Percentages
    CAST(SUM(CASE WHEN ServiceTime LIKE N'1%' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(ServiceTime), 0) AS DECIMAL(5,2)) AS [Excellent_ServiceTime_Percent],
    CAST(SUM(CASE WHEN DeliveryAgent LIKE N'1%' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(DeliveryAgent), 0) AS DECIMAL(5,2)) AS [Excellent_Agent_Percent]
FROM SurveyPivot;
```

## CRITICAL Reminders:
- **MUST USE CTE WITH PIVOT STRUCTURE** - This is the required format
- **MUST GROUP BY AnswerUniqueID** in the CTE
- **MUST include ALL question IDs** provided in the input using IN clause
- **Use MAX(CASE WHEN T1.ID = N'question-id' THEN T2.Answer END)** for each question
- **Always prefix Unicode strings with N** (e.g., N'question-id', N'1%')
- **Analyze the pivoted data** to show distributions, percentages, and correlations
- **The goal:** See patterns in how users answered different questions together
"""

quantitative_prompt_template = ChatPromptTemplate(
    [("system", quantitative_system_message), ("user", user_prompt)]
)
