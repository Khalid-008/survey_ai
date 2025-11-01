"""
Helper functions to build dynamic SQL prompts with runtime data.
These functions construct the user prompts that are passed to the LLM templates.
"""


def build_quantitative_prompt(questions: list[dict], question_ids: list[str], 
                               formatted_questions: str, distinct_answers_text: str) -> str:
    """Build the dynamic prompt for quantitative analysis."""
    return f"""
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


def build_correlation_prompt(questions: list[dict], question_ids: list[str],
                             formatted_questions: str, distinct_answers_text: str) -> str:
    """Build the dynamic prompt for correlation analysis."""
    return f"""
CORRELATION ANALYSIS REQUEST:

Generate a query to analyze how responses to these {len(questions)} questions correlate with each other.
The key field is AnswerUniqueID, which identifies individual respondents across all questions.

All Question IDs for correlation analysis:
{', '.join([f"N'{qid}'" for qid in question_ids])}

Questions to analyze:
{formatted_questions}

CRITICAL - ACTUAL DISTINCT ANSWERS FOUND IN DATABASE:
{distinct_answers_text}

REQUIREMENTS FOR CORRELATION QUERY:
1. Use CTE (WITH clause) to PIVOT data by AnswerUniqueID
2. Each row in the CTE should represent one respondent with their answers to ALL questions
3. For EACH question, capture the answer:
   - MAX(CASE WHEN T1.ID = N'question-id' THEN T2.Answer END) AS Q1_Response
   - MAX(CASE WHEN T1.ID = N'question-id' THEN T2.Answer END) AS Q2_Response
4. Main SELECT should use ACTUAL question text as column aliases:
   - Q1_Response AS [Actual Question 1 Text Here]
   - Q2_Response AS [Actual Question 2 Text Here]
5. Include counts and percentages for each combination:
   - ResponseCount
   - PercentageOfTotal
   - PercentageWithinQ1 (partition by first question)
6. Group by response columns only (not question text since it's just an alias)
7. Use EXACT distinct answer values provided above
8. Use N prefix for all string literals (Unicode support)
9. Order results by ResponseCount DESC (highest correlations first)

Example structure:
```sql
WITH ResponsePivot AS (
    SELECT 
        T2.AnswerUniqueID,
        MAX(CASE WHEN T1.ID = N'qid1' THEN T2.Answer END) AS Q1_Response,
        MAX(CASE WHEN T1.ID = N'qid2' THEN T2.Answer END) AS Q2_Response
    FROM SurveyQuestion AS T1
    JOIN SurveyAnswer AS T2 ON T1.ID = T2.SQID
    WHERE T1.ID IN (N'qid1', N'qid2')
    GROUP BY T2.AnswerUniqueID
)
SELECT 
    Q1_Response AS [Rate your satisfaction with service time?],
    Q2_Response AS [Rate your satisfaction with delivery agent?],
    COUNT(*) AS ResponseCount,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS DECIMAL(5,2)) AS PercentageOfTotal,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(PARTITION BY Q1_Response) AS DECIMAL(5,2)) AS PercentageWithinQ1
FROM ResponsePivot
WHERE Q1_Response IS NOT NULL AND Q2_Response IS NOT NULL
GROUP BY Q1_Response, Q2_Response
ORDER BY ResponseCount DESC;
```

NOTE: Use the ACTUAL question text from the input as column aliases (the text in square brackets).
For example, if Question 1 is "ما مدى رضاك عن وقت التسليم؟", use:
Q1_Response AS [ما مدى رضاك عن وقت التسليم؟]
"""


def build_qualitative_prompt(question_id: str, question_text: str) -> str:
    """Build the dynamic prompt for qualitative analysis."""
    return f"""
Question ID: {question_id}
Question Text: {question_text}

Please generate a query that:
1. Uses the question_id ('{question_id}') for filtering (WHERE T1.ID = N'{question_id}')
2. Returns the question text and all answers in a grouped format
3. Uses STRING_AGG to combine answers into a JSON-like array format
"""

