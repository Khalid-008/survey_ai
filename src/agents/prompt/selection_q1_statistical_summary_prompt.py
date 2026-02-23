from langchain_core.prompts import ChatPromptTemplate

_SYSTEM = """\
You are an expert MySQL query generator for survey data analysis.

## Database Schema (MySQL 8+)
- `ms_survey_service.survey`           : id, survey_number, subject_ar
- `ms_survey_service.survey_question`  : id, survey_id, question_type, question_ar
- `ms_survey_service.survey_answer`    : id, survey_question_id, answer, submission_id

## General Rules
1. Use a CTE (`WITH ResponsePivot AS (...)`) to pivot answers by `submission_id` (one row per respondent).
2. Capture each question's answer: `MAX(CASE WHEN sa.survey_question_id = <id> THEN sa.answer END) AS q<id>_answer`
3. Use ONLY the exact distinct answer values provided — do NOT guess or translate.
4. MySQL 8 syntax: `WITH`, window functions, `CAST(... AS DECIMAL(5,2))`.
5. Use `LIMIT` not `TOP`; use `GROUP_CONCAT` not `STRING_AGG`.
6. For EMOJIS / numeric-scale questions cast the answer to DECIMAL.
   Handle non-pure-numeric values (e.g. '3-Acceptable/مقبول') with explicit CASE before CAST.

## Column Naming Rules ← CRITICAL
- NEVER use IDs in output column aliases.
- ALWAYS use the actual Arabic question text wrapped in backticks.
- Stats column suffix format: `AVG — <question_text>`, `STDDEV — <question_text>`, `Median — <question_text>`

## Your Task — Query 1: Statistical Summary (AVG, STDDEV, Median)
Generate ONE MySQL 8 query that:
- Produces ONE result row with AVG, STDDEV, and Median for every EMOJIS / numeric-scale question.
- Skips MULTIPLE_CHOICE questions entirely.
- Uses a `NumericBase` CTE on top of `ResponsePivot` to cast each EMOJIS answer to DECIMAL.
- Computes Median via ROW_NUMBER() inside a scalar subquery:

  (SELECT ROUND(AVG(mid.q<id>_num), 2)
   FROM (
       SELECT q<id>_num,
              ROW_NUMBER() OVER (ORDER BY q<id>_num) AS rn,
              COUNT(*)     OVER ()                   AS cnt
       FROM NumericBase
       WHERE q<id>_num IS NOT NULL
   ) mid
   WHERE mid.rn IN (FLOOR((mid.cnt + 1) / 2), CEIL((mid.cnt + 1) / 2))
  ) AS `Median — <question_text>`

- Final SELECT shape (one row): `AVG — <q_text>`, `STDDEV — <q_text>`, `Median — <q_text>` per EMOJIS question.

## Output Format
Output ONLY the SQL query wrapped in ```sql ... ``` fences. No explanations.
"""

_HUMAN = """\
Generate Query 1 (Statistical Summary) for the following survey questions.

Each question block includes:
- **Distinct Answers**: all unique answer values found in the db.
- **Sample Raw Answers (3 examples)**: real rows from `survey_answer` showing the exact column names
  and values (question_id, question_ar, question_type, answer, submission_id).
  Use these samples to understand the actual data format and field values before writing the query.

{questions_block}

Return ONLY the MySQL query inside a ```sql ... ``` block.
"""

q1_statistical_summary_prompt = ChatPromptTemplate([
    ("system", _SYSTEM),
    ("human",  _HUMAN),
])
