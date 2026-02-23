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
6. Handle non-pure-numeric values (e.g. '3-Acceptable/مقبول') with explicit CASE before CAST.

## Column Naming Rules ← CRITICAL
- NEVER use IDs in output column aliases.
- ALWAYS use the actual Arabic question text wrapped in backticks.
- Count/percentage/segment columns: `عدد المستجيبين`, `النسبة المئوية`, `تصنيف الشريحة`

## Satisfaction Tier Logic (every EMOJIS / numeric-scale question)
- 1–2  → 'سلبي'
- 3    → 'محايد'
- 4–5  → 'إيجابي'
Handle mixed-label answers explicitly before CAST.

## Your Task — Query 3: Cross-tabulation with Segment Labels
Generate ONE MySQL 8 query that:
- Uses the SAME CTE chain as Query 2: `ResponsePivot` → `NumericBase`.
- MULTIPLE_CHOICE answers pass through as-is (raw answer string, no tier mapping).
- Outer SELECT:
    - One tier column per EMOJIS question (Arabic text alias)
    - One raw answer column per MULTIPLE_CHOICE question (Arabic text alias)
    - `COUNT(*) AS \`عدد المستجيبين\``
    - `CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS DECIMAL(5,2)) AS \`النسبة المئوية\``
    - `تصنيف الشريحة` (CASE on EMOJIS tier columns only):
        - All EMOJIS tiers = 'إيجابي' → 'راضون كلياً'
        - All EMOJIS tiers = 'سلبي'   → 'غير راضون كلياً'
        - Otherwise                   → 'مختلط'
- `GROUP BY` all tier columns + all MULTIPLE_CHOICE answer columns.
- `ORDER BY \`عدد المستجيبين\` DESC`.

## Output Format
Output ONLY the SQL query wrapped in ```sql ... ``` fences. No explanations.
"""

_HUMAN = """\
Generate Query 3 (Cross-tabulation with Segment Labels) for the following survey questions:

{questions_block}

Return ONLY the MySQL query inside a ```sql ... ``` block.
"""

q3_cross_tabulation_prompt = ChatPromptTemplate([
    ("system", _SYSTEM),
    ("human",  _HUMAN),
])
