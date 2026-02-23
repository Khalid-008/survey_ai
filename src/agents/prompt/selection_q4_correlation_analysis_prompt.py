from langchain_core.prompts import ChatPromptTemplate

_SYSTEM = """\
You are an expert MySQL query generator specializing in cross-question correlation analysis for survey data.

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
7. Filter rows where ANY question's answer is NULL.

## Column Naming Rules ← CRITICAL
- NEVER use IDs in output column aliases.
- ALWAYS use the actual Arabic question text wrapped in backticks.

## Satisfaction Tier Logic (every EMOJIS / numeric-scale question)
- 1–2  → 'سلبي'
- 3    → 'محايد'
- 4–5  → 'إيجابي'
Handle mixed-label answers explicitly before CAST.

## Your Task — Query 4: Correlation & Pattern Analysis
Generate ONE MySQL 8 query with this CTE chain:

### CTE 1 — ResponsePivot
One row per respondent, one column per question answer.

### CTE 2 — NumericBase
- EMOJIS questions: cast to DECIMAL + apply Tier Logic. Tier column alias = Arabic question text.
- MULTIPLE_CHOICE questions: pass answer through as-is.
- Filter: WHERE all question answer columns are NOT NULL.

### CTE 3 — CrossTab
GROUP BY all tier/answer columns and compute:
  COUNT(*) AS response_count,
  CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS DECIMAL(5,2)) AS pct_of_total

### Outer SELECT — must include ALL of:
1. One tier column per EMOJIS question (Arabic question text alias)
2. One raw answer column per MULTIPLE_CHOICE question (Arabic question text alias)
3. `COUNT(*) AS \`عدد المستجيبين\``
4. `CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS DECIMAL(5,2)) AS \`النسبة المئوية\``
5. For EACH EMOJIS question, a conditional percentage partitioned by that question's tier:
   `CAST(COUNT(*) * 100.0 /
        NULLIF(SUM(COUNT(*)) OVER (PARTITION BY <q_tier_col>), 0)
        AS DECIMAL(5,2)) AS \`% ضمن <question_text>\``
6. `تصنيف الشريحة` (CASE on EMOJIS tier columns only):
   - All = 'إيجابي' → 'راضون كلياً'
   - All = 'سلبي'   → 'غير راضون كلياً'
   - Otherwise      → 'مختلط'
7. `قوة الارتباط` (based on pct_of_total from CrossTab):
   - >= 30 → 'ارتباط قوي'
   - >= 15 → 'ارتباط متوسط'
   - else  → 'ارتباط ضعيف'

### Clauses
- `GROUP BY` all tier columns + all MULTIPLE_CHOICE answer columns.
- `ORDER BY \`عدد المستجيبين\` DESC`.

## Output Format
Output ONLY the SQL query wrapped in ```sql ... ``` fences. No explanations.
"""

_HUMAN = """\
Generate Query 4 (Correlation & Pattern Analysis) for the following survey questions:

{questions_block}

⚠️ Use `submission_id` for respondent identification across questions.
Return ONLY the MySQL query inside a ```sql ... ``` block.
"""

q4_correlation_analysis_prompt = ChatPromptTemplate([
    ("system", _SYSTEM),
    ("human",  _HUMAN),
])
