import os
import sys
import re
import json
import warnings
from collections import defaultdict
from datetime import datetime
from typing import Any

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

# ─────────────────────────── path setup ───────────────────────────────────────
current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(current_file_path))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from llms.models import model
from data.operations import get_selection_questions_data, execute_raw_query

warnings.filterwarnings("ignore")

EXPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "exports", "selection_results"
)
os.makedirs(EXPORTS_DIR, exist_ok=True)


# ============================================================================
# FILE EXPORT
# حفظ النتائج في ملف للمراجعة
# ============================================================================

def _save_results_to_file(
    survey_number: str,
    grouped: dict,
    distinct: dict,
    selection_sql: str,
    selection_result: pd.DataFrame,
    correlation_sql: str,
    correlation_result: pd.DataFrame,
    errors: list
) -> str:
    """
    يحفظ الكويريات والنتائج في ملف نصي + CSV للمراجعة.
    يُرجع مسار الملف الرئيسي.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{survey_number}_{ts}"

    # ── ملف النص الرئيسي ─────────────────────────────────────────────────────
    txt_path = os.path.join(EXPORTS_DIR, f"{base_name}_selection_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"SELECTION PIPELINE REPORT\n")
        f.write(f"Survey   : {survey_number}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        # معلومات الأسئلة والإجابات المميزة
        f.write("── QUESTIONS & DISTINCT ANSWERS ──\n")
        for qid, info in grouped.items():
            f.write(f"\nQ{qid}: {info['text']}\n")
            f.write(f"  Type   : {info['type']}\n")
            f.write(f"  Distinct answers ({len(distinct.get(qid, []))}):\n")
            for ans in distinct.get(qid, []):
                f.write(f"    • {ans}\n")

        # كويري التحليل الفردي
        f.write("\n\n── SELECTION ANALYSIS SQL ──\n")
        f.write(selection_sql if selection_sql else "(not generated)")
        f.write("\n\n── SELECTION ANALYSIS RESULT ──\n")
        if not selection_result.empty:
            f.write(selection_result.to_string(index=False))
            f.write(f"\n\n({len(selection_result)} rows)\n")
        else:
            f.write("(empty)\n")

        # كويري العلاقة
        f.write("\n\n── CORRELATION SQL ──\n")
        f.write(correlation_sql if correlation_sql else "(not generated)")
        f.write("\n\n── CORRELATION RESULT ──\n")
        if not correlation_result.empty:
            f.write(correlation_result.to_string(index=False))
            f.write(f"\n\n({len(correlation_result)} rows)\n")
        else:
            f.write("(empty)\n")

        # الأخطاء
        if errors:
            f.write("\n\n── ERRORS ──\n")
            for err in errors:
                f.write(f"  • {err}\n")

    # ── ملفات CSV (إن وُجدت نتائج) ───────────────────────────────────────────
    if not selection_result.empty:
        csv_sel = os.path.join(EXPORTS_DIR, f"{base_name}_selection_result.csv")
        selection_result.to_csv(csv_sel, index=False, encoding="utf-8-sig")
        print(f"📄 Selection CSV  : {csv_sel}")

    if not correlation_result.empty:
        csv_cor = os.path.join(EXPORTS_DIR, f"{base_name}_correlation_result.csv")
        correlation_result.to_csv(csv_cor, index=False, encoding="utf-8-sig")
        print(f"📄 Correlation CSV: {csv_cor}")

    print(f"📄 Report TXT     : {txt_path}")
    return txt_path


# ============================================================================
# STEP 1 — FETCH & GROUP DATA
# جلب وتجميع البيانات
# ============================================================================

def fetch_selection_questions(survey_number: str) -> pd.DataFrame:
    """
    تجلب الأسئلة غير النصية مع إجاباتها من قاعدة البيانات.
    تُرجع DataFrame بالأعمدة:
        question_id, question_ar, question_type, answer, submission_id
    """
    print(f"\n{'='*60}")
    print(f"📋 Fetching selection questions for survey: {survey_number}")
    print(f"{'='*60}")

    df = get_selection_questions_data(survey_number)

    if df.empty:
        print("⚠️  No selection questions found for this survey.")
        return df

    unique_questions = df["question_id"].nunique()
    print(f"✅ Found {unique_questions} selection question(s), {len(df)} total answer rows.")
    return df


def group_answers_by_question(df: pd.DataFrame) -> dict[int, dict[str, Any]]:
    """
    يُنظّم الإجابات في قاموس مفتاحه question_id.

    يُرجع:
    {
        question_id: {
            "text": "نص السؤال",
            "type": "question_type",
            "answers": ["إجابة1", "إجابة2", ...]
        }
    }
    """
    grouped: dict[int, dict[str, Any]] = {}

    for question_id, group in df.groupby("question_id"):
        grouped[int(question_id)] = {
            "text": group["question_ar"].iloc[0],
            "type": group["question_type"].iloc[0],
            "answers": group["answer"].dropna().tolist()
        }

    return grouped


def get_distinct_answers_per_question(grouped: dict[int, dict[str, Any]]) -> dict[int, list[str]]:
    """
    يحسب الإجابات المميزة (distinct) لكل سؤال.

    يُرجع:
    {
        question_id: ["إجابة_مميزة_1", "إجابة_مميزة_2", ...]
    }
    """
    distinct: dict[int, list[str]] = {}

    for qid, info in grouped.items():
        unique_answers = sorted(set(str(a) for a in info["answers"] if a))
        distinct[qid] = unique_answers
        print(f"   Q{qid} ({info['text'][:40]}...): {len(unique_answers)} distinct answers")

    return distinct


# ============================================================================
# HELPERS — FORMAT DATA FOR PROMPTS
# تنسيق البيانات للطلبات
# ============================================================================

def _format_questions_block(grouped: dict, distinct: dict) -> str:
    """يُنسّق معلومات الأسئلة وإجاباتها المميزة لاستخدامها في الـ prompt."""
    lines = []
    for qid, info in grouped.items():
        lines.append(f"Question ID : {qid}")
        lines.append(f"Question    : {info['text']}")
        lines.append(f"Type        : {info['type']}")
        answers_str = " | ".join(distinct.get(qid, []))
        lines.append(f"Distinct Answers ({len(distinct.get(qid, []))}): {answers_str}")
        lines.append("")
    return "\n".join(lines)


def _extract_sql(text: str) -> str:
    """يستخرج أول كتلة SQL من نص الرد."""
    # محاولة استخراج كتلة ```sql ... ```
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # إذا لم توجد علامة، أُرجع النص كاملاً
    return text.strip()


# ============================================================================
# STEP 2 — SELECTION ANALYSIS AGENT
# وكيل تحليل الأسئلة الفردية
# ============================================================================

_SELECTION_SYSTEM = """\
You are an expert MySQL query generator for survey data analysis.

## Database Schema (MySQL 8+)
- `ms_survey_service.survey`           : id, survey_number, subject_ar
- `ms_survey_service.survey_question`  : id, survey_id, question_type, question_ar
- `ms_survey_service.survey_answer`    : id, survey_question_id, answer, submission_id

## Your Task
Generate a **single valid MySQL 8 query** that analyzes the distribution of answers
for ALL provided selection questions at once.

## Requirements
1. Use a CTE (`WITH`) to pivot answers by `submission_id` (one row per respondent)
2. For EACH question capture its answer:
   `MAX(CASE WHEN sa.survey_question_id = <id> THEN sa.answer END) AS q<id>_answer`
3. In the outer SELECT, compute per-question answer counts and percentages.
4. Use **only** the exact distinct answer values provided — do NOT guess or translate.
5. Use MySQL 8 syntax: `WITH`, window functions, `CAST(... AS DECIMAL(5,2))`.
6. Do NOT use `TOP` (use `LIMIT`), do NOT use `STRING_AGG` (use `GROUP_CONCAT`).
7. Output ONLY the SQL query, wrapped in ```sql ... ``` fences.

## Required Output Structure
```sql
WITH ResponsePivot AS (
    SELECT
        sa.submission_id,
        MAX(CASE WHEN sa.survey_question_id = <q1_id> THEN sa.answer END) AS q<q1_id>_answer,
        MAX(CASE WHEN sa.survey_question_id = <q2_id> THEN sa.answer END) AS q<q2_id>_answer
        -- ... one column per question
    FROM ms_survey_service.survey_answer sa
    WHERE sa.survey_question_id IN (<q1_id>, <q2_id>, ...)
    GROUP BY sa.submission_id
)
SELECT
    -- For each question: answer value, count, percentage
    q<q1_id>_answer                                        AS `<question 1 text>`,
    COUNT(*) AS q<q1_id>_count,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(PARTITION BY q<q1_id>_answer IS NOT NULL)
         AS DECIMAL(5,2)) AS q<q1_id>_pct
    -- repeat for each question
FROM ResponsePivot
GROUP BY q<q1_id>_answer, q<q2_id>_answer
ORDER BY q<q1_id>_count DESC;
```
"""

_SELECTION_USER = """\
Generate the analysis query for the following survey selection questions.

{questions_block}

Return ONLY the MySQL query in a ```sql ... ``` block.
"""

selection_analysis_prompt = ChatPromptTemplate([
    ("system", _SELECTION_SYSTEM),
    ("human", _SELECTION_USER),
])


def build_selection_agent():
    """يُرجع chain = prompt | model لتوليد كويري التحليل الفردي."""
    return selection_analysis_prompt | model


# ============================================================================
# STEP 3 — CORRELATION AGENT
# وكيل تحليل العلاقة بين الأسئلة
# ============================================================================

_CORRELATION_SYSTEM = """\
You are an expert MySQL query generator specializing in cross-question correlation analysis.

## Database Schema (MySQL 8+)
- `ms_survey_service.survey_answer`    : id, survey_question_id, answer, submission_id

## Your Task
Generate a **single valid MySQL 8 query** that shows how responses to different
selection questions correlate with each other using `submission_id` as the
respondent identifier.

## Requirements
1. Use a CTE (`WITH`) to PIVOT by `submission_id`:
   - Each row = one respondent
   - Each column = one question's answer (using MAX(CASE WHEN ...))
2. The outer SELECT must:
   - Show each question's answer column (renamed to actual question text using backtick aliases)
   - Show `COUNT(*) AS ResponseCount`
   - Show `CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS DECIMAL(5,2)) AS PercentageOfTotal`
   - Show `CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(PARTITION BY q<first_id>_answer) AS DECIMAL(5,2)) AS PercentageWithinQ1`
3. Filter out rows where ANY question's answer is NULL
4. GROUP BY all question answer columns
5. ORDER BY ResponseCount DESC
6. Use ONLY the exact distinct answer values provided
7. MySQL 8 syntax only (WITH, window functions, LIMIT, no STRING_AGG, no TOP)
8. Output ONLY the SQL query in ```sql ... ``` fences.
"""

_CORRELATION_USER = """\
Generate a correlation analysis query for the following survey selection questions.

{questions_block}

⚠️ Use `submission_id` for respondent identification across questions.
Return ONLY the MySQL query in a ```sql ... ``` block.
"""

correlation_analysis_prompt = ChatPromptTemplate([
    ("system", _CORRELATION_SYSTEM),
    ("human", _CORRELATION_USER),
])


def build_correlation_agent():
    """يُرجع chain = prompt | model لتوليد كويري تحليل العلاقة."""
    return correlation_analysis_prompt | model


# ============================================================================
# MAIN PIPELINE ORCHESTRATOR
# المُنسّق الرئيسي للمسار
# ============================================================================

def run_selection_pipeline(survey_number: str) -> dict[str, Any]:
    """
    المسار الكامل لتحليل أسئلة الخيارات في الاستبيان.

    الخطوات:
        1. جلب الأسئلة غير النصية مع إجاباتها
        2. تجميع الإجابات وحساب الإجابات المميزة
        3. وكيل 1: يولّد كويري تحليل توزيع الإجابات لكل سؤال
        4. تنفيذ كويري التحليل
        5. وكيل 2: يولّد كويري علاقة (correlation) بين الأسئلة
        6. تنفيذ كويري العلاقة

    يُرجع قاموساً يحتوي على:
        questions_count      : عدد الأسئلة المُحللة
        distinct_per_question: {question_id: [قيم مميزة]}
        selection_sql        : كويري التحليل الفردي
        selection_result     : DataFrame نتيجة كويري التحليل
        correlation_sql      : كويري العلاقة
        correlation_result   : DataFrame نتيجة كويري العلاقة
        errors               : قائمة أي أخطاء حدثت
    """
    print(f"\n{'#'*60}")
    print(f"# SELECTION PIPELINE — Survey: {survey_number}")
    print(f"{'#'*60}\n")

    result: dict[str, Any] = {
        "questions_count": 0,
        "distinct_per_question": {},
        "selection_sql": "",
        "selection_result": pd.DataFrame(),
        "correlation_sql": "",
        "correlation_result": pd.DataFrame(),
        "errors": []
    }

    # ── 1. جلب البيانات ──────────────────────────────────────────────────────
    print("📥 Step 1/5 — Fetching selection questions...")
    df = fetch_selection_questions(survey_number)

    if df.empty:
        result["errors"].append("No selection questions found.")
        return result

    # ── 2. تجميع وحساب الإجابات المميزة ─────────────────────────────────────
    print("\n📊 Step 2/5 — Grouping answers and computing distinct values...")
    grouped = group_answers_by_question(df)
    distinct = get_distinct_answers_per_question(grouped)

    result["questions_count"] = len(grouped)
    result["distinct_per_question"] = distinct

    questions_block = _format_questions_block(grouped, distinct)

    # ── 3. وكيل 1 — توليد كويري التحليل الفردي ───────────────────────────────
    print("\n🤖 Step 3/5 — Selection analysis agent generating SQL...")
    selection_agent = build_selection_agent()

    try:
        selection_response = selection_agent.invoke({"questions_block": questions_block})
        selection_sql = _extract_sql(selection_response.content)
        result["selection_sql"] = selection_sql
        print("✅ Selection SQL generated.")
        print(f"   Preview: {selection_sql[:120].replace(chr(10), ' ')}...")
    except Exception as e:
        err = f"Selection agent failed: {e}"
        print(f"❌ {err}")
        result["errors"].append(err)
        selection_sql = ""

    # ── 4. تنفيذ كويري التحليل الفردي ────────────────────────────────────────
    if selection_sql:
        print("\n⚡ Step 4a/5 — Executing selection analysis query...")
        try:
            result["selection_result"] = execute_raw_query(selection_sql)
            print(f"✅ Selection result: {len(result['selection_result'])} rows")
        except Exception as e:
            err = f"Selection query execution failed: {e}"
            print(f"❌ {err}")
            result["errors"].append(err)

    # ── 5. وكيل 2 — توليد كويري العلاقة ─────────────────────────────────────
    print("\n🤖 Step 4b/5 — Correlation agent generating SQL...")
    correlation_agent = build_correlation_agent()

    try:
        correlation_response = correlation_agent.invoke({"questions_block": questions_block})
        correlation_sql = _extract_sql(correlation_response.content)
        result["correlation_sql"] = correlation_sql
        print("✅ Correlation SQL generated.")
        print(f"   Preview: {correlation_sql[:120].replace(chr(10), ' ')}...")
    except Exception as e:
        err = f"Correlation agent failed: {e}"
        print(f"❌ {err}")
        result["errors"].append(err)
        correlation_sql = ""

    # ── 6. تنفيذ كويري العلاقة ───────────────────────────────────────────────
    if correlation_sql:
        print("\n⚡ Step 5/5 — Executing correlation query...")
        try:
            result["correlation_result"] = execute_raw_query(correlation_sql)
            print(f"✅ Correlation result: {len(result['correlation_result'])} rows")
        except Exception as e:
            err = f"Correlation query execution failed: {e}"
            print(f"❌ {err}")
            result["errors"].append(err)

    # ── ملخص + حفظ ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✅ SELECTION PIPELINE COMPLETE — Survey: {survey_number}")
    print(f"   Questions analyzed : {result['questions_count']}")
    print(f"   Selection rows     : {len(result['selection_result'])}")
    print(f"   Correlation rows   : {len(result['correlation_result'])}")
    if result["errors"]:
        print(f"   ⚠️  Errors ({len(result['errors'])}):")
        for err in result["errors"]:
            print(f"      • {err}")
    print(f"{'='*60}\n")

    # حفظ النتائج في ملف للمراجعة
    report_path = _save_results_to_file(
        survey_number=survey_number,
        grouped=grouped,
        distinct=distinct,
        selection_sql=result["selection_sql"],
        selection_result=result["selection_result"],
        correlation_sql=result["correlation_sql"],
        correlation_result=result["correlation_result"],
        errors=result["errors"]
    )
    result["report_path"] = report_path

    return result

