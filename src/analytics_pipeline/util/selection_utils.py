import os
import re
from datetime import datetime

import pandas as pd


EXPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "exports", "selection_results"
)
os.makedirs(EXPORTS_DIR, exist_ok=True)


# ── SQL extraction ────────────────────────────────────────────────────────────

def extract_sql(text: str) -> str:
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


# ── SQL cleaning & splitting ───────────────────────────────────────────────────

def clean_sql_queries(raw_output: str) -> list[str]:
    """
    Strips prose mixed into LLM output, auto-fixes COUNT() and CAST AS INT,
    then splits the output into individual executable SQL queries.
    """
    SQL_KEYWORDS = ('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE', '--')

    lines = raw_output.split('\n')
    cleaned_lines: list[str] = []
    inside_query = False

    for line in lines:
        stripped = line.strip()

        # Skip prose lines that appear before any SQL keyword
        if not inside_query and stripped and not any(
            stripped.upper().startswith(kw) for kw in SQL_KEYWORDS
        ):
            continue

        if any(stripped.upper().startswith(kw) for kw in SQL_KEYWORDS):
            inside_query = True

        if inside_query:
            cleaned_lines.append(line)

        if stripped.endswith(';'):
            inside_query = False

    joined = '\n'.join(cleaned_lines)

    # Fix COUNT() → COUNT(*)
    joined = re.sub(r'\bCOUNT\s*\(\s*\)', 'COUNT(*)', joined)

    # Fix CAST(x AS INT) → CAST(x AS SIGNED)
    joined = re.sub(
        r'CAST\(([^)]+)\s+AS\s+INT\)',
        lambda m: f'CAST({m.group(1)} AS SIGNED)',
        joined,
        flags=re.IGNORECASE,
    )

    # Split into individual queries on the semicolon boundary
    queries: list[str] = []
    current: list[str] = []

    for line in joined.split('\n'):
        current.append(line)
        if line.strip().endswith(';'):
            query = '\n'.join(current).strip()
            if query and len(query) > 20:
                queries.append(query)
            current = []

    return queries


# ── SQL validation ─────────────────────────────────────────────────────────────

def validate_query(sql: str) -> tuple[bool, list[str]]:
    """
    Checks basic MySQL 5.7 rule compliance before execution.
    Returns (is_valid, list_of_issues).
    """
    issues: list[str] = []

    if re.search(r'\bCOUNT\s*\(\s*\)', sql):
        issues.append("❌ COUNT() without * found")

    if re.search(r'\bOVER\s*\(', sql, re.IGNORECASE):
        issues.append("❌ Window function OVER() found")

    if re.search(r'CAST\s*\(.*AS\s+INT\b', sql, re.IGNORECASE):
        issues.append("❌ CAST AS INT found (use SIGNED)")

    first_line = sql.strip().split('\n')[0].strip()
    if not first_line.startswith(('--', 'SELECT', 'WITH')):
        issues.append("❌ Query starts with prose text, not SQL")

    return len(issues) == 0, issues


# ── Questions block formatter ─────────────────────────────────────────────────

def format_questions_block(grouped: dict, distinct: dict, sample_answers: dict | None = None, survey_number: str = "") -> str:
    lines = []
    if survey_number:
        lines.append(f"Survey Number : {survey_number}")
        lines.append("")
    for qid, info in grouped.items():
        lines.append(f"Question ID : {qid}")
        lines.append(f"Question    : {info['text']}")
        lines.append(f"Type        : {info['type']}")
        answers_str = " | ".join(distinct.get(qid, []))
        lines.append(f"Distinct Answers ({len(distinct.get(qid, []))}): {answers_str}")
        if sample_answers and qid in sample_answers and sample_answers[qid]:
            samples_str = " | ".join(str(s) for s in sample_answers[qid])
            lines.append(f"Sample Raw Answers (3 examples): {samples_str}")
        lines.append("")
    return "\n".join(lines)


# ── Results export ────────────────────────────────────────────────────────────

def save_results_to_file(
    survey_number: str,
    query_results: list[dict],
    errors: list
) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{survey_number}_{ts}"

    txt_path = os.path.join(EXPORTS_DIR, f"{base_name}_selection_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("SELECTION PIPELINE REPORT\n")
        f.write(f"Survey   : {survey_number}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("── QUESTIONS ──\n")

        for idx, qr in enumerate(query_results, start=1):
            label = qr.get("label", f"Query {idx}")
            sql   = qr.get("sql", "")
            df    = qr.get("result", pd.DataFrame())

            f.write(f"\n\n── GENERATED SQL: {label} ──\n")
            f.write(sql if sql else "(not generated)")

            f.write(f"\n\n── RESULT: {label} ──\n")
            if not df.empty:
                f.write(df.to_string(index=False))
                f.write(f"\n\n({len(df)} rows)\n")
            else:
                f.write("(empty)\n")

        if errors:
            f.write("\n\n── ERRORS ──\n")
            for err in errors:
                f.write(f"  • {err}\n")

    print(f"📄 Report TXT: {txt_path}")
    return txt_path
