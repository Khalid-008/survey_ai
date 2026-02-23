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


# ── Questions block formatter ─────────────────────────────────────────────────

def format_questions_block(grouped: dict, distinct: dict) -> str:
    lines = []
    for qid, info in grouped.items():
        lines.append(f"Question ID : {qid}")
        lines.append(f"Question    : {info['text']}")
        lines.append(f"Type        : {info['type']}")
        answers_str = " | ".join(distinct.get(qid, []))
        lines.append(f"Distinct Answers ({len(distinct.get(qid, []))}): {answers_str}")
        lines.append("")
    return "\n".join(lines)


# ── Results export ────────────────────────────────────────────────────────────

def save_results_to_file(
    survey_number: str,
    grouped: dict,
    distinct: dict,
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

        f.write("── QUESTIONS & DISTINCT ANSWERS ──\n")
        for qid, info in grouped.items():
            f.write(f"\nQ{qid}: {info['text']}\n")
            f.write(f"  Type   : {info['type']}\n")
            f.write(f"  Distinct answers ({len(distinct.get(qid, []))}):\n")
            for ans in distinct.get(qid, []):
                f.write(f"    • {ans}\n")

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

    for idx, qr in enumerate(query_results, start=1):
        df = qr.get("result", pd.DataFrame())
        label_safe = re.sub(r"[^\w]", "_", qr.get("label", f"query_{idx}"))[:40]
        if not df.empty:
            csv_path = os.path.join(EXPORTS_DIR, f"{base_name}_{label_safe}.csv")
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"📄 CSV [{label_safe}]: {csv_path}")

    print(f"📄 Report TXT: {txt_path}")
    return txt_path
