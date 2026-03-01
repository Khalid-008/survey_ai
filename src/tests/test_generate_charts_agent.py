"""
test_generate_charts_agent.py
=============================
اختبار مستقل لنود generate_charts_agent.

يقرأ البيانات الحقيقية من ملف:
    tests/chart_agent_snapshot.json
الذي يُنشأ تلقائياً عند تشغيل النظام الكامل.

كيفية تشغيل النظام أولاً لتوليد الـ snapshot:
    - شغّل السيرفر عادي وارسل أي طلب  ← يُنشئ chart_agent_snapshot.json

ثم شغّل هذا الملف:
    cd "c:/Users/Khalid PC/Desktop/Channels/survey_ai/src"
    python tests/test_generate_charts_agent.py
"""

import os
import sys
import json

# ── إعداد المسار ──────────────────────────────────────────────────────────────
current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(current_file_path))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from langchain_core.messages import HumanMessage
from agents.graphs.nodes import generate_charts_agent

# ── مسار الـ snapshot ──────────────────────────────────────────────────────────
SNAPSHOT_PATH = os.path.join(src_dir, "tests", "chart_agent_snapshot.json")


def main():
    print("=" * 60)
    print("TEST: generate_charts_agent")
    print("=" * 60)

    # ── تحقق من وجود الملف ────────────────────────────────────────────────────
    if not os.path.exists(SNAPSHOT_PATH):
        print(f"❌ الملف غير موجود: {SNAPSHOT_PATH}")
        print("   شغّل النظام الكامل أولاً لتوليد الـ snapshot.")
        return

    # ── قراءة البيانات الحقيقية ───────────────────────────────────────────────
    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        snapshot = json.load(f)

    survey_id                  = snapshot["survey_id"]
    text_questions_result      = snapshot.get("text_questions_result", {})
    selection_questions_result = snapshot.get("selection_questions_result", {})

    print(f"📂 تم تحميل الـ snapshot من: {SNAPSHOT_PATH}")
    print(f"📋 Survey ID         : {survey_id}")
    print(f"📝 Text questions    : {len(text_questions_result)} سؤال")
    print(f"☑️  Selection queries : {len(selection_questions_result.get('query_results', []))} كويري")
    print()

    # ── بناء الـ state وتشغيل النود ───────────────────────────────────────────
    mock_state = {
        "survey_id":                   survey_id,
        "messages":                    [HumanMessage(content="حلل الاستبيان")],
        "survey_data":                 [],
        "text_questions_result":       text_questions_result,
        "selection_questions_result":  selection_questions_result,
        "chart_configs":               [],
    }

    result = generate_charts_agent(mock_state)

    # ── عرض النتائج ───────────────────────────────────────────────────────────
    update       = result.update if hasattr(result, "update") else {}
    chart_configs = update.get("chart_configs", [])

    print("\n" + "=" * 60)
    print(f"✅ النتيجة: {len(chart_configs)} مخطط بياني")
    print("=" * 60)

    for i, chart in enumerate(chart_configs, 1):
        print(f"\n📊 مخطط {i}:")
        print(f"   النوع  : {chart.get('type', '?')}")
        print(f"   العنوان: {chart.get('title', 'بدون عنوان')}")
        labels = chart.get("data", {}).get("labels", [])
        print(f"   التسميات ({len(labels)}): {labels[:5]}")

    # ── حفظ المخرجات ──────────────────────────────────────────────────────────
    output_path = os.path.join(src_dir, "tests", "chart_output.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chart_configs, f, ensure_ascii=False, indent=2)
    print(f"\n💾 المخرجات محفوظة في: {output_path}")


if __name__ == "__main__":
    main()
