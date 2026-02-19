import os
import sys
import datetime

# إعداد المسار للاستيراد
current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from analytics_pipeline.util.config import EXPORTS_DIR


# ============================================================================
# SYNTHESIS UTILITIES
# أدوات توليد التقرير التنفيذي
# ============================================================================


def format_analytics_messages(analysis_results):
    """تحويل نتائج التحليل إلى رسائل جاهزة للتلخيص."""
    messages = []

    for q_id, results in analysis_results.items():
        q_text = results.get("survey_question", f"Question {q_id}")
        sentiment_dist = results.get("sentiment_distribution", "N/A")
        topics_analysis = results.get("topics_analysis", "")
        entities_analysis = results.get("entities_analysis", "")

        msg_parts = [f"Question: {q_text}"]
        msg_parts.append(f"Sentiment Distribution: {sentiment_dist}")

        if topics_analysis:
            msg_parts.append(f"Topics Analysis: {topics_analysis}")
        if entities_analysis:
            msg_parts.append(f"Entities Analysis: {entities_analysis}")

        messages.append("\n".join(msg_parts))

    return messages


def save_report(content, survey_id):
    """حفظ التقرير التنفيذي في ملف."""
    os.makedirs(EXPORTS_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"executive_summary_{survey_id}_{timestamp}.txt"
    filepath = os.path.join(EXPORTS_DIR, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"💾 Executive summary saved to: {filepath}")
    return filepath
