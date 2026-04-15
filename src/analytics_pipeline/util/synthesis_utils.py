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
        q_text = results.get("question_ar", f"Question {q_id}")
        dominant_sentiment = results.get("dominant_sentiment", "N/A")
        top_positive_topics = results.get(
            "top_positive_topics", []
        )  # list of (topic, count)
        top_entities = results.get("top_entities", [])
        sentiment_distribution = results.get("sentiment_distribution", {})
        final_summary = results.get("final_summary", "")

        msg_parts = [f"Question: {q_text}"]
        msg_parts.append(f"Dominant Sentiment: {dominant_sentiment}")

        if sentiment_distribution:
            dist_str = ", ".join(f"{k}: {v}" for k, v in sentiment_distribution.items())
            msg_parts.append(f"Sentiment Distribution: {dist_str}")

        if top_positive_topics:
            topics_str = ", ".join(f"{t} ({c})" for t, c in top_positive_topics)
            msg_parts.append(f"Top Positive Topics: {topics_str}")
        if top_entities:
            msg_parts.append(f"Top Entities: {', '.join(top_entities)}")
        if final_summary:
            msg_parts.append(f"Summary: {final_summary}")

        messages.append("\n".join(msg_parts))

    return messages


def save_report(content, survey_number):
    """حفظ التقرير التنفيذي في ملف."""
    os.makedirs(EXPORTS_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"executive_summary_{survey_number}_{timestamp}.txt"
    filepath = os.path.join(EXPORTS_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"💾 Executive summary saved to: {filepath}")
    return filepath
