import json
import re


# ============================================================================
# CHART GENERATION UTILITIES
# أدوات توليد الرسوم البيانية
# ============================================================================


def extract_json_from_response(response_text):
    """
    استخراج JSON من رد اللغوي، مع التعامل مع كتل الكود وأي نص إضافي.

    يُرجع: قائمة من إعدادات الرسوم البيانية
    """
    # إزالة كتل الكود بصيغة markdown
    response_text = re.sub(r'```json\s*', '', response_text)
    response_text = re.sub(r'```\s*', '', response_text)

    # محاولة إيجاد نمط مصفوفة JSON
    json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
    else:
        json_str = response_text.strip()

    try:
        charts = json.loads(json_str)
        if isinstance(charts, list):
            return charts
        elif isinstance(charts, dict):
            return [charts]
        else:
            raise ValueError("Parsed JSON is neither list nor dict")
    except json.JSONDecodeError as e:
        print(f"⚠️ Failed to parse JSON: {e}")
        print(f"Response text: {response_text[:500]}")
        return []


def format_analysis_summary(analysis_results):
    """
    تنسيق نتائج التحليل في ملخص موجز لتوليد الرسوم البيانية.

    يُرجع: نص ملخص منسق
    """
    summary_parts = []

    for q_id, results in analysis_results.items():
        q_text = results.get("survey_question", f"Question {q_id}")

        # توزيع المشاعر
        sentiment_dist = results.get("sentiment_distribution", {})
        if sentiment_dist:
            summary_parts.append(f"**السؤال**: {q_text}")
            summary_parts.append(f"**توزيع المشاعر**: {sentiment_dist}")

        # تحليل المواضيع
        topics_analysis = results.get("topics_analysis", "")
        if topics_analysis:
            summary_parts.append(f"**تحليل المواضيع**: {topics_analysis}")

        # تحليل الكيانات
        entities_analysis = results.get("entities_analysis", "")
        if entities_analysis:
            summary_parts.append(f"**الكيانات المذكورة**: {entities_analysis}")

        # أبرز المواضيع مع الأعداد
        top_topics = results.get("top_topics", [])
        if top_topics:
            if isinstance(top_topics, dict):
                topics_str = ", ".join([f"{topic} ({count})" for topic, count in list(top_topics.items())[:5]])
            elif isinstance(top_topics, list):
                topics_str = ", ".join([f"{t.get('topic', t.get('label', 'N/A'))} ({t.get('count', 0)})" for t in top_topics[:5]])
            else:
                topics_str = str(top_topics)
            summary_parts.append(f"**أبرز المواضيع**: {topics_str}")

        # أبرز الكيانات مع الأعداد
        top_entities = results.get("top_entities", [])
        if top_entities:
            if isinstance(top_entities, dict):
                entities_str = ", ".join([f"{entity} ({count})" for entity, count in list(top_entities.items())[:5]])
            elif isinstance(top_entities, list):
                entities_str = ", ".join([f"{e.get('entity', e.get('label', 'N/A'))} ({e.get('count', 0)})" for e in top_entities[:5]])
            else:
                entities_str = str(top_entities)
            summary_parts.append(f"**أبرز الكيانات**: {entities_str}")

        summary_parts.append("---")

    return "\n".join(summary_parts)
