import json
import re


# ============================================================================
# CHART GENERATION UTILITIES
# أدوات توليد الرسوم البيانية
# ============================================================================


def _repair_json_array(json_str):
    """
    إصلاح أخطاء شائعة في JSON الناتج عن النموذج اللغوي:
    1. الفاصلة المزدوجة:  ["1",,"2"]  →  ["1","2"]
    2. قوس الفتح المفقود: [{...},"type":...}]  →  [{...},{"type":...}]
    """
    # إزالة الفواصل المزدوجة (مثل ["1",,"2","3"])
    repaired = re.sub(r',\s*,', ',', json_str)

    # إضافة { بعد الفاصلة مباشرةً إذا جاء بعدها مفتاح JSON مباشرة بدون قوس فتح
    repaired = re.sub(r',\s*\"(type|title|data)\":', r',{\"\\1\":', repaired)
    return repaired


def _extract_chart_objects(text):
    """
    استراتيجية احتياطية: استخراج كائنات JSON فردية عندما تفشل قراءة المصفوفة الكاملة.
    مفيدة عندما يكون الرد مقطوعاً في المنتصف (truncated response).
    """
    charts = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start:i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict) and 'type' in obj and 'data' in obj:
                        charts.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None
    return charts


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

    # المحاولة الأولى: تحليل مباشر
    try:
        charts = json.loads(json_str)
        if isinstance(charts, list):
            return charts
        elif isinstance(charts, dict):
            return [charts]
    except json.JSONDecodeError as e:
        print(f"⚠️ Failed to parse JSON (attempt 1): {e}")

        # المحاولة الثانية: إصلاح الأخطاء الشائعة ثم إعادة التحليل
        repaired = _repair_json_array(json_str)
        print(f"🔧 Attempting JSON repair...")
        try:
            charts = json.loads(repaired)
            if isinstance(charts, list):
                print(f"✅ JSON repaired successfully ({len(charts)} charts)")
                return charts
            elif isinstance(charts, dict):
                return [charts]
        except json.JSONDecodeError as e2:
            print(f"⚠️ Failed to parse JSON (attempt 2 after repair): {e2}")
            print(f"Response text: {response_text[:500]}")

        # المحاولة الثالثة: استخراج الكائنات المكتملة فردياً (للردود المقطوعة)
        print(f"🔧 Attempting fallback: extracting individual chart objects...")
        charts = _extract_chart_objects(repaired)
        if charts:
            print(f"✅ Fallback recovered {len(charts)} chart object(s)")
            return charts

    return []


def format_analysis_summary(text_questions_result, selection_questions_result=None):
    """
    تنسيق نتائج التحليل في ملخص موجز لتوليد الرسوم البيانية.

    - text_questions_result   : يُستخدم منه توزيع المشاعر فقط (بدون مواضيع أو كيانات)
    - selection_questions_result : نتائج استعلامات أسئلة الخيارات

    يُرجع: نص ملخص منسق
    """
    parts = []

    # ── 1) توزيع المشاعر من أسئلة النص ──────────────────────────────────────
    sentiment_parts = []
    for q_id, results in text_questions_result.items():
        q_text = results.get("survey_question", f"Question {q_id}")
        sentiment_dist = results.get("sentiment_distribution", {})
        if sentiment_dist:
            sentiment_parts.append(f"**السؤال**: {q_text}")
            sentiment_parts.append(f"**توزيع المشاعر**: {sentiment_dist}")
            sentiment_parts.append("---")

    if sentiment_parts:
        parts.append("## تحليل المشاعر (أسئلة النص)\n\n" + "\n".join(sentiment_parts))

    # ── 2) نتائج أسئلة الخيارات ──────────────────────────────────────────────
    if selection_questions_result and isinstance(selection_questions_result, dict):
        sel_parts = []
        for qr in selection_questions_result.get("query_results", []):
            label = qr.get("label", "")
            rows  = qr.get("result", [])
            if rows:
                sel_parts.append(
                    f"### {label}\n"
                    + json.dumps(rows, ensure_ascii=False, indent=2)
                )
        if sel_parts:
            parts.append("## نتائج أسئلة الخيارات\n\n" + "\n\n".join(sel_parts))

    return "\n\n---\n\n".join(parts) if parts else "No analytical data available."
