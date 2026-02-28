import json
from langchain_core.messages import SystemMessage


# ── System Prompt (ثابت — البرومبت فقط) ──────────────────────────────────────

SYNTHESIS_SYSTEM_PROMPT = """\
You are a Strategic Synthesis Agent — a senior business intelligence consultant.
Transform raw analytical outputs into polished executive intelligence without revealing
the underlying technical processes. Respond exclusively in Arabic.

---

OUTPUT FORMAT

Return ONLY a valid JSON object. No text before or after it, no markdown, no code fences.
Start with { and end with }.

JSON escaping rules:
- Use \\n for line breaks inside strings
- Escape double quotes as \\" — never escape single quotes (\\' is invalid JSON)
- No trailing commas

Required fields:

{
  "detailed_analysis": "Main narrative with \\n\\n between paragraphs. Use **bold** for emphasis.",
  "executive_summary": "2-3 sentence C-level summary.",
  "key_metrics": ["Metric 1 with context", "Metric 2 with context"],
  "recommendations": ["Actionable recommendation 1", "Actionable recommendation 2"],
  "visualizations": []
}

Field guide:
- detailed_analysis — narrative body (see writing standards below)
- executive_summary — answers the core business question concisely
- key_metrics — 3-6 supporting data points
- recommendations — actionable next steps derived from the analysis
- visualizations — leave as empty array []

---

WRITING STANDARDS

Style:
✅ Executive language — confident, clear, actionable
✅ Numbers as digits with symbols: 85%, SAR2.5M, 1,250 units
✅ Natural flow: "Revenue grew 23% to SAR4.2M"
✅ **Bold text** for section emphasis (never ## headers)

❌ Technical references: "SQL shows", "according to data", "Question 296"
❌ Written numbers: "twenty-three percent"
❌ ## headers anywhere in content

Structure:
- Open with the most impactful insight
- Build a logical narrative; end with a forward-looking perspective
- Round percentages reasonably: 97.2% → 97%

---

QUALITY CHECKLIST

□ Answers the core business question directly
□ All metrics have strategic context
□ No data conflicts left unreconciled
□ Valid JSON — all required fields present
□ Numbers as digits, no ## headers, bold used for emphasis only
□ Recommendations are actionable
"""


# ── Data Builder (كود — يبني البيانات الديناميكية) ───────────────────────────

def _build_analytics_section(selection_questions_result: dict, text_questions_result: list) -> str:
    """تجميع نتائج التحليلين في نص واحد يُرسل للـ LLM."""
    parts = []

    # 0) دليل الأسئلة — يعطي النموذج أسماء الأسئلة بدلاً من أرقامها
    if selection_questions_result and isinstance(selection_questions_result, dict):
        metadata = selection_questions_result.get("questions_metadata", [])
        if metadata:
            glossary_lines = ["## دليل الأسئلة\n"]
            for q in metadata:
                glossary_lines.append(f"- **Q{q['id']}** ({q['type']}): {q['text']}")
            parts.append("\n".join(glossary_lines))

    # 1) نتائج تحليل أسئلة النصوص (NLP)
    if text_questions_result:
        text_block_parts = []
        for i, msg in enumerate(text_questions_result):
            text_block_parts.append(f"### تحليل {i + 1}\n{msg}")
        parts.append("## نتائج تحليل أسئلة النصوص (NLP)\n\n" + "\n\n".join(text_block_parts))

    # 2) نتائج تحليل أسئلة الاختيار
    if selection_questions_result and isinstance(selection_questions_result, dict):
        sel_block_parts = []
        for qr in selection_questions_result.get("query_results", []):
            label = qr.get("label", "")
            rows  = qr.get("result", [])
            if rows:
                sel_block_parts.append(
                    f"### {label}\n"
                    + json.dumps(rows, ensure_ascii=False, indent=2)
                )
        if sel_block_parts:
            parts.append("## نتائج تحليل أسئلة الاختيار\n\n" + "\n\n".join(sel_block_parts))

    return "\n\n---\n\n".join(parts) if parts else "لا توجد بيانات تحليلية متاحة."


# ── Main Function ─────────────────────────────────────────────────────────────

def synthesis_agent_prompt_function(survey_subject, selection_questions_result, text_questions_result):
    """
    Creates a synthesis agent prompt to transform analytics data into a structured JSON report.

    Args:
        survey_subject:             Title / subject of the survey
        selection_questions_result: dict from analyze_selection_questions (query_results list)
        text_questions_result:      list of strings from analyze_text_questions (NLP enrichment)

    Returns a list of messages for the LLM.
    """
    analytics_section = _build_analytics_section(selection_questions_result, text_questions_result)

    prompt_text = f"""{SYNTHESIS_SYSTEM_PROMPT}

---

## CURRENT TASK

**موضوع الاستبيان:** {survey_subject}

**البيانات التحليلية للتوليف:**

{analytics_section}

---

قم بتوليف البيانات التحليلية أعلاه في تقرير تنفيذي شامل بصيغة JSON فقط.
ركّز على استخراج رؤى قابلة للتنفيذ خاصة بموضوع الاستبيان: "{survey_subject}".
أجب باللغة العربية حصراً.
تأكد من صحة JSON قبل الإرسال."""

    return [SystemMessage(content=prompt_text)]


# ── Compatibility Wrapper ─────────────────────────────────────────────────────

class SynthesisAgentPrompt:
    """Wrapper class to maintain .invoke() compatibility"""

    def invoke(self, inputs):
        survey_subject             = inputs.get("survey_subject", "Survey Data")
        selection_questions_result = inputs.get("selection_questions_result", {})
        text_questions_result      = inputs.get("text_questions_result", [])
        return synthesis_agent_prompt_function(survey_subject, selection_questions_result, text_questions_result)


# Create the instance with the same name for compatibility
synthesis_agent_prompt = SynthesisAgentPrompt()
