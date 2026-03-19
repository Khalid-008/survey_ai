import ast
import json
import traceback

import pandas as pd
from llms.models import model
from agents.prompt.semantic_normalization_prompt import semantic_normalization_prompt
from agents.prompt.text_analysis_batch_prompt import text_analysis_batch_prompt
from agents.prompt.text_analysis_synthesis_prompt import text_analysis_synthesis_prompt
from helper.utils import parse_json_response


def analyze_text_batch(
    question_id: int,
    question_ar: str,
    batch_records: list[dict],
    batch_idx: int,
    total_batches: int,
) -> tuple[list[dict], dict | None]:

    label = f"Q{question_id} | Batch {batch_idx}/{total_batches}"

    answers_payload = [
        {"answer_id": int(r["answer_id"]), "answer": str(r.get("answer", "") or "")}
        for r in batch_records
    ]

    try:
        prompt = text_analysis_batch_prompt.invoke(
            {
                "question_id": question_id,
                "question_ar": question_ar,
                "answers_json": json.dumps(answers_payload, ensure_ascii=False),
            }
        )
        response = model.invoke(prompt)
        parsed = parse_json_response(response.content, context=label)

        if not parsed:
            return [], None

        answers_analysis = parsed.get("answers_analysis", [])
        batch_summary = {
            "top_topics": parsed.get("top_topics", []),
            "top_entities": parsed.get("top_entities", []),
            "summary": parsed.get("summary", ""),
        }

        print(
            f"   ✅ {label} → Stage 1 done " f"({len(answers_analysis)} answers tagged)"
        )
        return answers_analysis, batch_summary

    except Exception as e:
        print(f"   ❌ {label} → Stage 1 failed: {e}")
        print(traceback.format_exc())
        return [], None


def synthesize_batch_summaries(
    question_id: int,
    question_ar: str,
    batch_summaries: list[dict],
    total_answers: int,
) -> dict | None:
    """
    Call LLM Stage 2 (synthesis) with all batch summaries for one question.

    Returns:
        final_report dict or None on failure
    """
    label = f"Q{question_id} | Stage 2 synthesis"

    try:
        prompt = text_analysis_synthesis_prompt.invoke(
            {
                "question_id": question_id,
                "question_ar": question_ar,
                "total_answers": total_answers,
                "batch_summaries_json": json.dumps(batch_summaries, ensure_ascii=False),
            }
        )
        response = model.invoke(prompt)
        parsed = parse_json_response(response.content, context=label)

        if parsed:
            print(f"   ✅ {label} → done")
        else:
            print(f"   ⚠️  {label} → parse failed, storing raw batch summaries")

        return parsed

    except Exception as e:
        print(f"   ❌ {label} → failed: {e}")
        print(traceback.format_exc())
        return None


def semantic_normalization(
    raw_counts: list[tuple[str, int]], top_n: int = 10
) -> list[tuple[str, int]]:
    """Group string frequencies semantically using LLM mapping."""
    if not raw_counts:
        return []

    unique_items = [item[0] for item in raw_counts]
    print(f"🔄 جاري توحيد {len(unique_items)} عنصر متكرر باستخدام الذكاء الاصطناعي...")

    prompt_messages = semantic_normalization_prompt.invoke({
        "items_json": json.dumps(unique_items, ensure_ascii=False)
    })

    try:
        response = model.invoke(prompt_messages)
        content = response.content.replace("```json", "").replace("```", "").strip()
        mapping = json.loads(content)
        
        aggregated_counts = {}
        for item, count in raw_counts:
            normalized_item = mapping.get(item, item)
            aggregated_counts[normalized_item] = aggregated_counts.get(normalized_item, 0) + count
            
        result = sorted(aggregated_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        print("✅ تم توحيد العناصر وتجميع التكرارات بنجاح!")
        return result
        
    except Exception as e:
        print(f"⚠️ فشل عملية توحيد العناصر: {str(e)}")
        return sorted(raw_counts, key=lambda x: x[1], reverse=True)[:top_n]
