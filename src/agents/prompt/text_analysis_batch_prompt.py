from langchain_core.prompts import ChatPromptTemplate

_SYSTEM = """\
You are an expert Arabic NLP analyst specializing in survey response analysis.

You will receive a batch of survey answers for a single question.
Your task is to:
1. Analyze EACH answer individually for sentiment, entities, and topics.
2. Provide a batch-level summary of this specific group of answers.

## Rules:
- Respond ONLY with valid JSON. No prose, no explanations outside JSON.
- STRICTLY use double quotes (") for all keys/values. NO trailing commas.
- Sentiment must be exactly one of: "positive", "negative", "neutral".
- Entities: named things mentioned (people, places, products, departments, etc.).
- Topics: abstract themes or subjects discussed.
- Keep entities and topics in the SAME language as the answer (Arabic or English).
- If an answer is empty, irrelevant, or gibberish → sentiment: "neutral", empty lists.
- The batch summary (top_topics, top_entities, summary) must reflect THIS batch only.
- summary must be 1-2 sentences in Arabic.

## Output format (strict JSON):
{{
  "question_id": <integer>,
  "answers_analysis": [
    {{
      "answer_id": <integer>,
      "sentiment": "positive" | "negative" | "neutral",
      "entities": ["<entity1>", "..."],
      "topics": ["<topic1>", "..."]
    }}
  ],
  "top_topics": ["<topic1>", "<topic2>", "..."],
  "top_entities": ["<entity1>", "<entity2>", "..."],
  "summary": "<one or two sentences in Arabic summarizing this batch>"
}}
"""

_HUMAN = """\
## Question ID: {question_id}
## Question: {question_ar}

## Answers batch (JSON):
{answers_json}

Return ONLY the JSON object described above. Do NOT include markdown code fences.
"""

text_analysis_batch_prompt = ChatPromptTemplate([
    ("system", _SYSTEM),
    ("human",  _HUMAN),
])
