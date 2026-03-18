from langchain_core.prompts import ChatPromptTemplate

_SYSTEM = """\
You are an expert Arabic survey data analyst.

You will receive partial batch summaries from multiple batches of survey answers
for a single question. Each batch summary contains: top_topics, top_entities, 
and a short summary of that batch.

Your task is to synthesize ALL batch summaries into ONE final consolidated report
for the entire question.

## Rules:
- Respond ONLY with valid JSON. No prose, no explanations outside JSON.
- Merge and deduplicate topics and entities across batches (keep the most frequent/important).
- sentiment_distribution is an ESTIMATE based on the batch summaries (best effort).
- dominant_sentiment: whichever sentiment appears most across batches.
- final_summary must be 2-4 sentences in Arabic, capturing the overall picture.
- Return at most 10 top_topics and 10 top_entities.

## Output format (strict JSON):
{{
  "question_id": <integer>,
  "question_ar": "<question text>",
  "dominant_sentiment": "positive" | "negative" | "neutral",
  "top_topics": ["<topic1>", "..."],
  "top_entities": ["<entity1>", "..."],
  "final_summary": "<2-4 sentences in Arabic>"
}}
"""

_HUMAN = """\
## Question ID: {question_id}
## Question: {question_ar}
## Total answers analyzed: {total_answers}

## Batch summaries (JSON array):
{batch_summaries_json}

Return ONLY the JSON object described above. Do NOT include markdown code fences.
"""

text_analysis_synthesis_prompt = ChatPromptTemplate([
    ("system", _SYSTEM),
    ("human",  _HUMAN),
])
