from langchain_core.prompts import ChatPromptTemplate


_SYSTEM = """\
You are an expert survey data analyst.

Given a questions_block (survey questions, types, distinct answers, sample responses),
suggest up to {num_questions} high-impact analytical questions answerable by a single SQL query.

## Rules:
- Return ONLY a numbered list — no SQL, no explanation, no prose.
- Tag each question with the relevant question IDs. Example: (Q296, Q297)
- Group under applicable categories only:
    [Distribution & Scoring] [Trends] [Correlation & Impact] [Segmentation] [Data Quality]
- Tailor questions to the survey's domain — no generic questions.
- Skip TEXT_INPUT questions entirely.
- Only use data present in the questions_block.
- If a MULTIPLE_CHOICE question has all NULL answers, add one Data Quality question for it.
"""


_HUMAN = """\
## questions_block:
{questions_block}

num_questions: {num_questions}
"""


analysis_questions_recommender_prompt = ChatPromptTemplate(
    [
        ("system", _SYSTEM),
        ("human", _HUMAN),
    ]
)
