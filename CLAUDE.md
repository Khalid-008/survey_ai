# Survey AI - Claude Code Instructions

## Project Overview
**Survey AI** is a multi-agent AI system for analyzing survey data. It provides intelligent insights, visualizations, and recommendations based on survey responses using LangGraph workflows and LangChain agents.

## Tech Stack
- **Backend**: Flask (Python)
- **AI Framework**: LangGraph + LangChain
- **LLM**: Google Gemini (gemini-2.5-flash)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Database**: MySQL (Alibaba Cloud RDS)
- **State Management**: Redis (session/checkpoint storage)
- **Vector Store**: In-memory vector store for RAG
- **Monitoring**: Langfuse (AI observability)
- **Frontend**: Single-page HTML/JS application (RTL Arabic UI)

## Architecture

### Multi-Agent Workflow
The system uses a **LangGraph StateGraph** with the following nodes:

1. **selected_messages_node**: Trims conversation history (last 10 messages)
2. **retrieve_survey_question**: Fetches survey questions from MySQL
3. **get_relevant_question**: LLM selects relevant questions based on user query
4. **get_answers**: Retrieves answers from database using SQL queries
5. **classify_data_type**: Determines if data is qualitative (RAG) or quantitative (SQL)
6. **upload_rag** / **generate_analysis_from_rag**: For text-heavy data
7. **generate_analysis_from_sql**: For numerical/structured data
8. **synthesis_agent**: Combines all analyses into final response with visualizations

### Data Flow
```
User Query → Message Trimming → Retrieve Questions → Get Relevant Questions
→ Loop through questions → Get Answers → Classify Data Type
→ (RAG or SQL Analysis) → Synthesis → Visualization → JSON Response
```

## Key Files

### Backend
- **src/app.py** - Flask API endpoint (`/survey_insight`)
- **src/agents/graphs/workflow.py** - Main LangGraph workflow
- **src/agents/graphs/nodes.py** - All graph nodes
- **src/agents/graphs/setup.py** - State schema
- **src/data/operations.py** - Database operations
- **src/data/text_to_sql.py** - Text-to-SQL conversion
- **src/llms/models.py** - LLM and embeddings models
- **src/helper/utils.py** - JSON extraction and utility functions

### Frontend
- **ai-data-chat.html** - Full chat interface with Chart.js rendering

### Configuration
- **.env** - Environment variables (API keys, DB connection, Redis URI)
- **requirements.txt** - Python dependencies

## Data Models

### SurveyResult (src/data/models.py)
```python
class SurveyResult:
    id: str
    subject: str
    description: str
    question: str
    answer: str
```

### State (LangGraph)
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
    selected_messages: list
    survey_id: int
    questions: list
    answers: dict
    next_question: int
    analysis: list
```

## API Specification

### POST /survey_insight
**Request:**
```json
{
  "request": {
    "survey_id": "123",
    "message": "ما هي أكثر الإجابات شيوعاً؟",
    "session_id": "session_123456"
  }
}
```

**Response:**
```json
{
  "executive_summary": "نص تنفيذي...",
  "detailed_analysis": "تحليل مفصل...",
  "key_metrics": ["مقياس 1", "مقياس 2"],
  "recommendations": ["توصية 1", "توصية 2"],
  "visualizations": [
    {
      "chart_description": "وصف الرسم البياني",
      "chart_html": "<div>...</div>"
    }
  ]
}
```

## Database Schema
- **Survey Questions**: Retrieved via `get_survey_questions(survey_id)`
- **Answers**: Retrieved via SQL queries generated from natural language
- **MySQL Connection**: Uses environment variable `DB_URI`

## Environment Variables
```bash
PROJECT_NAME=SurveyAI
MODEL=gemini-2.5-flash
EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2
GOOGLE_API_KEY=<api_key>
DB_URI=<mysql_connection_string>
REDIS_URI=redis://localhost:6377
LANGFUSE_PUBLIC_KEY=<key>
LANGFUSE_SECRET_KEY=<key>
LANGFUSE_HOST=http://localhost:3000
```

## Visualization System
- Charts generated as **HTML strings** (Chart.js embedded)
- Rendered in iframes on frontend
- Supported chart types: Bar, Column, Pie, Donut, Area, Histogram, Scatter, Bubble, Pyramid (see charts.txt)

## Development Notes

### Running the Application
```bash
# Install dependencies
pip install -r requirements.txt

# Start Flask server
python src/app.py  # Runs on http://0.0.0.0:5566

# Open frontend
# Open ai-data-chat.html in browser
```

### Adding New Agents
1. Create prompt in `src/agents/prompt/`
2. Add node function in `src/agents/graphs/nodes.py`
3. Register in `src/agents/graphs/workflow.py`
4. Update State schema if needed

### Debugging
- **Langfuse**: Monitor traces at http://localhost:3000
- **Redis**: Check sessions via Redis CLI
- **Logs**: Flask debug mode enabled in app.py

## Common Patterns

### LLM Invocation
```python
from llms.models import model
prompt = template.invoke({"var": value})
result = model.invoke(prompt)
content = result.content
```

### JSON Extraction from LLM Response
```python
from helper.utils import extract_json
data = extract_json(llm_response)
```

### Database Query
```python
from data.operations import run_query
from data.text_to_sql import write_query
query = write_query("natural language question")
results = run_query(query)
```

## Important Considerations
1. **RTL Support**: Frontend is Arabic (right-to-left)
2. **Session Management**: Redis stores conversation state
3. **Context Limits**: Messages trimmed to last 10 to avoid token limits
4. **Error Handling**: All nodes wrapped in try-except with RuntimeError
5. **CORS**: Enabled for local development

## Testing
- **test.py**, **test1.py**, **test2.py**, **test3.py**, **test4.py** - Various test scripts
- Test files located in `src/` directory

## File Structure Summary
```
survey_ai-1.1/
├── .env                      # Environment configuration
├── requirements.txt          # Python dependencies
├── ai-data-chat.html         # Frontend UI
├── charts.txt                # Chart types reference
├── sql.txt                   # SQL examples
├── Surveys.txt               # Survey metadata
├── data.csv                  # Sample data
├── src/
│   ├── app.py               # Flask API
│   ├── agents/
│   │   ├── graphs/
│   │   │   ├── workflow.py  # Main workflow
│   │   │   ├── nodes.py     # Agent nodes
│   │   │   └── setup.py     # State schema
│   │   └── prompt/          # All prompts
│   ├── data/
│   │   ├── models.py        # Data models
│   │   ├── connection.py    # DB connection
│   │   ├── operations.py    # DB operations
│   │   └── text_to_sql.py   # NL to SQL
│   ├── llms/
│   │   └── models.py        # LLM configurations
│   ├── helper/
│   │   ├── utils.py         # Utilities
│   │   ├── log.py           # Logging
│   │   └── tracer.py        # Langfuse tracer
│   └── test*.py             # Test files
```

## Tips for Claude Code
- When modifying agents, always update corresponding prompts
- Test changes with `src/test.py` or similar test files
- Check Langfuse traces for debugging LLM calls
- Arabic text should remain RTL in frontend
- Visualizations must return valid HTML strings
- Always handle JSON parsing errors from LLM responses