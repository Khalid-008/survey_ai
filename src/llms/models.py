import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
llm_mux_url = os.getenv("LLM_MUX_URL")
llm_mux_api_key = os.getenv("LLM_MUX_API_KEY")
qwen3_mux_url = os.getenv("QWEN3_LLM_MUX_URL")
qwen3_mux_api_key = os.getenv("QWEN3_LLM_MUX_API_KEY")

# Primary model (GPT-OSS via LLM Mux)
model = ChatOpenAI(
    base_url=llm_mux_url,
    api_key=llm_mux_api_key,
    model="gpt-oss-120b",
    timeout=60,
    max_retries=2,
    max_tokens=4096,
)

qwen3_model = ChatOpenAI(
    base_url=qwen3_mux_url,
    api_key=qwen3_mux_api_key,
    model="Qwen/Qwen3.5-27B-FP8",
    timeout=300,
    max_retries=2,
    max_tokens=16384,
)

# Alternative Google model
google_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1,
    max_tokens=65536,
    timeout=60,
    max_retries=2,
    api_key=google_api_key,
    top_p=0.95,
)

google_embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", api_key=google_api_key
)
