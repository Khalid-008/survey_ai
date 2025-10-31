from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from langchain_together import ChatTogether
from pydantic import SecretStr
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI


load_dotenv()

model_name = os.getenv("MODEL")
api_key = SecretStr(os.getenv("GOOGLE_API_KEY") or "")

# model = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=1,
#     max_tokens=65536,
#     timeout=None,
#     max_retries=2,
#     api_key=api_key,
#     top_p=0.95,
# )

model = ChatOpenAI(
    base_url="https://llmmux.channels-ai.online/v1",
    api_key="VSaNFvQ6eUWHw4y3oLrNDUnOiFbzEyUJfKhAAYeFwCuAr8X3BBCMyAlEMLclfFni",
    model="gpt-oss-120b"
)

pro_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=1,
    max_tokens=65536,
    timeout=None,
    max_retries=3,
    api_key=api_key,
    top_p=0.95,
)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1,
    max_tokens=65536,
    timeout=None,
    max_retries=2,
    api_key="AIzaSyDAyflsrZEK5PTD-ZcKEQc2ofcOtwzfQwc",
    top_p=0.95,
)

google_embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
