import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()

model_name = os.getenv("MODEL")
api_key = SecretStr(os.getenv("GOOGLE_API_KEY") or "")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")


# model = ChatAnthropic(
#     model="claude-sonnet-4-5-20250929",
#     api_key=anthropic_api_key,
#     temperature=1,
#     max_tokens=8192
# )

model = ChatOpenAI(
    base_url="https://llmmux.channels-ai.online/v1",
    api_key="VSaNFvQ6eUWHw4y3oLrNDUnOiFbzEyUJfKhAAYeFwCuAr8X3BBCMyAlEMLclfFni",
    model="gpt-oss-120b"
)

# model = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=1,
#     max_tokens=65536,
#     timeout=None,
#     max_retries=2,
#     api_key="AIzaSyDAyflsrZEK5PTD-ZcKEQc2ofcOtwzfQwc",
#     top_p=0.95,
# )


google_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1,
    max_tokens=65536,
    timeout=None,
    max_retries=2,
    api_key="AIzaSyDAyflsrZEK5PTD-ZcKEQc2ofcOtwzfQwc",
    top_p=0.95,
)

google_embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
