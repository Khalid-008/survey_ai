# import os
# import base64
# from langfuse import get_client
# from langfuse.langchain import CallbackHandler
# from opentelemetry import trace
# from opentelemetry.sdk.trace import TracerProvider
# from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
# from opentelemetry.sdk.trace.export import SimpleSpanProcessor
# from dotenv import load_dotenv

# load_dotenv()
# langFuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
# langFuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
# langFuse_host = os.getenv("LANGFUSE_HOST")

# os.environ["LANGFUSE_PUBLIC_KEY"] = langFuse_public_key
# os.environ["LANGFUSE_SECRET_KEY"] = langFuse_secret_key
# os.environ["LANGFUSE_HOST"] = langFuse_host

# def tracer_provider():
#     langfuse_client = get_client()
    
#     if langfuse_client.auth_check():
#         print("Langfuse client is authenticated and ready!")
#     else:
#         print("Authentication failed. Please check your credentials and host.")

#     LANGFUSE_AUTH = base64.b64encode(
#         f"{os.environ.get('LANGFUSE_PUBLIC_KEY')}:{os.environ.get('LANGFUSE_SECRET_KEY')}".encode()
#     ).decode()

#     langfuse_host = os.environ.get("LANGFUSE_HOST")
#     if langfuse_host is None:
#         raise ValueError("LANGFUSE_HOST environment variable is not set")

#     os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = langfuse_host + "/api/public/otel"
#     os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

#     trace_provider = TracerProvider()
#     trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

#     trace.set_tracer_provider(trace_provider)

#     langfuse_handler = CallbackHandler()
    
#     return langfuse_handler
