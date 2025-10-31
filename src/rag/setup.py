# # from rag.prompts import rag_prompt
# from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_ollama import ChatOllama
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langgraph.graph import START, StateGraph
# from typing_extensions import List, TypedDict
# from llms.models import embeddings_model


# # # Define the application state
# # class State(TypedDict):
# #     model: ChatOllama
# #     question: str
# #     context: List[Document]
# #     answer: str
# #     docs: List[Document]
# #     chunk_overlap: int
# #     chunk_size: int

# # Create vector store
# vector_store = InMemoryVectorStore(embeddings_model)

# # Step 1: Split and index
# def initiate_setup():
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200
#     )
#     all_splits = text_splitter.split_documents(state["docs"])
#     _ = vector_store.add_documents(documents=all_splits)
#     return {}

# # Step 2: Retrieve relevant docs
# def retrieve(state: State):
#     retrieved_docs = vector_store.similarity_search(state["messages"])
#     return {"context": retrieved_docs}

# # Step 3: Generate response
# # from langchain_core.messages import SystemMessage, HumanMessage
# # from langchain_core.prompts import ChatPromptTemplate


# def generate(state: State):
#     docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
#     messages = rag_prompt.invoke(
#         {
#             "question": state['question'],
#             "context": docs_content
#         }
#     )
    
#     response = state["model"].invoke(messages)
#     answer_content = getattr(response, "content", response)
#     return {"answer": answer_content}


# # Build and run the graph
# def call_model_with_rag(model : ChatOllama, question: str, document: List[Document], chunk_size: int, chunk_overlap: int):
#     graph_builder = StateGraph(State).add_sequence([initiate_setup, retrieve, generate])
#     graph_builder.add_edge(START, "initiate_setup")
#     graph = graph_builder.compile()

#     response = graph.invoke({
#         "model": model,
#         "question": question,
#         "docs": document,
#         "chunk_size": chunk_size,
#         "chunk_overlap": chunk_overlap,
#         "context": [],
#         "answer": "",
#     })

#     return response["answer"]