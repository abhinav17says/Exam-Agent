"""Notebook workflow, with upload and chat adapters for Streamlit.

Core functions come from uncommented notebook cells. Only file handling,
per-session ownership, API-key injection and UI result extraction are adapters.
"""
import json
import tempfile
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import FlashrankRerank
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv(Path(__file__).with_name(".env"))

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]


def prepare_chunks_with_headers(markdown_chunks):
    processed_docs = []
    for chunk in markdown_chunks:
        header = chunk.metadata.get('Header 2', 'General')
        content_with_header = f'SECTION: {header}\nCONTENT: {chunk.page_content}'
        doc = Document(page_content=content_with_header, metadata=chunk.metadata)
        processed_docs.append(doc)
    return processed_docs


def get_expanded_query(user_query, llm):
    query_gen_prompt = ChatPromptTemplate.from_template('You are a search optimizer. Rewrite the following user question to look like a document section heading or a technical specification title.\nUser Question: {question}\nOptimized Search Term:')
    chain = query_gen_prompt | llm
    response = chain.invoke({'question': user_query})
    return f'{user_query} {response.content}'


def setup_advanced_faiss_retriever(docs, api_key=None):
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    compressor = FlashrankRerank(top_n=10)
    base_retriever = vectorstore.as_retriever(search_kwargs={'k': 20})
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
    return compression_retriever


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_chatbot(md_header_splits, llm, api_key=None):
    """Scope the notebook's tool and graph to the uploaded document."""
    @tool
    def rag_tool(query):
        """
        Retrieve relevant information from the pdf document.
        Always use this tool to answer user's query. If the query is not related to the document, return "no context found related to your query". 
        """
        processed_docs = prepare_chunks_with_headers(md_header_splits)
        adv_retriever = setup_advanced_faiss_retriever(processed_docs, api_key=api_key)
        enhanced_query = get_expanded_query(query, llm)
        result = adv_retriever.invoke(enhanced_query)
        context = [processed_docs.page_content for processed_docs in result]
        metadata = [processed_docs.metadata for processed_docs in result]
        return {'query': query, 'context': context, 'metadata': metadata}

    tools = [rag_tool]
    llm_with_tools = llm.bind_tools(tools)

    def chat_node(state: ChatState):
        messages = state['messages']
        response = llm_with_tools.invoke(messages)
        return {'messages': [response]}

    tool_node = ToolNode(tools)
    graph = StateGraph(ChatState)
    graph.add_node('chat_node', chat_node)
    graph.add_node('tools', tool_node)
    graph.add_edge(START, 'chat_node')
    graph.add_conditional_edges('chat_node', tools_condition)
    graph.add_edge('tools', 'chat_node')
    return graph.compile()


class ExamAgent:
    """Adapt graph results to the existing Streamlit chat interface."""

    def __init__(self, chatbot, filename, chunk_count):
        self.chatbot = chatbot
        self.filename = filename
        self.chunk_count = chunk_count

    def ask(self, question, history):
        messages = [*history, HumanMessage(content=question)]
        result = self.chatbot.invoke({"messages": messages})
        sources = []
        for message in result["messages"][len(messages):]:
            if isinstance(message, ToolMessage) and message.name == "rag_tool":
                try:
                    payload = json.loads(message.content)
                except (TypeError, ValueError):
                    continue
                if isinstance(payload, dict):
                    sources.extend(
                        {"content": content, "metadata": metadata}
                        for content, metadata in zip(
                            payload.get("context", []), payload.get("metadata", [])
                        )
                    )
        return result["messages"][-1].content, sources


def process_uploaded_pdf(file_bytes, filename="document.pdf", api_key=None):
    """Replace the notebook's fixed PDF path with a temporary uploaded PDF."""
    from docling.document_converter import DocumentConverter

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        converter = DocumentConverter()
        loader = converter.convert(tmp_path)
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(loader.document.export_to_markdown())
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    llm = ChatOpenAI(model='gpt-4o-mini', api_key=api_key)
    chatbot = build_chatbot(md_header_splits, llm, api_key=api_key)
    return ExamAgent(chatbot, filename, len(md_header_splits))
