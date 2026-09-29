"""Notebook workflow, with upload and chat adapters for Streamlit.

Core functions come from uncommented notebook cells. Only file handling,
per-session ownership, API-key injection and UI result extraction are adapters.
"""
import json
import gc
import logging
import tempfile
from pathlib import Path
from threading import Lock
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
_conversion_lock = Lock()
logger = logging.getLogger(__name__)

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


def convert_pdf_to_markdown(file_bytes, low_memory=False):
    """Run Docling once at a time to avoid overlapping model allocations."""
    with _conversion_lock:
        try:
            return _convert_pdf_to_markdown(file_bytes, low_memory)
        finally:
            # Release converter/pipeline cycles before chat loads its reranker.
            gc.collect()


def _convert_pdf_to_markdown(file_bytes, low_memory):
    from docling.document_converter import DocumentConverter, PdfFormatOption

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        if low_memory:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.accelerator_options import AcceleratorOptions
            from docling.datamodel.pipeline_options import PdfPipelineOptions

            options = PdfPipelineOptions(
                do_ocr=False,
                do_table_structure=True,
                layout_batch_size=1,
                table_batch_size=1,
                ocr_batch_size=1,
                queue_max_size=2,
                accelerator_options=AcceleratorOptions(device="cpu", num_threads=1),
            )
            converter = DocumentConverter(format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options)
            })
        else:
            converter = DocumentConverter()
        logger.warning("PDF conversion starting (low_memory=%s)", low_memory)
        loader = converter.convert(tmp_path)
        markdown = loader.document.export_to_markdown()
        logger.warning("PDF conversion finished (%s characters)", len(markdown))
        return markdown
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def process_uploaded_markdown(file_bytes, filename="document.md", api_key=None):
    """Accept locally exported Docling Markdown without loading PDF models."""
    try:
        markdown = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Please upload a Markdown file saved as UTF-8.") from exc
    return _agent_from_markdown(markdown, filename, api_key)


def _agent_from_markdown(markdown, filename, api_key):
    if not markdown.strip():
        raise ValueError("No readable text was found. For a scanned PDF, turn off lower-memory mode "
                         "or convert it locally and upload the exported Markdown.")
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown)
    if not md_header_splits:
        raise ValueError("No searchable sections were found in this document.")
    llm = ChatOpenAI(model='gpt-4o-mini', api_key=api_key)
    chatbot = build_chatbot(md_header_splits, llm, api_key=api_key)
    return ExamAgent(chatbot, filename, len(md_header_splits))


def process_uploaded_pdf(file_bytes, filename="document.pdf", api_key=None, *, low_memory=False):
    """Default to the notebook converter; optionally reduce PDF model memory."""
    markdown = convert_pdf_to_markdown(file_bytes, low_memory=low_memory)
    return _agent_from_markdown(markdown, filename, api_key)
