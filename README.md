# Exam Agent

A Streamlit frontend for the PDF question-answering workflow in `exam_agent.ipynb`.
The notebook remains available for experiments; the app runs without executing notebook cells.

## Run locally (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run exam_agent_frontend.py
```

Open http://localhost:8501. Put `OPENAI_API_KEY=your-key` in a local `.env` file,
configure it in Streamlit secrets, or enter it in the app's password field.
Do not commit API keys.

1. Upload a PDF or select the included CDS notification.
2. Click **Process document** and wait for indexing to finish.
3. Ask questions, inspect numbered supporting excerpts, and download your conversation.

Processing another document starts a new conversation. Each browser session owns its
index and history in memory; restarting the server or ending the session discards them.
Selecting a new file does not switch the active index until processing succeeds.

## How it works

`exam_agent_backend.py` follows the uncommented workflow in `exam_agent.ipynb`:
Docling conversion, MarkdownHeaderTextSplitter for H1/H2/H3 headings, the
`Header 2` (or `General`) content prefix, the original query expansion prompt,
FAISS retrieval (`k=20`), FlashRank reranking (`top_n=10`), and the LangGraph
chat/tool loop. There is no recursive character splitting or extra answer prompt.

As in the notebook, each `rag_tool` call creates the FAISS index and reranker.
This repeats embedding work and API costs. The model decides whether to call
the tool. Retrieved excerpts are shown when it does; answer citation numbering
is not enforced by the notebook's prompt.

The app adds temporary upload handling, session ownership, API-key injection,
and extraction of tool results for the UI. Notebook inspection statements,
unused intermediate indexes and the hard-coded example question are omitted.

The first retrieval may download local parsing/reranking models. Internet access
and an OpenAI key with available quota are required. Document text is sent to
OpenAI for embeddings and answers; API charges apply. Excerpts identify sections,
not PDF page numbers.

## PDF processing on hosts with limited memory

The UI defaults to **Lower-memory PDF processing**: OCR is disabled, stage batches
are reduced to one, queues are bounded, and CPU inference uses one thread. Table
structure extraction stays enabled. This is intended for PDFs with selectable text;
scanned pages and text inside images will not be read. Turn it off to use the
notebook's default Docling conversion. Header splitting and retrieval are unchanged.
PDF conversions are serialized within the server process to avoid loading multiple
sets of models concurrently. Even this mode may exceed a free host's memory limit.

If the app terminates during model loading, convert the PDF on your local machine:

```powershell
.\.venv\Scripts\python.exe export_pdf.py noti_cds.pdf
```

Upload the resulting `noti_cds.md` to the hosted app. The export uses Docling's
original defaults and Markdown export, preserving the notebook's input to
MarkdownHeaderTextSplitter, including extracted tables. The hosted Markdown path
does not load Docling's PDF models. It still runs embeddings, FAISS and FlashRank
when the agent calls its retrieval tool, so hosting capacity must still be checked.

An abrupt stop without a traceback during model loading suggests a resource kill,
but does not prove it. Check hosting resource metrics or platform messages. The
Hugging Face unauthenticated-download warning alone is not a conversion failure.
