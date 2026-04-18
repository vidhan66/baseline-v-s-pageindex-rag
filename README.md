# RAG Comparison: Baseline vs PageIndex

A side-by-side benchmarking app that runs two fundamentally different RAG pipelines on the same PDF and compares their answers, token usage, retrieval quality, and latency in real time.

## What Are These Two Approaches?

### Baseline RAG (Vector / Chunk-based)
The classical RAG pipeline. The PDF is split into fixed-size overlapping text chunks, each chunk is embedded into a vector using OpenAI embeddings, and all vectors are stored in a local [Chroma](https://www.trychroma.com/) vector database. At query time, the question is embedded and the most semantically similar chunks are retrieved using MMR (Maximal Marginal Relevance) search. Those chunks form the context passed to GPT-4o-mini.

**How it retrieves:** Similarity search — "which chunks are closest to the question in embedding space?"

### PageIndex RAG (Tree / Reasoning-based)
A vectorless approach. PageIndex processes the document into a **hierarchical semantic tree** — a structured outline enriched with summaries and full node text. At query time, the tree (without node text) is passed to GPT-4o-mini, which reasons over the structure to identify which nodes likely contain the answer. Only those nodes' text is then used as context for the final answer.

**How it retrieves:** Reasoning-based tree search — "which sections of this document, by title and summary, are relevant to this question?"

## Why Caching?

The original implementation re-built the Baseline vector index and re-submitted the PDF to PageIndex on every single query. This caused the Baseline pipeline to time out regularly due to the cost of embedding all chunks at query time, and wasted PageIndex credits re-submitting the same document repeatedly.

The fix separates indexing from retrieval. Both pipelines now have a one-time **prepare** step that runs when you click "Index PDF":

- **Baseline:** chunks the PDF, generates embeddings, and persists the Chroma vectorstore to disk.
- **PageIndex:** submits the PDF to the PageIndex cloud API, polls until the tree is built, fetches it, and writes it to a local `tree_cache.json`.

From that point on, every query only does retrieval and generation work. **The latency and token numbers shown in the comparison reflect pure retrieval performance only** not indexing overhead which is what actually matters for a fair benchmark.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        streamlit_app.py                         │
│                     (Main UI + Orchestration)                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │    parallel_runner.py   │
              │  (Process pool bridge)  │
              └──────┬──────────┬───────┘
                     │          │
       ┌─────────────▼──┐   ┌───▼──────────────┐
       │  baseline_rag  │   │  pageindex_rag   │
       │  (Worker proc) │   │  (Worker proc)   │
       └────────┬───────┘   └────────┬─────────┘
                │                    │
       ┌────────▼───────┐   ┌────────▼─────────┐
       │  Chroma (local)│   │ tree_cache.json   │
       │  vector store  │   │ (local, pre-built)│
       └────────┬───────┘   └────────┬─────────┘
                │                    │
       ┌────────▼────────────────────▼─────────┐
       │           OpenAI GPT-4o-mini           │
       │      (answer generation for both)      │
       └───────────────────────────────────────┘
```

### Indexing Phase (one-time, on "Index PDF" click)

```
PDF Upload
    │
    ├──► Baseline: PyPDFLoader → chunk (1000 chars, 200 overlap)
    │              → OpenAI Embeddings → Chroma DB persisted to /tmp
    │
    └──► PageIndex: submit_document() → poll is_retrieval_ready()
                    → get_tree(node_summary=True) → tree_cache.json
```

### Query Phase (on "Ask" click — retrieval only, both run concurrently)

```
Question
    │
    ├──► Baseline Worker: load Chroma → MMR retrieval (k=8, fetch_k=30)
    │                     → prompt → GPT-4o-mini → answer + stats
    │
    └──► PageIndex Worker: load tree_cache.json → GPT-4o-mini tree search
                           → assemble node text → GPT-4o-mini answer + stats
    │
    └──► Side-by-side results: answer | tokens | retrieval stats | latency
```

Both pipelines run in separate worker processes via `ProcessPoolExecutor`. This is necessary because Streamlit's main thread owns an event loop that conflicts with `asyncio.run()`, and it ensures a crash in one pipeline never affects the other.

## Tech Stack

| Layer | Baseline | PageIndex |
|---|---|---|
| PDF Loading | `pypdf` via LangChain | PageIndex Cloud API |
| Chunking | `RecursiveCharacterTextSplitter` | Semantic tree (built by PageIndex) |
| Embeddings | `OpenAIEmbeddings` (text-embedding-ada-002) | None (vectorless) |
| Vector Store | `Chroma` (local, persisted to `/tmp`) | None |
| Retrieval | MMR similarity search | LLM tree reasoning |
| LLM | GPT-4o-mini | GPT-4o-mini (×2 calls: tree search + answer) |
| Orchestration | LangChain | Direct OpenAI async client |

| App Component | Technology |
|---|---|
| UI | Streamlit |
| Concurrency | `ProcessPoolExecutor` + `asyncio.gather` |
| Config | `python-dotenv` |
| PageIndex SDK | `pageindex` (Python SDK) |
| OpenAI SDK | `openai` (async) |

## Benchmark Results

Tested on small PDFs (1–5 pages). All numbers are **retrieval + generation only** — indexing is cached upfront and excluded from measurement.

**1-page PDF**

| | Baseline | PageIndex |
|---|---|---|
| Prompt tokens | 940 | 797 |
| Completion tokens | 308 | 102 |
| Total tokens | 1,248 | 899 |
| Latency | 13,028ms | 5,286ms |
| Retrieved | 4 chunks, ~955 ctx tokens | 2 nodes, 491 chars, ~122 ctx tokens |

**5-page PDF**

| | Baseline | PageIndex |
|---|---|---|
| Prompt tokens | 2,045 | 2,410 |
| Completion tokens | 240 | 217 |
| Total tokens | 2,285 | 2,627 |
| Latency | 8,926ms | 5,812ms |
| Retrieved | 8 chunks, ~1,406 ctx tokens | 7 nodes, 4,660 chars, ~1,165 ctx tokens |

**Takeaways:**

PageIndex is faster at query time across both tests. On the 1-page PDF it also used fewer tokens since the tree was small and node selection precise. On the 5-page PDF the token counts flipped — PageIndex used slightly more because the full tree structure adds overhead to the search prompt.

Answer character differed more than accuracy. Baseline produced detailed, structured answers that stay close to the source text — better for in-depth reading or structured extraction. PageIndex produced shorter, more summarised answers — better for quick review or revision of a document. Neither was strictly more accurate; they suit different use cases.


## System Constraints & Failure Modes

**`{"detail":"LimitReached"}` from PageIndex** — you've hit the free tier's 200-credit or 200 active-page cap. Delete old documents at [dash.pageindex.ai](https://dash.pageindex.ai) to free up active pages. Set `PAGEINDEX_DOC_ID` in your `.env` to reuse an already-submitted document and skip submission entirely. Avoid clicking "Index PDF" multiple times on the same file.

**`TimeoutError: PageIndex not ready within 300s`** — PDF is too large or cloud processing is slow. Keep PDFs under 20 pages on the free tier. Increase `max_wait_seconds` in `prepare_pageindex_index()` if you need to wait longer.

**Chroma `InvalidCollectionException`** — vectorstore on disk is stale or was built with a different model. Delete `/tmp/rag_cmp_baseline_index/` and re-index.

**`json.JSONDecodeError` in PageIndex query** — GPT-4o-mini occasionally returns malformed JSON for the node list. Retry the question.

**Baseline returns empty or off-topic answer** — MMR retrieved chunks that don't contain the answer, common with vague questions or very short PDFs. Make the question more specific.

## Setup Instructions

### Prerequisites
- Python 3.11+
- An **OpenAI API key** — [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- A **PageIndex API key** — see below

### Getting a PageIndex API Key

1. Sign up at [dash.pageindex.ai](https://dash.pageindex.ai).
2. Go to **API Keys** in the sidebar → **Create API Key**. Copy it immediately.
3. The free tier gives you **200 credits** total — 1 credit per page indexed, one-time. That's roughly 4–5 small PDFs.

> **Use small PDFs only on the free tier** — ideally under 20 pages. A single large document can burn your entire free balance in one go. Delete test documents from the dashboard after use to recover your active page count.

### Installation

```bash
git clone <your-repo-url>
cd <project-root>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```
### Configure Environment Variables

Create a `.env` file in the project root:

```env
# Required
OPENAI_API_KEY=sk-...
PAGEINDEX_API_KEY=pi-...

# Optional
# BASELINE_TIMEOUT_SEC=45
# PAGEINDEX_TIMEOUT_SEC=60
```
API keys can also be entered in the app sidebar — they override `.env` for that session.

### Run the App

```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`. Upload a PDF, click **Index PDF** (one-time setup for both pipelines), then ask questions.

## Limitations

- Conversation memory lives in Streamlit session state only — resets on page refresh.
- The Chroma vectorstore is local and not portable across machines — re-index after moving environments.
- Both pipelines use GPT-4o-mini. Change the model string in each pipeline's `Config` class to swap models.
- PageIndex free tier is suitable for small documents only. 