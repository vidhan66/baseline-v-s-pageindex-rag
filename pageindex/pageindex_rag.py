from pageindex import PageIndexClient
import pageindex.utils as utils
from dotenv import load_dotenv
import os
import openai
import json
import asyncio
import logging
import subprocess
import sys
import time
import hashlib
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    DEFAULT_PDF_PATH = r"C:\Users\vidha\Downloads\The Art of Persuasion by Bob Burg.pdf"

load_dotenv()

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _default_cache_path_for_pdf(pdf_path: str) -> str:
    """Return a stable, per-PDF path for the PageIndex tree cache file.

    Mirrors the logic in baseline_rag._default_vectorstore_path_for_pdf so
    both pipelines use the same hashing scheme and root directory.
    """
    key = hashlib.sha256(os.path.abspath(pdf_path).encode("utf-8")).hexdigest()[:16]
    base_dir = Path(__file__).resolve().parent.parent / "data" / "pageindex_cache"
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir / f"{key}.json")


def _load_cache(cache_path: str) -> dict:
    """Load the JSON cache written by prepare_pageindex_index."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"PageIndex cache not found at: {cache_path}. "
            "Run prepare_pageindex_index() first."
        )
    with open(cache_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


async def call_llm(prompt, model="gpt-4o-mini", temperature=0):
    logger.info("Calling LLM model=%s, prompt_chars=%d", model, len(prompt))
    response = await oa_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    text = (response.choices[0].message.content or "").strip()
    logger.info("LLM response received, output_chars=%d", len(text))
    return text


# ---------------------------------------------------------------------------
# prepare_pageindex_index  (one-time, offline step — mirrors prepare_baseline_index)
# ---------------------------------------------------------------------------

def prepare_pageindex_index(
    pdf_path: str,
    cache_path: str | None = None,
    max_wait_seconds: int = 300,
    poll_interval_seconds: int = 5,
) -> dict:
    """Submit the PDF to PageIndex, wait until retrieval is ready, then
    persist the document tree + doc_id to a local JSON cache file.

    This is the equivalent of baseline_rag.prepare_baseline_index():
    it handles all the slow, one-time work (submission, processing, tree
    fetch) so that run_pageindex_query_cached() only has to do the
    per-query LLM calls — making latency and token counts directly
    comparable to run_baseline_query_cached().

    Returns
    -------
    dict with keys:
        ok              – True on success
        doc_id          – PageIndex document id
        cache_path      – where the cache was written
        node_count      – total nodes in the fetched tree
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pageindex_api_key = os.getenv("PAGEINDEX_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    if not pageindex_api_key:
        raise ValueError("PAGEINDEX_API_KEY not set in environment")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    resolved_cache_path = cache_path or _default_cache_path_for_pdf(pdf_path)
    pi_client = PageIndexClient(api_key=pageindex_api_key)

    # ── 1. Submit document (or reuse existing doc_id from env) ──────────────
    existing_doc_id = (os.getenv("PAGEINDEX_DOC_ID") or "").strip()
    if existing_doc_id:
        doc_id = existing_doc_id
        logger.info("prepare: reusing PAGEINDEX_DOC_ID=%s (skipping submission)", doc_id)
    else:
        logger.info("prepare: submitting document %s", pdf_path)
        submit_resp = pi_client.submit_document(pdf_path)
        logger.info("prepare: submit response type=%s value=%s", type(submit_resp).__name__, submit_resp)

        doc_id = None
        if isinstance(submit_resp, dict):
            doc_id = (
                submit_resp.get("doc_id")
                or submit_resp.get("document_id")
                or (submit_resp.get("result", {}) or {}).get("doc_id")
                or (submit_resp.get("data", {}) or {}).get("doc_id")
            )
        elif isinstance(submit_resp, str):
            doc_id = submit_resp.strip()

        if not doc_id or not isinstance(doc_id, str):
            raise ValueError(f"Could not extract doc_id from submit response: {submit_resp}")
        logger.info("prepare: document submitted, doc_id=%s", doc_id)

    # ── 2. Poll until retrieval is ready ────────────────────────────────────
    elapsed = 0
    while elapsed < max_wait_seconds:
        if pi_client.is_retrieval_ready(doc_id):
            logger.info("prepare: retrieval ready after %ds", elapsed)
            break
        logger.info("prepare: waiting for processing... elapsed=%ds", elapsed)
        time.sleep(poll_interval_seconds)          # blocking is fine here (one-time setup)
        elapsed += poll_interval_seconds
    else:
        raise TimeoutError(f"PageIndex not ready within {max_wait_seconds}s for doc_id={doc_id}")

    # ── 3. Fetch the full tree (with node text + summaries) ─────────────────
    logger.info("prepare: fetching document tree")
    tree = pi_client.get_tree(doc_id, node_summary=True)["result"]
    node_map = utils.create_node_mapping(tree)
    node_count = len(node_map)
    logger.info("prepare: tree fetched, %d nodes", node_count)

    # ── 4. Persist everything to the local cache ────────────────────────────
    cache_payload = {
        "doc_id": doc_id,
        "pdf_path": os.path.abspath(pdf_path),
        "tree": tree,
        "cached_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(resolved_cache_path, "w", encoding="utf-8") as fh:
        json.dump(cache_payload, fh, ensure_ascii=False, indent=2)
    logger.info("prepare: cache written to %s", resolved_cache_path)

    return {
        "ok": True,
        "doc_id": doc_id,
        "cache_path": resolved_cache_path,
        "node_count": node_count,
    }


# ---------------------------------------------------------------------------
# run_pageindex_query_cached  (hot path — mirrors run_baseline_query_cached)
# ---------------------------------------------------------------------------

async def run_pageindex_query_cached(
    pdf_path: str,
    query: str,
    memory: List[str] | None = None,
    cache_path: str | None = None,
) -> dict:
    """Answer a query using a pre-built PageIndex cache (no submission, no polling).

    This is the direct equivalent of baseline_rag.run_baseline_query_cached():
    it skips all indexing overhead so that elapsed_ms and token_usage reflect
    only the retrieval + generation work, making benchmarks fair.

    The cache must have been created by prepare_pageindex_index() beforehand.
    """
    start = time.perf_counter()
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    openai_api_key = os.getenv("OPENAI_API_KEY")
    pageindex_api_key = os.getenv("PAGEINDEX_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    if not pageindex_api_key:
        raise ValueError("PAGEINDEX_API_KEY not set in environment")

    def _accumulate_usage(resp):
        usage = resp.usage
        if usage:
            token_usage["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            token_usage["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
            token_usage["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)

    # ── 1. Load pre-built cache ──────────────────────────────────────────────
    resolved_cache_path = cache_path or _default_cache_path_for_pdf(pdf_path)
    cache = _load_cache(resolved_cache_path)
    doc_id = cache["doc_id"]
    tree = cache["tree"]
    logger.info("cached query: loaded tree from cache (doc_id=%s)", doc_id)

    tree_without_text = utils.remove_fields(tree.copy(), fields=["text"])
    node_map = utils.create_node_mapping(tree)

    oa_client = openai.AsyncOpenAI(api_key=openai_api_key)

    # ── 2. Tree search: find relevant nodes ──────────────────────────────────
    search_prompt = f"""
You are given a question and a tree structure of a document.
Each node contains a node id, node title, and a corresponding summary.
Find nodes likely to contain the answer.

Question: {query}
Document tree structure:
{json.dumps(tree_without_text, indent=2)}

Return JSON only:
{{"node_list": ["node_id_1", "node_id_2"]}}
"""
    search_resp = await oa_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": search_prompt}],
        temperature=0,
    )
    _accumulate_usage(search_resp)
    tree_search_result = (search_resp.choices[0].message.content or "").strip()
    try:
        tree_search_result_json = json.loads(tree_search_result)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON from tree search: %s | raw: %s", exc, tree_search_result)
        raise
    node_list = tree_search_result_json.get("node_list", [])

    # ── 3. Assemble context from retrieved nodes ─────────────────────────────
    relevant_content = "\n\n".join(
        node_map[node_id]["text"] for node_id in node_list if node_id in node_map
    )
    context_chars = len(relevant_content)
    logger.info(
        "cached query: retrieved %d nodes, %d context chars (≈%d tokens)",
        len(node_list), context_chars, context_chars // 4,
    )

    # ── 4. Generate answer ───────────────────────────────────────────────────
    memory_text = "\n".join(memory or []) if memory else "No previous conversation."
    answer_prompt = f"""
Answer the question based on the context and conversation memory:
Question: {query}
Conversation memory:
{memory_text}
Context:
{relevant_content}
Provide a clear concise answer based only on context.
"""
    answer_resp = await oa_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": answer_prompt}],
        temperature=0,
    )
    _accumulate_usage(answer_resp)
    answer = (answer_resp.choices[0].message.content or "").strip()

    return {
        "ok": True,
        "answer": answer,
        "doc_id": doc_id,
        "token_usage": token_usage,
        "retrieval_usage": {
            "retrieved_nodes": len(node_list),
            "context_chars": context_chars,
            "estimated_context_tokens": context_chars // 4,
        },
        "elapsed_ms": int((time.perf_counter() - start) * 1000),
    }


# ---------------------------------------------------------------------------
# run_pageindex_query  (original — submit + poll every call, kept for compat)
# ---------------------------------------------------------------------------

async def run_pageindex_query(pdf_path: str, query: str, memory: List[str] | None = None) -> dict:
    start = time.perf_counter()
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pageindex_api_key = os.getenv("PAGEINDEX_API_KEY")

    def _accumulate_usage(resp):
        usage = resp.usage
        if usage:
            token_usage["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            token_usage["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
            token_usage["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    if not pageindex_api_key:
        raise ValueError("PAGEINDEX_API_KEY not set in environment")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")
    pi_client = PageIndexClient(api_key=pageindex_api_key)
    oa_client = openai.AsyncOpenAI(api_key=openai_api_key)

    existing_doc_id = (os.getenv("PAGEINDEX_DOC_ID") or "").strip()
    if existing_doc_id:
        doc_id = existing_doc_id
    else:
        submit_resp = pi_client.submit_document(pdf_path)
        doc_id = submit_resp["doc_id"] if isinstance(submit_resp, dict) else str(submit_resp)
    max_wait_seconds = int(os.getenv("PAGEINDEX_READY_TIMEOUT_SEC", "300"))
    poll_interval_seconds = int(os.getenv("PAGEINDEX_READY_POLL_SEC", "5"))
    elapsed = 0
    while elapsed < max_wait_seconds:
        if pi_client.is_retrieval_ready(doc_id):
            break
        await asyncio.sleep(poll_interval_seconds)
        elapsed += poll_interval_seconds
    else:
        raise TimeoutError(f"PageIndex not ready within {max_wait_seconds}s for doc_id={doc_id}")

    tree = pi_client.get_tree(doc_id, node_summary=True)["result"]
    tree_without_text = utils.remove_fields(tree.copy(), fields=["text"])
    node_map = utils.create_node_mapping(tree)

    search_prompt = f"""
You are given a question and a tree structure of a document.
Each node contains a node id, node title, and a corresponding summary.
Find nodes likely to contain the answer.

Question: {query}
Document tree structure:
{json.dumps(tree_without_text, indent=2)}

Return JSON only:
{{"node_list": ["node_id_1", "node_id_2"]}}
"""
    search_resp = await oa_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": search_prompt}],
        temperature=0,
    )
    _accumulate_usage(search_resp)
    tree_search_result = (search_resp.choices[0].message.content or "").strip()
    tree_search_result_json = json.loads(tree_search_result)
    node_list = tree_search_result_json.get("node_list", [])

    relevant_content = "\n\n".join(
        node_map[node_id]["text"] for node_id in node_list if node_id in node_map
    )
    memory_text = "\n".join(memory or []) if memory else "No previous conversation."
    answer_prompt = f"""
Answer the question based on the context and conversation memory:
Question: {query}
Conversation memory:
{memory_text}
Context:
{relevant_content}
Provide a clear concise answer based only on context.
"""
    answer_resp = await oa_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": answer_prompt}],
        temperature=0,
    )
    _accumulate_usage(answer_resp)
    answer = (answer_resp.choices[0].message.content or "").strip()
    return {
        "ok": True,
        "answer": answer,
        "doc_id": doc_id,
        "token_usage": token_usage,
        "elapsed_ms": int((time.perf_counter() - start) * 1000),
    }


# ---------------------------------------------------------------------------
# CLI entry-point (unchanged behaviour, now uses cached path when available)
# ---------------------------------------------------------------------------

async def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set in environment")
    if not os.getenv("PAGEINDEX_API_KEY"):
        raise ValueError("PAGEINDEX_API_KEY not set in environment")

    pdf_path = os.getenv("PDF_PATH", Config.DEFAULT_PDF_PATH)
    query = os.getenv("RAG_QUERY", "Summarize the section montivating the unmotivated.")

    # Auto-use cached path if a cache already exists for this PDF
    resolved_cache_path = _default_cache_path_for_pdf(pdf_path)
    if os.path.exists(resolved_cache_path):
        logger.info("Cache found at %s — using run_pageindex_query_cached", resolved_cache_path)
        result = await run_pageindex_query_cached(pdf_path, query)
    else:
        logger.info("No cache found — running prepare_pageindex_index first")
        prepare_pageindex_index(pdf_path)
        result = await run_pageindex_query_cached(pdf_path, query)

    print(f"\n{'='*60}")
    print(f"Query:  {query}")
    print(f"Answer: {result['answer']}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    asyncio.run(main())