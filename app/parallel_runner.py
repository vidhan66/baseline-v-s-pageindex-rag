import asyncio
import importlib.util
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_function_from_file(file_path: Path, function_name: str):
    """Dynamically load a named function from a file path."""
    spec = importlib.util.spec_from_file_location(f"dyn_{file_path.stem}", str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, function_name, None)
    if fn is None:
        raise AttributeError(f"{function_name} not found in {file_path}")
    return fn


# ---------------------------------------------------------------------------
# Baseline helpers
# ---------------------------------------------------------------------------

def prepare_baseline_index_in_process(
    pdf_path: str,
    openai_api_key: str,
    baseline_vectorstore_path: str,
) -> dict:
    """One-time baseline index build (embeddings → Chroma).  Runs in a worker process."""
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    prepare_baseline_index = _load_function_from_file(
        ROOT_DIR / "baseline" / "baseline_rag.py",
        "prepare_baseline_index",
    )
    return prepare_baseline_index(pdf_path, baseline_vectorstore_path)


def run_baseline_in_process(
    pdf_path: str,
    question: str,
    memory: list[str],
    openai_api_key: str,
    baseline_vectorstore_path: str,
) -> dict:
    """Per-query baseline retrieval + generation.  Runs in a worker process."""
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    run_baseline_query_cached = _load_function_from_file(
        ROOT_DIR / "baseline" / "baseline_rag.py",
        "run_baseline_query_cached",
    )
    return asyncio.run(
        run_baseline_query_cached(
            pdf_path,
            question,
            memory,
            baseline_vectorstore_path,
        )
    )


# ---------------------------------------------------------------------------
# PageIndex helpers  (NEW — previously missing, causing ImportError in streamlit_app.py)
# ---------------------------------------------------------------------------

def prepare_pageindex_index_in_process(
    pdf_path: str,
    openai_api_key: str,
    pageindex_api_key: str,
    pageindex_cache_path: str,
) -> dict:
    """One-time PageIndex setup: submit PDF → poll until ready → fetch tree → write cache.

    Mirrors prepare_baseline_index_in_process so the Streamlit app can call
    both in symmetric spinners before any questions are asked.
    Runs in a worker process to avoid blocking the Streamlit event loop.
    """
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if pageindex_api_key:
        os.environ["PAGEINDEX_API_KEY"] = pageindex_api_key

    prepare_pageindex_index = _load_function_from_file(
        ROOT_DIR / "pageindex" / "pageindex_rag.py",
        "prepare_pageindex_index",
    )
    return prepare_pageindex_index(pdf_path, pageindex_cache_path)


def run_pageindex_cached_in_process(
    pdf_path: str,
    question: str,
    memory: list[str],
    openai_api_key: str,
    pageindex_api_key: str,
    pageindex_cache_path: str,
) -> dict:
    """Per-query PageIndex retrieval + generation using a pre-built tree cache.

    Replaces the old run_pageindex_in_process which re-submitted and re-polled
    on every single query, making latency comparisons completely unfair.
    Runs in a worker process to avoid blocking the Streamlit event loop.
    """
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if pageindex_api_key:
        os.environ["PAGEINDEX_API_KEY"] = pageindex_api_key

    run_pageindex_query_cached = _load_function_from_file(
        ROOT_DIR / "pageindex" / "pageindex_rag.py",
        "run_pageindex_query_cached",
    )
    return asyncio.run(
        run_pageindex_query_cached(
            pdf_path,
            question,
            memory,
            pageindex_cache_path,
        )
    )


# ---------------------------------------------------------------------------
# Legacy helper kept for backward-compatibility (not used by streamlit_app.py)
# ---------------------------------------------------------------------------

def run_pageindex_in_process(
    pdf_path: str,
    question: str,
    memory: list[str],
    openai_api_key: str,
    pageindex_api_key: str,
    pageindex_doc_id: str,
) -> dict:
    """Original per-query runner that re-submits + re-polls on every call.

    Kept for backward-compatibility only.  Prefer run_pageindex_cached_in_process
    for any new code.
    """
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if pageindex_api_key:
        os.environ["PAGEINDEX_API_KEY"] = pageindex_api_key
    if pageindex_doc_id:
        os.environ["PAGEINDEX_DOC_ID"] = pageindex_doc_id
    else:
        os.environ.pop("PAGEINDEX_DOC_ID", None)
    run_pageindex_query = _load_function_from_file(
        ROOT_DIR / "pageindex" / "pageindex_rag.py",
        "run_pageindex_query",
    )
    return asyncio.run(run_pageindex_query(pdf_path, question, memory))