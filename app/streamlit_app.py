import asyncio
from concurrent.futures import ProcessPoolExecutor
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import List
import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from app.core.config import AppConfig
    from app.core.models import MemoryTurn
    from app.parallel_runner import (
        prepare_baseline_index_in_process,
        prepare_pageindex_index_in_process,
        run_baseline_in_process,
        run_pageindex_cached_in_process,
    )
except ModuleNotFoundError:
    from core.config import AppConfig
    from core.models import MemoryTurn
    from parallel_runner import (
        prepare_baseline_index_in_process,
        prepare_pageindex_index_in_process,
        run_baseline_in_process,
        run_pageindex_cached_in_process,
    )

load_dotenv(dotenv_path=ROOT_DIR / ".env")
st.set_page_config(page_title="Baseline vs PageIndex RAG", layout="wide")


# ---------------------------------------------------------------------------
# Asyncio helper
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run an async coroutine safely from Streamlit's synchronous main thread.

    Streamlit ≥ 1.18 runs in a thread that already has a running event loop,
    so a plain asyncio.run() raises 'cannot run nested event loop'.  We
    spin up a fresh loop in a dedicated thread to avoid the conflict.
    """
    result_holder = {}

    def _target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_holder["value"] = loop.run_until_complete(coro)
        except Exception as exc:  # noqa: BLE001
            result_holder["error"] = exc
        finally:
            loop.close()

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join()

    if "error" in result_holder:
        raise result_holder["error"]
    return result_holder["value"]


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def init_state() -> None:
    if "initialized" in st.session_state:
        return
    cfg = AppConfig()
    st.session_state.cfg = cfg
    st.session_state.pdf_indexed = False
    st.session_state.pdf_path = ""
    st.session_state.baseline_memory: List[MemoryTurn] = []
    st.session_state.pageindex_memory: List[MemoryTurn] = []
    st.session_state.baseline_vectorstore_path = ""
    st.session_state.baseline_retrieval_usage = {}
    st.session_state.pageindex_cache_path = ""
    st.session_state.pageindex_doc_id = ""          # display only
    st.session_state.openai_api_key = ""
    st.session_state.pageindex_api_key = ""
    st.session_state.initialized = True


def apply_runtime_api_keys() -> None:
    openai_key = (st.session_state.get("openai_api_key") or "").strip()
    pageindex_key = (st.session_state.get("pageindex_api_key") or "").strip()
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if pageindex_key:
        os.environ["PAGEINDEX_API_KEY"] = pageindex_key


# ---------------------------------------------------------------------------
# Core async runner
# ---------------------------------------------------------------------------

async def run_both(
        question: str,
        cfg,
        pdf_path,
        baseline_mem,
        pageindex_mem,
        baseline_vectorstore_path,
        pageindex_cache_path,
        openai_api_key,
        pageindex_api_key,
    ):
    """Run both RAG pipelines concurrently in separate worker processes."""
    baseline_mem = [
        f"User: {m.question}\nAssistant: {m.answer}"
        for m in baseline_mem
    ]
    pageindex_mem = [
        f"User: {m.question}\nAssistant: {m.answer}"
        for m in pageindex_mem
    ]

    baseline_vectorstore_path = (baseline_vectorstore_path).strip()
    pageindex_cache_path = (pageindex_cache_path).strip()

    openai_api_key = (
        (openai_api_key).strip() or cfg.openai_api_key
    )
    pageindex_api_key = (
        (pageindex_api_key).strip() or cfg.pageindex_api_key
    )

    baseline_timeout = getattr(
        cfg, "baseline_timeout_sec", getattr(cfg, "answer_timeout_sec", 45)
    )
    pageindex_timeout = getattr(
        cfg, "pageindex_timeout_sec", getattr(cfg, "answer_timeout_sec", 45)
    )

    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=2) as process_pool:
        baseline_future = loop.run_in_executor(
            process_pool,
            run_baseline_in_process,
            pdf_path,
            question,
            baseline_mem,
            openai_api_key,
            baseline_vectorstore_path,
        )
        pageindex_future = loop.run_in_executor(
            process_pool,
            run_pageindex_cached_in_process,
            pdf_path,
            question,
            pageindex_mem,
            openai_api_key,
            pageindex_api_key,
            pageindex_cache_path,
        )
        baseline_result, pageindex_result = await asyncio.gather(
            asyncio.wait_for(baseline_future, timeout=baseline_timeout),
            asyncio.wait_for(pageindex_future, timeout=pageindex_timeout),
            return_exceptions=True,
        )
    return baseline_result, pageindex_result


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _render_retrieval_caption(result: dict, pipeline: str) -> None:
    """Render symmetric stats caption for either pipeline result."""
    tok = result.get("token_usage", {})
    ret = result.get("retrieval_usage", {})
    elapsed = result.get("elapsed_ms", 0)

    token_str = (
        f"prompt={tok.get('prompt_tokens', 0)}, "
        f"completion={tok.get('completion_tokens', 0)}, "
        f"total={tok.get('total_tokens', 0)}"
    )

    if pipeline == "baseline":
        retrieval_str = (
            f"chunks={ret.get('retrieved_chunks', 0)}, "
            f"est_ctx_tokens={ret.get('estimated_context_tokens', 0)}"
        )
    else:  # pageindex
        retrieval_str = (
            f"nodes={ret.get('retrieved_nodes', 0)}, "
            f"ctx_chars={ret.get('context_chars', 0)}, "
            f"est_ctx_tokens={ret.get('estimated_context_tokens', 0)}"
        )

    st.caption(
        f"Tokens: {token_str} | time={elapsed}ms | retrieval: {retrieval_str}"
    )


# ---------------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------------

def main() -> None:
    init_state()
    cfg = st.session_state.cfg
    st.title("Baseline RAG vs PageIndex RAG")
    st.caption("Upload one PDF, ask one question, compare both answers side-by-side.")

    # ── Sidebar: API keys ────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("API Keys")
        st.text_input(
            "OpenAI API Key",
            key="openai_api_key",
            type="password",
            help="Used by both Baseline and PageIndex answer generation.",
        )
        st.text_input(
            "PageIndex API Key",
            key="pageindex_api_key",
            type="password",
            help="Used to submit/query document nodes from PageIndex.",
        )
        st.caption("Keys are used for this app session only.")

        # Show indexing info if available
        if st.session_state.get("pdf_indexed"):
            st.divider()
            st.subheader("Index Info")
            baseline_usage = st.session_state.get("baseline_retrieval_usage", {})
            if baseline_usage:
                st.caption(
                    f"Baseline chunks: {baseline_usage.get('chunk_count', '?')} | "
                    f"est. embed tokens: {baseline_usage.get('estimated_embedding_tokens', '?')}"
                )
            doc_id = st.session_state.get("pageindex_doc_id", "")
            if doc_id:
                st.caption(f"PageIndex doc_id: {doc_id}")

    # ── PDF upload + indexing ────────────────────────────────────────────────
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded is not None and st.button("Index PDF", type="primary"):
        pdf_bytes = uploaded.read()
        if not pdf_bytes:
            st.error("Uploaded PDF is empty.")
            st.stop()

        with st.spinner("Saving uploaded PDF..."):
            try:
                fd, tmp_path = tempfile.mkstemp(suffix=".pdf", prefix="rag_cmp_")
                os.close(fd)
                with open(tmp_path, "wb") as out:
                    out.write(pdf_bytes)
                st.session_state.pdf_path = tmp_path
                # Reset stale index state
                st.session_state.pdf_indexed = False
                st.session_state.pageindex_cache_path = ""
                st.session_state.pageindex_doc_id = ""
                st.session_state.baseline_vectorstore_path = ""
            except Exception as exc:
                st.error(f"Failed to save PDF: {exc}")
                st.stop()

        openai_key = (
            (st.session_state.get("openai_api_key") or "").strip() or cfg.openai_api_key
        )
        pageindex_key = (
            (st.session_state.get("pageindex_api_key") or "").strip() or cfg.pageindex_api_key
        )

        if not openai_key:
            st.warning("Provide an OpenAI API key to build the baseline index.")
            st.stop()
        if not pageindex_key:
            st.warning("Provide a PageIndex API key to build the PageIndex cache.")
            st.stop()

        # ── Baseline index (embeddings → Chroma) ────────────────────────────
        with st.spinner("Preparing baseline embeddings/index…"):
            index_dir = os.path.join(
                tempfile.gettempdir(),
                "rag_cmp_baseline_index",
                Path(tmp_path).stem,
            )
            try:
                idx_info = prepare_baseline_index_in_process(tmp_path, openai_key, index_dir)
                st.session_state.baseline_vectorstore_path = idx_info.get(
                    "vectorstore_path", ""
                )
                st.session_state.baseline_retrieval_usage = idx_info.get(
                    "retrieval_usage", {}
                )
            except Exception as exc:
                st.error(f"Baseline indexing failed: {exc}")
                st.stop()

        # ── PageIndex prepare (submit → poll → tree fetch → cache) ──────────
        with st.spinner(
            "Preparing PageIndex cache (submit → process → tree fetch)… "
            "This may take a minute for large PDFs."
        ):
            pageindex_cache_dir = os.path.join(
                tempfile.gettempdir(),
                "rag_cmp_pageindex_cache",
                Path(tmp_path).stem,
            )
            os.makedirs(pageindex_cache_dir, exist_ok=True)
            pageindex_cache_path = os.path.join(pageindex_cache_dir, "tree_cache.json")
            try:
                pi_info = prepare_pageindex_index_in_process(
                    tmp_path,
                    openai_key,
                    pageindex_key,
                    pageindex_cache_path,
                )
                st.session_state.pageindex_cache_path = pi_info.get(
                    "cache_path", pageindex_cache_path
                )
                st.session_state.pageindex_doc_id = pi_info.get("doc_id", "")
            except Exception as exc:
                st.error(f"PageIndex preparation failed: {exc}")
                st.stop()

        st.session_state.pdf_indexed = True
        st.success(
            f"PDF indexed. "
            f"Baseline: {idx_info.get('chunk_count', '?')} chunks | "
            f"PageIndex: {pi_info.get('node_count', '?')} nodes. "
            f"You can now ask questions."
        )

    # ── Question input + Ask button ──────────────────────────────────────────
    question = st.text_input("Ask a question")
    ask_clicked = st.button("Ask", type="secondary")

    if ask_clicked:
        apply_runtime_api_keys()

        # Guard: API keys
        if not (
            (st.session_state.get("openai_api_key") or "").strip() or cfg.openai_api_key
        ):
            st.warning("Please provide your OpenAI API key in the sidebar.")
            st.stop()
        if not (
            (st.session_state.get("pageindex_api_key") or "").strip()
            or cfg.pageindex_api_key
        ):
            st.warning("Please provide your PageIndex API key in the sidebar.")
            st.stop()

        # Guard: index ready
        if not st.session_state.pdf_indexed:
            st.warning("Please upload and index a PDF first.")
            st.stop()
        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()
        if not (st.session_state.get("baseline_vectorstore_path") or "").strip():
            st.warning("Baseline index not ready. Please click 'Index PDF' again.")
            st.stop()
        if not (st.session_state.get("pageindex_cache_path") or "").strip():
            st.warning("PageIndex cache not ready. Please click 'Index PDF' again.")
            st.stop()

        # Run both pipelines concurrently
        with st.spinner("Running both RAG systems in parallel…"):
            # Use _run_async to avoid the 'cannot run nested event loop' error
            # that occurs because Streamlit's thread may already have a loop.
            baseline_result, pageindex_result = _run_async(
                run_both(
                    question.strip(),
                    st.session_state.cfg,
                    st.session_state.pdf_path,
                    st.session_state.baseline_memory,
                    st.session_state.pageindex_memory,
                    st.session_state.get("baseline_vectorstore_path"),
                    st.session_state.get("pageindex_cache_path"),
                    st.session_state.get("openai_api_key"),
                    st.session_state.get("pageindex_api_key"),
                )
            )

        # ── Side-by-side results ─────────────────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Baseline RAG")
            if isinstance(baseline_result, Exception):
                st.error(
                    f"Failed: {type(baseline_result).__name__}: {baseline_result}"
                )
            else:
                if baseline_result.get("ok"):
                    st.write(baseline_result.get("answer", ""))
                else:
                    st.error(baseline_result.get("error") or "Unknown error")
                _render_retrieval_caption(baseline_result, "baseline")

        with col2:
            st.subheader("PageIndex RAG")
            if isinstance(pageindex_result, Exception):
                st.error(
                    f"Failed: {type(pageindex_result).__name__}: {pageindex_result}"
                )
            else:
                if pageindex_result.get("ok"):
                    st.write(pageindex_result.get("answer", ""))
                else:
                    st.error(pageindex_result.get("error") or "Unknown error")
                _render_retrieval_caption(pageindex_result, "pageindex")

        # ── Update per-strategy memory buffers ───────────────────────────────
        if not isinstance(baseline_result, Exception) and baseline_result.get("ok"):
            st.session_state.baseline_memory.append(
                MemoryTurn(
                    question=question.strip(),
                    answer=baseline_result.get("answer", ""),
                )
            )
            st.session_state.baseline_memory = st.session_state.baseline_memory[
                -cfg.memory_turns :
            ]

        if not isinstance(pageindex_result, Exception) and pageindex_result.get("ok"):
            # Persist doc_id for display only; cache_path was already set during prepare
            doc_id = (pageindex_result.get("doc_id") or "").strip()
            if doc_id:
                st.session_state.pageindex_doc_id = doc_id
            st.session_state.pageindex_memory.append(
                MemoryTurn(
                    question=question.strip(),
                    answer=pageindex_result.get("answer", ""),
                )
            )
            st.session_state.pageindex_memory = st.session_state.pageindex_memory[
                -cfg.memory_turns :
            ]


if __name__ == "__main__":
    main()