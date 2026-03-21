import asyncio
from concurrent.futures import ProcessPoolExecutor
import os
import sys
import tempfile
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
        run_baseline_in_process,
        run_pageindex_in_process,
    )
except ModuleNotFoundError:
    # Fallback when streamlit executes this file with app/ as the root.
    from core.config import AppConfig
    from core.models import MemoryTurn
    from parallel_runner import (
        prepare_baseline_index_in_process,
        run_baseline_in_process,
        run_pageindex_in_process,
    )

load_dotenv(dotenv_path=ROOT_DIR / ".env")
st.set_page_config(page_title="Baseline vs PageIndex RAG", layout="wide")


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
    st.session_state.pageindex_doc_id = ""
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


async def run_both(question: str):
    cfg = st.session_state.cfg
    pdf_path = st.session_state.pdf_path
    baseline_mem = [f"User: {m.question}\nAssistant: {m.answer}" for m in st.session_state.baseline_memory]
    pageindex_mem = [f"User: {m.question}\nAssistant: {m.answer}" for m in st.session_state.pageindex_memory]
    baseline_vectorstore_path = (st.session_state.get("baseline_vectorstore_path") or "").strip()
    openai_api_key = (st.session_state.get("openai_api_key") or "").strip() or cfg.openai_api_key
    pageindex_api_key = (st.session_state.get("pageindex_api_key") or "").strip() or cfg.pageindex_api_key
    pageindex_doc_id = (st.session_state.get("pageindex_doc_id") or "").strip()
    baseline_timeout = getattr(cfg, "baseline_timeout_sec", getattr(cfg, "answer_timeout_sec", 45))
    pageindex_timeout = getattr(cfg, "pageindex_timeout_sec", getattr(cfg, "answer_timeout_sec", 45))

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
            run_pageindex_in_process,
            pdf_path,
            question,
            pageindex_mem,
            openai_api_key,
            pageindex_api_key,
            pageindex_doc_id,
        )
        baseline_result, pageindex_result = await asyncio.gather(
            asyncio.wait_for(baseline_future, timeout=baseline_timeout),
            asyncio.wait_for(pageindex_future, timeout=pageindex_timeout),
            return_exceptions=True,
        )
    return baseline_result, pageindex_result


def main() -> None:
    init_state()
    cfg = st.session_state.cfg
    st.title("Baseline RAG vs PageIndex RAG")
    st.caption("Upload one PDF, ask one question, compare both answers side-by-side.")

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
                st.session_state.pdf_indexed = True
                st.session_state.pageindex_doc_id = ""
                with st.spinner("Preparing baseline embeddings/index..."):
                    openai_key = (st.session_state.get("openai_api_key") or "").strip() or cfg.openai_api_key
                    if not openai_key:
                        st.warning("Provide OpenAI API key to build baseline index.")
                        st.stop()
                    index_dir = os.path.join(
                        tempfile.gettempdir(),
                        "rag_cmp_baseline_index",
                        Path(tmp_path).stem,
                    )
                    idx_info = prepare_baseline_index_in_process(tmp_path, openai_key, index_dir)
                    st.session_state.baseline_vectorstore_path = idx_info.get("vectorstore_path", "")
                    st.session_state.baseline_retrieval_usage = idx_info.get("retrieval_usage", {})
                st.success("PDF saved and baseline index prepared. You can now ask questions.")
            except Exception as exc:
                st.error(f"Failed to save PDF: {exc}")

    question = st.text_input("Ask a question")
    ask_clicked = st.button("Ask", type="secondary")

    if ask_clicked:
        apply_runtime_api_keys()
        if not ((st.session_state.get("openai_api_key") or "").strip() or cfg.openai_api_key):
            st.warning("Please provide your OpenAI API key in the sidebar.")
            st.stop()
        if not ((st.session_state.get("pageindex_api_key") or "").strip() or cfg.pageindex_api_key):
            st.warning("Please provide your PageIndex API key in the sidebar.")
            st.stop()
        if not st.session_state.pdf_indexed:
            st.warning("Please upload and index a PDF first.")
            st.stop()
        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()
        if not (st.session_state.get("baseline_vectorstore_path") or "").strip():
            st.warning("Baseline index not ready. Please click 'Index PDF' again.")
            st.stop()

        with st.spinner("Running both RAG systems..."):
            baseline_result, pageindex_result = asyncio.run(run_both(question.strip()))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Baseline RAG")
            if isinstance(baseline_result, Exception):
                st.error(f"Failed: {type(baseline_result).__name__}: {baseline_result}")
            else:
                if baseline_result.get("ok"):
                    st.write(baseline_result.get("answer", ""))
                else:
                    st.error(baseline_result.get("error") or "Unknown error")
                b_tok = baseline_result.get("token_usage", {})
                b_ret = baseline_result.get("retrieval_usage", {})
                st.caption(
                    f"Tokens: prompt={b_tok.get('prompt_tokens', 0)}, "
                    f"completion={b_tok.get('completion_tokens', 0)}, "
                    f"total={b_tok.get('total_tokens', 0)} | "
                    f"time={baseline_result.get('elapsed_ms', 0)}ms | "
                    f"retrieval: chunks={b_ret.get('retrieved_chunks', 0)}, "
                    f"est_ctx_tokens={b_ret.get('estimated_context_tokens', 0)}"
                )

        with col2:
            st.subheader("PageIndex RAG")
            if isinstance(pageindex_result, Exception):
                st.error(f"Failed: {type(pageindex_result).__name__}: {pageindex_result}")
            else:
                if pageindex_result.get("ok"):
                    st.write(pageindex_result.get("answer", ""))
                else:
                    st.error(pageindex_result.get("error") or "Unknown error")
                p_tok = pageindex_result.get("token_usage", {})
                st.caption(
                    f"Tokens: prompt={p_tok.get('prompt_tokens', 0)}, "
                    f"completion={p_tok.get('completion_tokens', 0)}, "
                    f"total={p_tok.get('total_tokens', 0)} | "
                    f"time={pageindex_result.get('elapsed_ms', 0)}ms"
                )

        # simple strategy-specific memory buffers
        if not isinstance(baseline_result, Exception) and baseline_result.get("ok"):
            st.session_state.baseline_memory.append(
                MemoryTurn(question=question.strip(), answer=baseline_result.get("answer", ""))
            )
            st.session_state.baseline_memory = st.session_state.baseline_memory[-cfg.memory_turns :]
        if not isinstance(pageindex_result, Exception) and pageindex_result.get("ok"):
            doc_id = (pageindex_result.get("doc_id") or "").strip()
            if doc_id:
                st.session_state.pageindex_doc_id = doc_id
            st.session_state.pageindex_memory.append(
                MemoryTurn(question=question.strip(), answer=pageindex_result.get("answer", ""))
            )
            st.session_state.pageindex_memory = st.session_state.pageindex_memory[-cfg.memory_turns :]


if __name__ == "__main__":
    main()

