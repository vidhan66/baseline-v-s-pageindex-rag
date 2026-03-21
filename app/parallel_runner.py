import asyncio
import importlib.util
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_function_from_file(file_path: Path, function_name: str):
    spec = importlib.util.spec_from_file_location(f"dyn_{file_path.stem}", str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, function_name, None)
    if fn is None:
        raise AttributeError(f"{function_name} not found in {file_path}")
    return fn


def run_baseline_in_process(
    pdf_path: str,
    question: str,
    memory: list[str],
    openai_api_key: str,
    baseline_vectorstore_path: str,
) -> dict:
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


def prepare_baseline_index_in_process(
    pdf_path: str,
    openai_api_key: str,
    baseline_vectorstore_path: str,
) -> dict:
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    prepare_baseline_index = _load_function_from_file(
        ROOT_DIR / "baseline" / "baseline_rag.py",
        "prepare_baseline_index",
    )
    return prepare_baseline_index(pdf_path, baseline_vectorstore_path)


def run_pageindex_in_process(
    pdf_path: str,
    question: str,
    memory: list[str],
    openai_api_key: str,
    pageindex_api_key: str,
    pageindex_doc_id: str,
) -> dict:
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
