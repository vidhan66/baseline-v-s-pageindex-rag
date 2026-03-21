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
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    DEFAULT_PDF_PATH = r"C:\Users\vidha\Downloads\The Art of Persuasion by Bob Burg.pdf"

load_dotenv()

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
        "token_usage": token_usage,
        "elapsed_ms": int((time.perf_counter() - start) * 1000),
    }

async def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set in environment")
    if not os.getenv("PAGEINDEX_API_KEY"):
        raise ValueError("PAGEINDEX_API_KEY not set in environment")

    existing_doc_id = (os.getenv("PAGEINDEX_DOC_ID") or "").strip()
    if existing_doc_id:
        doc_id = existing_doc_id
        logger.info("Using existing PAGEINDEX_DOC_ID=%s (skipping submission)", doc_id)
    else:
        pdf_path = os.getenv("PDF_PATH", Config.DEFAULT_PDF_PATH)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at: {pdf_path}")
        if not pdf_path.lower().endswith(".pdf"):
            raise ValueError(f"PDF_PATH must point to a .pdf file, got: {pdf_path}")
        logger.info("Using PDF: %s", pdf_path)

        logger.info("Submitting document to PageIndex")
        try:
            submit_resp = pi_client.submit_document(pdf_path)
        except Exception as e:
            err_text = str(e)
            if "LimitReached" in err_text:
                fallback_enabled = os.getenv("ENABLE_BASELINE_FALLBACK", "true").lower() in {"1", "true", "yes", "on"}
                logger.error(
                    "PageIndex limit reached for document submission. "
                    "This is a provider-side quota/workspace limit, not an OpenAI credit issue. "
                    "Set PAGEINDEX_DOC_ID to an existing processed document to continue without re-submitting."
                )
                if fallback_enabled:
                    baseline_script = os.path.abspath(
                        os.path.join(os.path.dirname(__file__), "..", "baseline", "baseline_rag.py")
                    )
                    if not os.path.exists(baseline_script):
                        raise FileNotFoundError(
                            f"Fallback baseline script not found at: {baseline_script}"
                        ) from e
                    logger.info("Falling back to baseline pipeline: %s", baseline_script)
                    subprocess.run([sys.executable, baseline_script], check=True)
                    return
                logger.info(
                    "Baseline fallback is disabled. Set ENABLE_BASELINE_FALLBACK=true to auto-fallback."
                )
            logger.exception("PageIndex submit_document failed: %s", e)
            raise

        logger.info("PageIndex submit response type=%s", type(submit_resp).__name__)
        logger.info("PageIndex submit response: %s", submit_resp)

        # Some SDK/API versions may return doc id under different keys or wrappers.
        doc_id = None
        if isinstance(submit_resp, dict):
            doc_id = (
                submit_resp.get("doc_id")
                or submit_resp.get("document_id")
                or (submit_resp.get("result", {}) or {}).get("doc_id")
                or (submit_resp.get("data", {}) or {}).get("doc_id")
            )
        elif isinstance(submit_resp, str):
            # Fallback for SDKs that return the raw id string.
            doc_id = submit_resp.strip()

        if not doc_id or not isinstance(doc_id, str):
            raise ValueError(
                f"Could not extract doc_id from submit response: {submit_resp}"
            )

        logger.info("Document submitted with doc_id=%s", doc_id)

    max_wait_seconds = int(os.getenv("PAGEINDEX_READY_TIMEOUT_SEC", "300"))
    poll_interval_seconds = int(os.getenv("PAGEINDEX_READY_POLL_SEC", "5"))
    elapsed = 0
    while elapsed < max_wait_seconds:
        if pi_client.is_retrieval_ready(doc_id):
            logger.info("Retrieval ready after %d seconds", elapsed)
            break
        logger.info("Waiting for processing... elapsed=%ds", elapsed)
        await asyncio.sleep(poll_interval_seconds)
        elapsed += poll_interval_seconds
    else:
        raise TimeoutError(f"PageIndex not ready within {max_wait_seconds}s for doc_id={doc_id}")

    tree = pi_client.get_tree(doc_id, node_summary=True)['result']
    logger.info("Loaded tree structure")

    query = os.getenv("RAG_QUERY", "Summarize the section montivating the unmotivated.")
    logger.info("Processing query: %s", query)

    tree_without_text = utils.remove_fields(tree.copy(), fields=['text'])

    search_prompt = f"""
    You are given a question and a tree structure of a document.
    Each node contains a node id, node title, and a corresponding summary.
    Your task is to find all nodes that are likely to contain the answer to the question.

    Question: {query}

    Document tree structure:
    {json.dumps(tree_without_text, indent=2)}

    Please reply in the following JSON format:
    {{
        "thinking": "<Your thinking process on which nodes are relevant to the question>",
        "node_list": ["node_id_1", "node_id_2", ..., "node_id_n"]
    }}
    Directly return the final JSON structure. Do not output anything else.
    """

    tree_search_result = await call_llm(search_prompt)

    node_map = utils.create_node_mapping(tree)
    try:
        tree_search_result_json = json.loads(tree_search_result)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON from tree search: %s", e)
        logger.error("Raw response: %s", tree_search_result)
        raise

    logger.info("Reasoning process:")
    utils.print_wrapped(tree_search_result_json['thinking'])

    logger.info("Retrieved nodes:")
    node_list = tree_search_result_json.get("node_list", [])
    if not isinstance(node_list, list) or not node_list:
        raise ValueError("node_list missing or empty in tree_search_result_json")
    for node_id in node_list:
        if node_id not in node_map:
            logger.warning("Skipping unknown node_id: %s", node_id)
            continue
        node = node_map[node_id]
        print(f"Node ID: {node['node_id']}\t Page: {node['page_index']}\t Title: {node['title']}")

    relevant_content = "\n\n".join(
        node_map[node_id]["text"] for node_id in node_list if node_id in node_map
    )
    if not relevant_content.strip():
        raise ValueError("Retrieved content is empty after node selection.")

    logger.info("Retrieved context preview:")
    utils.print_wrapped(relevant_content[:1000] + '...')

    answer_prompt = f"""
Answer the question based on the context:

Question: {query}
Context: {relevant_content}

Provide a clear, concise answer based only on the context provided.
"""

    logger.info("Generating final answer")
    answer = await call_llm(answer_prompt)
    logger.info("Answer generated")
    utils.print_wrapped(answer)

if __name__ == "__main__":
    asyncio.run(main())