import os
import logging
import time
import hashlib
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from openai import AsyncOpenAI

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
class Config:
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    RETRIEVAL_K = 8
    RETRIEVAL_FETCH_K = 30
    TEMPERATURE = 0
    VECTORSTORE_PATH = "../data/chroma_db"
    DEFAULT_PDF_PATH = r"C:\Users\vidha\Downloads\The Art of Persuasion by Bob Burg.pdf"


def _sanitize_text(value: str) -> str:
    # Strip invalid surrogate characters that break UTF-8 encoding.
    return (value or "").encode("utf-8", errors="ignore").decode("utf-8")


def _default_vectorstore_path_for_pdf(pdf_path: str) -> str:
    key = hashlib.sha256(os.path.abspath(pdf_path).encode("utf-8")).hexdigest()[:16]
    base_dir = Path(__file__).resolve().parent.parent / "data" / "chroma_db_cache"
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir / key)

def load_pdf_pages(pdf_path: str):
    """Load PDF pages as documents with page metadata."""
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        if not pages:
            raise ValueError(f"No pages extracted from {pdf_path}")
        total_chars = sum(len(page.page_content or "") for page in pages)
        logger.info(f"Extracted {len(pages)} pages, {total_chars} characters")
        return pages
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise

async def run_baseline_query(pdf_path: str, query: str, memory: List[str] | None = None) -> dict:
    start = time.perf_counter()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set in environment")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    pages = load_pdf_pages(pdf_path)
    for page in pages:
        page.page_content = _sanitize_text(page.page_content or "")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=len
    )
    chunked_docs = text_splitter.split_documents(pages)
    logger.info("Split text into %d chunks", len(chunked_docs))

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = Chroma.from_documents(
        chunked_docs,
        embeddings,
        persist_directory=Config.VECTORSTORE_PATH
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": Config.RETRIEVAL_K, "fetch_k": Config.RETRIEVAL_FETCH_K}
    )
    clean_query = _sanitize_text(query)
    docs = await retriever.ainvoke(clean_query)
    context = _sanitize_text("\n\n".join((d.page_content or "") for d in docs))

    memory_text = _sanitize_text("\n".join(memory or [])) if memory else "No previous conversation."
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant for question answering over a PDF.\n"
        "Use only the provided context and memory. If answer is missing, say you do not know.\n\n"
        "Conversation memory:\n{memory}\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{input}"
    ).format(memory=memory_text, context=context, input=clean_query)
    prompt = _sanitize_text(prompt)

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=Config.TEMPERATURE,
    )
    answer = _sanitize_text((resp.choices[0].message.content or "").strip())
    usage = resp.usage
    return {
        "ok": True,
        "answer": answer,
        "token_usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
        },
        "elapsed_ms": int((time.perf_counter() - start) * 1000),
    }


def prepare_baseline_index(pdf_path: str, vectorstore_path: str | None = None) -> dict:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set in environment")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    resolved_path = vectorstore_path or _default_vectorstore_path_for_pdf(pdf_path)
    pages = load_pdf_pages(pdf_path)
    for page in pages:
        page.page_content = _sanitize_text(page.page_content or "")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=len,
    )
    chunked_docs = text_splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    Chroma.from_documents(
        chunked_docs,
        embeddings,
        persist_directory=resolved_path,
    )
    return {
        "ok": True,
        "vectorstore_path": resolved_path,
        "chunk_count": len(chunked_docs),
        "retrieval_usage": {
            "estimated_embedding_tokens": int(
                sum(len(d.page_content or "") for d in chunked_docs) / 4
            ),
            "chunk_count": len(chunked_docs),
        },
    }


async def run_baseline_query_cached(
    pdf_path: str,
    query: str,
    memory: List[str] | None = None,
    vectorstore_path: str | None = None,
) -> dict:
    start = time.perf_counter()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set in environment")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    resolved_path = vectorstore_path or _default_vectorstore_path_for_pdf(pdf_path)
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = Chroma(
        persist_directory=resolved_path,
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": Config.RETRIEVAL_K, "fetch_k": Config.RETRIEVAL_FETCH_K},
    )
    clean_query = _sanitize_text(query)
    docs = await retriever.ainvoke(clean_query)
    context = _sanitize_text("\n\n".join((d.page_content or "") for d in docs))
    memory_text = _sanitize_text("\n".join(memory or [])) if memory else "No previous conversation."
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant for question answering over a PDF.\n"
        "Use only the provided context and memory. If answer is missing, say you do not know.\n\n"
        "Conversation memory:\n{memory}\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{input}"
    ).format(memory=memory_text, context=context, input=clean_query)
    prompt = _sanitize_text(prompt)

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=Config.TEMPERATURE,
    )
    answer = _sanitize_text((resp.choices[0].message.content or "").strip())
    usage = resp.usage
    return {
        "ok": True,
        "answer": answer,
        "token_usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
        },
        "retrieval_usage": {
            "retrieved_chunks": len(docs),
            "context_chars": len(context),
            "estimated_context_tokens": int(len(context) / 4),
        },
        "elapsed_ms": int((time.perf_counter() - start) * 1000),
    }

def main():
    """Main RAG pipeline with error handling."""
    try:
        # Validate API keys
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not set in environment")
        
        # Load local PDF (override with PDF_PATH env var if needed)
        pdf_path = os.getenv("PDF_PATH", Config.DEFAULT_PDF_PATH)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at: {pdf_path}")
        logger.info(f"Using PDF: {pdf_path}")
        
        # Query
        query = os.getenv(
            "RAG_QUERY",
            "Summarize the section montivating the unmotivated."
        )
        logger.info(f"Processing query: {query}")
        import asyncio
        response = asyncio.run(run_baseline_query(pdf_path, query))
        result = response.get("answer", "")
        
        if not result or result.strip() == "":
            logger.warning("Empty response from QA chain")
        else:
            logger.info("Query successful")
        
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"Answer: {result}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise

if __name__ == "__main__":
    main()