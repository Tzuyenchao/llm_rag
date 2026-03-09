from __future__ import annotations

import logging
import os
import re
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.converters import PyPDFToDocument
from haystack.components.embedders import (
    HuggingFaceAPIDocumentEmbedder,
    HuggingFaceAPITextEmbedder,
)
from haystack.components.generators import OpenAIGenerator
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore


# =========================
# Config
# =========================

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", BASE_DIR / "documents"))

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_INDEX = os.getenv("QDRANT_INDEX", "pdf_chunks")
QDRANT_RECREATE_INDEX = os.getenv("QDRANT_RECREATE_INDEX", "false").lower() == "true"
QDRANT_SIMILARITY = os.getenv("QDRANT_SIMILARITY", "cosine")
QDRANT_EMBEDDING_DIM = int(os.getenv("QDRANT_EMBEDDING_DIM", "1024"))

TEI_URL = os.getenv("TEI_URL", "http://127.0.0.1:8080")
TEI_TIMEOUT = float(os.getenv("TEI_TIMEOUT", "120"))

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "dummy")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-8B")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))

RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))
GENERATION_MAX_TOKENS = int(os.getenv("GENERATION_MAX_TOKENS", "1024"))
GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "0.2"))

# chunk strategy
CHUNK_SPLIT_BY = os.getenv("CHUNK_SPLIT_BY", "word")
CHUNK_LENGTH = int(os.getenv("CHUNK_LENGTH", "300"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# retrieval gating
MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.35"))
MIN_CONTENT_LENGTH = int(os.getenv("MIN_CONTENT_LENGTH", "20"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("api_server")


# =========================
# Models
# =========================

class ChatRequest(BaseModel):
    question: str = Field(..., description="使用者輸入的提問內容")
    system_prompt: str = Field(
        default=(
            "你是 RAG 文件助理。請只根據提供的參考內容回答。"
            "若參考內容不足或無法支持答案，請直接回覆：無法回覆該問題。"
            "回答請使用繁體中文。"
        ),
        description="系統提示詞",
    )
    top_k: int = Field(default=RETRIEVER_TOP_K, ge=1, le=20)
    max_tokens: int = Field(default=GENERATION_MAX_TOKENS, ge=64, le=4096)
    temperature: float = Field(default=GENERATION_TEMPERATURE, ge=0.0, le=2.0)


class ReindexRequest(BaseModel):
    recreate_index: bool = Field(default=False)


@dataclass
class SourceItem:
    document_id: str
    file_name: str
    page_number: int
    chunk_index: int
    chunk_word_count: int
    score: Optional[float]
    chunk_content: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    indexing: bool
    indexed_documents: int
    documents_dir: str
    qdrant_index: str
    chunk_split_by: str
    chunk_length: int
    chunk_overlap: int
    min_retrieval_score: float


class ReindexResponse(BaseModel):
    status: str
    message: str


# =========================
# App State
# =========================

class AppState:
    def __init__(self) -> None:
        self.indexing_lock = threading.Lock()
        self.indexing_in_progress = False
        self.indexing_error: Optional[str] = None
        self.document_store: Optional[QdrantDocumentStore] = None
        self.indexing_pipeline: Optional[Pipeline] = None
        self.rag_pipeline: Optional[Pipeline] = None


state = AppState()


# =========================
# Helpers
# =========================

def _secret_from_optional_token(token: str) -> Optional[Secret]:
    if token.strip():
        return Secret.from_token(token)
    return None


def _safe_int(value: Any, default: int = 1) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def _has_enough_information(documents: list[Document]) -> bool:
    if not documents:
        return False

    valid_docs = []
    for doc in documents:
        content = _clean_text(doc.content or "")
        score = _safe_float(getattr(doc, "score", None), default=None)

        if len(content) < MIN_CONTENT_LENGTH:
            continue
        if score is not None and score < MIN_RETRIEVAL_SCORE:
            continue

        valid_docs.append(doc)

    return len(valid_docs) > 0


def build_document_store(recreate_index: bool = False) -> QdrantDocumentStore:
    qdrant_api_key = _secret_from_optional_token(QDRANT_API_KEY)

    kwargs: dict[str, Any] = {
        "url": QDRANT_URL,
        "index": QDRANT_INDEX,
        "embedding_dim": QDRANT_EMBEDDING_DIM,
        "similarity": QDRANT_SIMILARITY,
        "recreate_index": recreate_index,
        "wait_result_from_api": True,
        "return_embedding": False,
    }
    if qdrant_api_key is not None:
        kwargs["api_key"] = qdrant_api_key

    return QdrantDocumentStore(**kwargs)


def build_indexing_pipeline(document_store: QdrantDocumentStore) -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component("converter", PyPDFToDocument())
    pipeline.add_component("cleaner", DocumentCleaner())

    # 先按頁切，保留頁碼
    pipeline.add_component(
        "page_splitter",
        DocumentSplitter(split_by="page", split_length=1, split_overlap=0),
    )

    # 再做真正的 chunk
    pipeline.add_component(
        "chunk_splitter",
        DocumentSplitter(
            split_by=CHUNK_SPLIT_BY,
            split_length=CHUNK_LENGTH,
            split_overlap=CHUNK_OVERLAP,
        ),
    )

    pipeline.add_component(
        "document_embedder",
        HuggingFaceAPIDocumentEmbedder(
            api_type="text_embeddings_inference",
            api_params={"url": TEI_URL, "timeout": TEI_TIMEOUT},
        ),
    )
    pipeline.add_component(
        "writer",
        DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE),
    )

    pipeline.connect("converter.documents", "cleaner.documents")
    pipeline.connect("cleaner.documents", "page_splitter.documents")
    pipeline.connect("page_splitter.documents", "chunk_splitter.documents")
    pipeline.connect("chunk_splitter.documents", "document_embedder.documents")
    pipeline.connect("document_embedder.documents", "writer.documents")
    return pipeline


def build_rag_pipeline(document_store: QdrantDocumentStore) -> Pipeline:
    prompt_template = """
{{ system_prompt }}

以下是檢索到的參考 Chunk：
{% for doc in documents %}
[Chunk {{ loop.index }}]
檔名: {{ doc.meta.get('file_name', '') }}
頁數: {{ doc.meta.get('page_number', '') }}
Chunk編號: {{ doc.meta.get('chunk_index', '') }}
內容:
{{ doc.content }}

{% endfor %}

使用者問題：{{ question }}

請嚴格遵守以下規則：
1. 只能依據上方檢索到的內容回答。
2. 不可使用外部知識補充。
3. 若檢索內容不足以支持答案，請直接輸出：無法回覆該問題
4. 若可以回答，答案請精簡、明確、使用繁體中文。
5. 不要額外捏造來源、頁碼、文件名稱或未出現的資訊。
""".strip()

    pipeline = Pipeline()
    pipeline.add_component(
        "text_embedder",
        HuggingFaceAPITextEmbedder(
            api_type="text_embeddings_inference",
            api_params={"url": TEI_URL, "timeout": TEI_TIMEOUT},
        ),
    )
    pipeline.add_component(
        "retriever",
        QdrantEmbeddingRetriever(document_store=document_store),
    )
    pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
    pipeline.add_component(
        "llm",
        OpenAIGenerator(
            api_base_url=LLM_BASE_URL,
            api_key=Secret.from_token(LLM_API_KEY),
            model=LLM_MODEL,
            generation_kwargs={
                "temperature": GENERATION_TEMPERATURE,
                "max_tokens": GENERATION_MAX_TOKENS,
            },
            timeout=LLM_TIMEOUT,
        ),
    )

    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")
    return pipeline


def normalize_page_documents(documents: list[Document]) -> list[Document]:
    normalized: list[Document] = []

    for doc in documents:
        meta = dict(doc.meta or {})

        file_path = meta.get("file_path") or meta.get("source") or ""
        if file_path:
            meta["file_name"] = Path(str(file_path)).name
        else:
            meta.setdefault("file_name", "unknown")

        meta["page_number"] = _safe_int(meta.get("page_number", 1), default=1)

        cleaned_content = _clean_text(doc.content or "")
        if not cleaned_content:
            continue

        normalized.append(
            Document(
                id=str(uuid.uuid4()),
                content=cleaned_content,
                meta=meta,
                embedding=doc.embedding,
            )
        )

    return normalized


def normalize_chunk_documents(documents: list[Document]) -> list[Document]:
    normalized: list[Document] = []

    for idx, doc in enumerate(documents, start=1):
        meta = dict(doc.meta or {})
        content = _clean_text(doc.content or "")
        if not content:
            continue

        meta["file_name"] = str(meta.get("file_name", "unknown"))
        meta["page_number"] = _safe_int(meta.get("page_number", 1), default=1)
        meta["chunk_index"] = idx
        meta["chunk_word_count"] = _word_count(content)

        normalized.append(
            Document(
                id=str(uuid.uuid4()),
                content=content,
                meta=meta,
                embedding=doc.embedding,
            )
        )

    return normalized


def get_pdf_paths() -> list[Path]:
    if not DOCUMENTS_DIR.exists():
        logger.warning("documents dir does not exist: %s", DOCUMENTS_DIR)
        return []
    return sorted(DOCUMENTS_DIR.glob("*.pdf"))


def index_documents(recreate_index: bool = False) -> None:
    if not state.indexing_lock.acquire(blocking=False):
        logger.info("indexing already in progress, skip")
        return

    state.indexing_in_progress = True
    state.indexing_error = None

    try:
        logger.info("start indexing documents")
        pdf_paths = get_pdf_paths()
        if not pdf_paths:
            logger.warning("no pdf files found under %s", DOCUMENTS_DIR)
            return

        document_store = build_document_store(recreate_index=recreate_index)
        indexing_pipeline = build_indexing_pipeline(document_store)

        converted = indexing_pipeline.get_component("converter").run(sources=pdf_paths)["documents"]
        cleaned = indexing_pipeline.get_component("cleaner").run(documents=converted)["documents"]

        page_docs = indexing_pipeline.get_component("page_splitter").run(documents=cleaned)["documents"]
        page_docs = normalize_page_documents(page_docs)

        chunk_docs = indexing_pipeline.get_component("chunk_splitter").run(documents=page_docs)["documents"]
        chunk_docs = normalize_chunk_documents(chunk_docs)

        embedded_docs = indexing_pipeline.get_component("document_embedder").run(documents=chunk_docs)["documents"]
        indexing_pipeline.get_component("writer").run(documents=embedded_docs)

        state.document_store = document_store
        state.indexing_pipeline = indexing_pipeline
        state.rag_pipeline = build_rag_pipeline(document_store)

        logger.info("indexing completed, %s chunk documents indexed", len(embedded_docs))
    except Exception as exc:
        state.indexing_error = str(exc)
        logger.exception("indexing failed: %s", exc)
    finally:
        state.indexing_in_progress = False
        state.indexing_lock.release()


# =========================
# FastAPI lifespan
# =========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state.document_store = build_document_store(recreate_index=QDRANT_RECREATE_INDEX)
        state.indexing_pipeline = build_indexing_pipeline(state.document_store)
        state.rag_pipeline = build_rag_pipeline(state.document_store)
    except Exception as exc:
        logger.exception("failed to initialize pipelines: %s", exc)
        state.indexing_error = str(exc)

    indexing_thread = threading.Thread(
        target=index_documents,
        kwargs={"recreate_index": QDRANT_RECREATE_INDEX},
        daemon=True,
    )
    indexing_thread.start()

    yield


app = FastAPI(title="LLM RAG API Server", version="0.2.0", lifespan=lifespan)


# =========================
# API Endpoints
# =========================

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    indexed_documents = 0
    if state.document_store is not None:
        try:
            indexed_documents = state.document_store.count_documents()
        except Exception:
            indexed_documents = 0

    status = "ok"
    if state.indexing_error:
        status = "degraded"

    return HealthResponse(
        status=status,
        indexing=state.indexing_in_progress,
        indexed_documents=indexed_documents,
        documents_dir=str(DOCUMENTS_DIR),
        qdrant_index=QDRANT_INDEX,
        chunk_split_by=CHUNK_SPLIT_BY,
        chunk_length=CHUNK_LENGTH,
        chunk_overlap=CHUNK_OVERLAP,
        min_retrieval_score=MIN_RETRIEVAL_SCORE,
    )


@app.post("/reindex", response_model=ReindexResponse)
def reindex(request: ReindexRequest) -> ReindexResponse:
    if state.indexing_in_progress:
        return ReindexResponse(status="running", message="indexing already in progress")

    thread = threading.Thread(
        target=index_documents,
        kwargs={"recreate_index": request.recreate_index},
        daemon=True,
    )
    thread.start()
    return ReindexResponse(status="accepted", message="reindex started")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if state.indexing_in_progress:
        raise HTTPException(status_code=503, detail="文件索引中，請稍後再試")

    if state.rag_pipeline is None:
        raise HTTPException(status_code=500, detail="RAG pipeline 尚未初始化")

    if state.indexing_error:
        raise HTTPException(status_code=500, detail=f"索引失敗: {state.indexing_error}")

    try:
        # 先檢索
        result = state.rag_pipeline.run(
            {
                "text_embedder": {"text": request.question},
                "retriever": {"top_k": request.top_k},
                "prompt_builder": {
                    "question": request.question,
                    "system_prompt": request.system_prompt,
                },
                "llm": {
                    "generation_kwargs": {
                        "temperature": request.temperature,
                        "max_tokens": request.max_tokens,
                    }
                },
            }
        )

        retrieved_docs: list[Document] = result["retriever"]["documents"]

        sources: list[dict[str, Any]] = []
        for doc in retrieved_docs:
            score = _safe_float(getattr(doc, "score", None), default=None)
            content = _clean_text(doc.content or "")

            source = SourceItem(
                document_id=str(doc.id or ""),
                file_name=str((doc.meta or {}).get("file_name", "unknown")),
                page_number=_safe_int((doc.meta or {}).get("page_number", 1), default=1),
                chunk_index=_safe_int((doc.meta or {}).get("chunk_index", 0), default=0),
                chunk_word_count=_safe_int((doc.meta or {}).get("chunk_word_count", 0), default=0),
                score=score,
                chunk_content=content,
            )
            sources.append(asdict(source))

        # 明確的檢索不足判定
        if not _has_enough_information(retrieved_docs):
            return ChatResponse(
                answer="無法回覆該問題",
                sources=sources,
            )

        replies = result["llm"]["replies"]
        answer = (replies[0] if replies else "").strip()

        if not answer:
            answer = "無法回覆該問題"

        # 若模型回答不穩，也統一收斂
        weak_patterns = [
            "資訊不足",
            "資料不足",
            "無法確定",
            "無法判斷",
            "不知道",
            "未提供",
        ]
        if any(p in answer for p in weak_patterns):
            answer = "無法回覆該問題"

        return ChatResponse(
            answer=answer,
            sources=sources,
        )
    except Exception as exc:
        logger.exception("chat failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
    )