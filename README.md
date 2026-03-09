# LLM RAG API Server

> **Note**  
> This project is created for a **technical interview assessment**.  
> The implementation focuses on demonstrating a local **RAG (Retrieval-Augmented Generation)** workflow.  
> Actual functionality, dependency versions, and deployment configurations may vary depending on real production environments and requirements.

This project implements a **local knowledge retrieval system** that accepts natural language questions and generates answers based on retrieved document content.

The system is built using:

- FastAPI
- Haystack
- Qdrant (local vector database)
- Text Embeddings Inference (local embedding server)
- vLLM (local LLM inference)

The entire system runs **fully offline** after deployment and does not rely on any cloud APIs.

---

- [LLM RAG API Server](#llm-rag-api-server)
  - [System Architecture](#system-architecture)
    - [Offline Architecture](#offline-architecture)
  - [Project Structure](#project-structure)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Start Vector Database](#start-vector-database)
  - [Start Embedding Server](#start-embedding-server)
  - [Start LLM Server](#start-llm-server)
  - [Start API Server](#start-api-server)
  - [Data Processing](#data-processing)
    - [Document Cleaning](#document-cleaning)
    - [Chunk Strategy](#chunk-strategy)
    - [Embedding Model](#embedding-model)
  - [Vector Database Configuration](#vector-database-configuration)
  - [Question Answering Pipeline](#question-answering-pipeline)
    - [Token Control and Prompt Size](#token-control-and-prompt-size)
  - [API](#api)
    - [Health Check](#health-check)
    - [Chat (RAG)](#chat-rag)
  - [Documents](#documents)
  - [Notes](#notes)
  - [Future Improvements](#future-improvements)
  - [License](#license)

---

## System Architecture

```
User Question
      │
      ▼
FastAPI /chat API
      │
      ▼
Embedding (TEI)
      │
      ▼
Vector Retrieval (Qdrant)
      │
      ▼
Prompt Construction
      │
      ▼
Local LLM (vLLM)
      │
      ▼
Answer + Source Document
```

The system follows a standard **RAG pipeline**:

1. Document preprocessing
2. Chunk splitting
3. Embedding generation
4. Vector database indexing
5. Query embedding
6. Vector retrieval
7. Prompt construction
8. Local LLM answer generation

### Offline Architecture

All components in this system run **entirely on local infrastructure**.

The system does **not call any external cloud APIs**.

Local services used:

| Component       | Implementation            |
| --------------- | ------------------------- |
| LLM inference   | vLLM                      |
| Embedding model | Text Embeddings Inference |
| Vector database | Qdrant                    |

Although OpenAI-compatible and HuggingFace-compatible APIs are used,
all requests are sent only to **locally hosted services**.

---

## Project Structure

```
llm-rag/
│
├─ api_server.py
├─ pyproject.toml
├─ README.md
│
├─ documents/
│   └─ *.pdf
│
├─ llm_server啟動指令.txt
├─ embedding_server啟動指令.txt
```

---

## Requirements

- Python 3.10
- Qdrant (local vector database)
- Text Embeddings Inference server
- vLLM LLM server

All models and services run **locally**.

---

## Installation

Make sure uv is installed:

```bash
pip install uv
```

Install dependencies using **uv** :

```bash
uv sync
```

---

## Start Vector Database

Example using Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

---

## Start Embedding Server

See:

```
embedding_server啟動指令.txt
```

This embedding server runs locally and generates embeddings without using any external API.

---

## Start LLM Server

See:

```
llm_server啟動指令.txt
```

The LLM server runs locally to generate responses.

---

## Start API Server

```bash
python api_server.py
```

When the API server starts:

1. The system scans the `./documents` directory
2. PDF documents are processed
3. Each page is split into chunks
4. Chunks are embedded using the local embedding model
5. Embeddings are stored in the Qdrant vector database

---

## Data Processing

### Document Cleaning

The system performs several preprocessing steps before indexing:

1. **PDF text extraction**

   Documents are loaded using Haystack's `PyPDFToDocument`, which extracts
   textual content from PDF files.

2. **Text normalization**

   The system performs basic text normalization to improve embedding quality:

   - Remove null characters
   - Merge excessive whitespace
   - Normalize line breaks
   - Trim leading/trailing whitespace

3. **Formatting cleanup**

   Haystack's `DocumentCleaner` is used to remove common formatting artifacts
   produced during PDF extraction.

4. **Empty content filtering**

   Chunks with empty or extremely short content are discarded to avoid indexing
   low-quality vectors.

---

### Chunk Strategy

To preserve document structure while enabling effective retrieval, the system
uses a **two-stage chunking strategy**.

1. **Page-level split**

Documents are first split by **PDF page** in order to preserve the original
document structure and page references.

Metadata preserved:

- file name
- page number

2. **Word-based chunk split**

Each page is then further split into smaller chunks to improve retrieval
granularity.

Chunk configuration:

| Parameter | Value |
|----------|------|
| Split unit | word |
| Chunk length | 300 words |
| Chunk overlap | 50 words |

Overlap is used to reduce the risk of splitting important context across
chunk boundaries.

Additional metadata stored for each chunk:

- file name
- page number
- chunk index
- chunk word count

---

### Embedding Model

Embedding model : BAAI/bge-m3

Reason for selection:

- Strong multilingual retrieval capability
- Good performance for RAG tasks
- Suitable for local deployment

Embedding generation is handled by a **local Text Embeddings Inference server**.

Embedding configuration:

| Parameter        | Value                                     |
| ---------------- | ----------------------------------------- |
| Vector dimension | 1024                                      |
| Data type        | float32                                   |
| Normalization    | handled internally by the embedding model |

---

## Vector Database Configuration

Vector database: **Qdrant (local)**

Configuration:

| Parameter        | Value   |
| ---------------- | ------- |
| Vector dimension | 1024    |
| Data type        | float32 |
| Distance metric  | cosine  |
| Index type       | HNSW    |

Qdrant is used as the vector database for this project because:

- It supports efficient **approximate nearest neighbor search**
- It can run fully **locally without cloud services**
- It provides a simple API suitable for lightweight RAG systems

The **HNSW index** is used because it offers a good balance between
retrieval accuracy and query latency.

In this implementation, Qdrant's **default HNSW configuration** is used,
which is sufficient for small to medium document collections.

---

## Question Answering Pipeline

When a user submits a question:

1. The question is embedded using the local embedding model.
2. The vector is used to search the Qdrant database.
3. The top relevant document chunks are retrieved.
4. Retrieved chunks are inserted into the prompt.
5. The prompt is sent to the local LLM.
6. The LLM generates an answer using only retrieved context.

If no relevant information is found, the system returns:

```
無法回覆該問題
```

to avoid hallucinated answers.

### Token Control and Prompt Size

To prevent excessively large prompts being sent to the LLM, several controls
are applied:

- The number of retrieved chunks is limited by **top_k**
- Each chunk has a maximum size determined by the **chunk length**
- LLM output length is limited by **max_tokens**

These constraints ensure the prompt remains within reasonable context limits
while still providing sufficient information for answer generation.

---

## API

### Health Check

```
GET /health
```

Example response:

```json
{
  "status": "ok",
  "indexing": false
}
```

---

### Chat (RAG)

```
POST /chat
```

Request:

```json
{
  "question": "What is the document about?"
}
```

Response:

```json
{
  "answer": "...",
  "sources": [
    {
      "file_name": "example.pdf",
      "page_number": 3,
      "content": "retrieved document chunk"
    }
  ]
}
```

---

## Documents

Place PDF files into:

```
./documents
```

Each page will be indexed as an independent vector document.

The system will return **source file names and page numbers** together with the generated answers.

---

## Notes

This project focuses on demonstrating a **basic local RAG system** rather than a
production-ready system.

Simplifications include:

- Single API server
- No authentication
- No conversation memory
- No re-ranking model

A re-ranking model is not included in this implementation in order to keep the
system lightweight and easy to deploy locally for the assessment.

For small document collections, embedding-based retrieval is usually sufficient.
However, a cross-encoder re-ranker could be added in the future to improve
retrieval precision.

---

## Future Improvements

Potential improvements include:

- Conversation history support
- Streaming LLM responses
- Re-ranking model for improved retrieval
- Hybrid search (BM25 + vector search)
- Multi-user session support

---

## License

Apache License 2.0
