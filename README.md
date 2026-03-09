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
    - [Embedding Model](#embedding-model)
  - [Vector Database Configuration](#vector-database-configuration)
    - [Collection Design](#collection-design)
    - [Index Strategy](#index-strategy)
  - [Question Answering Pipeline](#question-answering-pipeline)
    - [Token Control and Prompt Size](#token-control-and-prompt-size)
    - [Retrieval Limits](#retrieval-limits)
    - [Chunk Size Control](#chunk-size-control)
    - [Prompt Construction Strategy](#prompt-construction-strategy)
      - [1. System Instruction](#1-system-instruction)
      - [2. Retrieved Document Context](#2-retrieved-document-context)
      - [3. User Question](#3-user-question)
    - [Prompt Length Control](#prompt-length-control)
    - [Output Token Limit](#output-token-limit)
  - [Model and Design Choices](#model-and-design-choices)
    - [LLM Selection](#llm-selection)
    - [Embedding Model Selection](#embedding-model-selection)
    - [Vector Database Selection](#vector-database-selection)
    - [Index Method Selection](#index-method-selection)
    - [Chunking Strategy Considerations](#chunking-strategy-considerations)
    - [Resource Constraints](#resource-constraints)
    - [Retrieval Design Choice](#retrieval-design-choice)
    - [Optimization Trade-offs](#optimization-trade-offs)
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
Answer + Retrieved Source Chunks
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

To preserve document structure while enabling effective retrieval, the system
uses a **two-stage chunking strategy**.

1. **Page-level split**

Documents are first split by **PDF page** in order to preserve the original
document structure and page references.

Metadata preserved:

- file name
- page number

Keeping page information allows the system to return **accurate source
references** when generating answers.

2. **Word-based chunk split**

Each page is then further split into smaller chunks to improve retrieval
granularity.

Chunk configuration:

| Parameter     | Value     |
| ------------- | --------- |
| Split unit    | word      |
| Chunk length  | 300 words |
| Chunk overlap | 50 words  |

Overlap is used to reduce the risk of splitting important context across
chunk boundaries.

Additional metadata stored for each chunk:

- file name
- page number
- chunk index
- chunk word count

Each chunk is stored as an **independent vector document** in the vector
database, allowing fine-grained semantic retrieval.

This design balances:

- **traceability** (via page metadata)
- **retrieval precision** (via smaller chunks)

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

### Collection Design

Each chunk generated during preprocessing is stored as a **single vector
document** in the Qdrant collection.

Each vector entry contains:

Vector:

- embedding vector (1024 dimensions)

Payload (metadata):

- file_name
- page_number
- chunk_index
- chunk_word_count
- content

This metadata allows the system to:

- return the **exact document source**
- provide **page-level references**
- display the **retrieved content snippet** in API responses.

### Index Strategy

Qdrant uses the **HNSW (Hierarchical Navigable Small World) index**
for approximate nearest neighbor search.

HNSW is chosen because:

- it provides **fast similarity search**
- it scales well with growing document collections
- it offers a good trade-off between **accuracy and latency**

In this implementation, the **default HNSW parameters** are used since the
expected dataset size is relatively small.

---

## Question Answering Pipeline

When a user submits a question, the system executes the following steps:

1. **Query Embedding**

The question is converted into a vector using the same embedding model
(`bge-m3`) used during document indexing.

2. **Vector Retrieval**

The system performs **dense retrieval** using the embedded query vector.

A `QdrantEmbeddingRetriever` retrieves the **top_k most similar document
chunks** from the Qdrant vector database based on cosine similarity.

3. **Context Construction**

The retrieved chunks are ordered by similarity score and used as context
for the LLM prompt.

4. **Prompt Construction**

The prompt is built using:

- system instruction
- retrieved document context
- user question

5. **Answer Generation**

The constructed prompt is sent to the local LLM (served by vLLM) which
generates the final answer.

If no relevant information is found, the system returns:

```
無法回覆該問題
```

to avoid hallucinated answers.

### Token Control and Prompt Size

To prevent excessively large prompts from being sent to the LLM, several
controls are applied.

### Retrieval Limits

The number of retrieved chunks is limited by **top_k** to avoid injecting
too much context into the prompt.

### Chunk Size Control

Each chunk has a maximum size defined by the chunking configuration
(300 words with overlap). This prevents individual context blocks from
becoming excessively large.

### Prompt Construction Strategy

The prompt consists of three main components.

#### 1. System Instruction

A system instruction is used to guide the LLM to strictly rely on the
retrieved document context when generating answers.

The instruction explicitly tells the model to:

- answer only based on the retrieved document content
- avoid fabricating or guessing information
- return **"無法回覆該問題"** if the provided context does not contain
  sufficient information to answer the question

#### 2. Retrieved Document Context

The top-k retrieved chunks from the vector database are used as the
knowledge context for the LLM.

The retrieved chunks are:

- ordered by similarity score
- concatenated into a context block
- formatted with source information such as file name and page number

This allows the LLM to prioritize the most relevant information and
preserve document traceability.

Example context structure:

    [Context]

    Document: example.pdf | Page: 3
    Retrieved chunk text...

    Document: example.pdf | Page: 4
    Retrieved chunk text...

#### 3. User Question

The user question is appended after the context block so the LLM can
generate an answer based on the retrieved information.

Example prompt layout:

    System Instruction

    Context
    (retrieved chunks)

    Question
    (user question)

### Prompt Length Control

To prevent excessively large prompts from exceeding the LLM context
window, several constraints are applied:

- only the **top_k retrieved chunks** are included
- each chunk has a fixed maximum size defined by the chunking strategy
- if the combined context becomes too large, lower-ranked chunks may be truncated

These controls ensure that the prompt remains within the allowed token
limit while still providing sufficient context for accurate answers.

This prompt design helps reduce hallucination and ensures that the LLM
acts primarily as a reasoning layer on top of retrieved knowledge.

### Output Token Limit

The maximum number of tokens generated by the LLM is limited using
`max_tokens` to ensure stable response latency and prevent excessively long
responses.

---

## Model and Design Choices

This system is designed to demonstrate a **fully local RAG architecture**
while keeping the implementation simple, reproducible, and suitable for
a technical assessment environment.

Several design decisions were made based on **deployment constraints,
performance considerations, and implementation simplicity**.

### LLM Selection

The local LLM used in this system is **Qwen3**.

Qwen3 was selected for the following reasons:

- It provides strong **Chinese language comprehension and generation**
- It performs well on **instruction-based question answering**
- It can handle **Chinese-first document retrieval and response generation**
- It can be deployed efficiently in a **fully local vLLM environment**

Because the example documents in this project are mainly **Chinese PDF
documents**, Chinese capability is a key consideration in model selection.

For a RAG pipeline, the quality of the final answer depends not only on
retrieval quality, but also on whether the LLM can accurately understand
the retrieved context and generate fluent answers in the same language.

Qwen3 is therefore a suitable choice for this project because it offers a
good balance between:

- Chinese language performance
- local deployment feasibility
- inference efficiency

### Embedding Model Selection

The embedding model used in this system is **BAAI/bge-m3**.

Reasons for choosing this model:

- Strong **multilingual semantic retrieval capability**
- High performance on many **retrieval benchmarks**
- Suitable for **local deployment**
- Compatible with **Text Embeddings Inference server**

The model produces **1024-dimensional dense vectors**, which provide
a good balance between semantic representation quality and memory usage.

### Vector Database Selection

**Qdrant** was selected as the vector database because:

- It supports **efficient approximate nearest neighbor search**
- It can run **fully locally without managed cloud services**
- It provides a **simple REST API** suitable for lightweight systems
- It integrates well with **Haystack retrievers**

This makes it a good fit for a **local RAG prototype**.

### Index Method Selection

Qdrant uses the **HNSW (Hierarchical Navigable Small World) index**.

HNSW is widely used in vector databases because it provides:

- fast similarity search
- high recall in approximate nearest neighbor retrieval
- good scalability for growing datasets

In this implementation, the **default HNSW parameters** are used since the
dataset size is expected to be relatively small.

### Chunking Strategy Considerations

A **two-stage chunking strategy** was adopted:

1. page-level split
2. word-level chunk split

This approach balances two goals:

- preserving **document traceability** (page references)
- improving **retrieval precision** (smaller semantic units)

Chunk overlap is used to reduce the risk of losing important context
across chunk boundaries.

### Resource Constraints

One important requirement of this system is that it must run **fully offline**
after deployment.

Therefore:

- the LLM is served locally using **vLLM**
- embeddings are generated locally using **Text Embeddings Inference**
- vector storage uses a **local Qdrant instance**

This avoids reliance on external APIs or cloud-hosted services.

### Retrieval Design Choice

The system uses **dense vector retrieval only**.

A **re-ranking model is not included** in this implementation to keep the
architecture lightweight and easy to deploy.

For small document collections, dense retrieval using high-quality
embeddings is often sufficient.

However, in larger production systems a **cross-encoder re-ranker**
could be added to improve retrieval precision.

### Optimization Trade-offs

The current design prioritizes:

- simplicity
- reproducibility
- local deployment compatibility

rather than maximum retrieval accuracy.

Potential improvements for production systems may include:

- hybrid retrieval (BM25 + vector search)
- cross-encoder re-ranking
- dynamic chunking strategies

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
