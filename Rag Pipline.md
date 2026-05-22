
# AI Meeting Assistant — Updated Unified RAG Architecture & Pipeline

---

# 1. Project Overview

The AI Meeting Assistant is a production-oriented AI system designed to:

- process large meeting transcripts
- generate structured meeting notes
- store organizational meeting knowledge
- support semantic retrieval across historical meetings
- provide contextual AI-powered querying

The system combines:

- FastAPI
- LangChain
- OpenAI / Ollama
- OpenAI Embeddings
- ChromaDB
- Hierarchical Summarization
- Retrieval-Augmented Generation (RAG)

---

# 2. Core Goals

The system supports TWO major capabilities using ONE unified backend architecture.

---

# Capability 1 — Meeting Summarization

When users upload a transcript:

- extract transcript
- clean transcript
- chunk transcript
- summarize meeting
- generate:
  - agendas
  - summaries
  - action plans
  - deadlines

---

# Capability 2 — Historical Meeting Intelligence

Users can later ask questions such as:

```text
"What did we decide about deployment?"

"Show frontend-related discussions."

"What action items are still pending?"
````

without uploading a new transcript.

The AI retrieves information from previously uploaded meetings using semantic vector search.

---

# 3. High-Level Unified Architecture

```text
                    ┌─────────────────────┐
                    │   Frontend UI       │
                    └─────────┬───────────┘
                              │
                              ▼

                 ┌────────────────────────┐
                 │ Upload Transcript      │
                 │ OR Ask Question        │
                 └──────────┬─────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼

      ┌─────────────────┐      ┌─────────────────┐
      │ New Transcript? │      │ Existing Query? │
      └────────┬────────┘      └────────┬────────┘
               │                        │
               ▼                        ▼

      ┌─────────────────────────────────────────┐
      │ Shared Retrieval & Processing Layer     │
      └─────────────────┬───────────────────────┘
                        │
                        ▼

              ┌─────────────────────┐
              │ Extract Transcript  │
              └─────────┬───────────┘
                        ▼

              ┌─────────────────────┐
              │ Clean Transcript    │
              └─────────┬───────────┘
                        ▼

              ┌─────────────────────┐
              │ Dynamic Chunking    │
              └─────────┬───────────┘
                        ▼

              ┌─────────────────────┐
              │ Generate Embeddings │
              └─────────┬───────────┘
                        ▼

              ┌─────────────────────┐
              │ Store In ChromaDB   │
              └─────────┬───────────┘
                        ▼

         ┌────────────────────────────────┐
         │ Unified Retrieval Layer        │
         │ (semantic search engine)       │
         └─────────────┬──────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼

┌─────────────────────┐   ┌─────────────────────┐
│ Summarization Mode  │   │ Q&A / Chat Mode     │
└─────────┬───────────┘   └─────────┬───────────┘
          │                         │
          ▼                         ▼

┌─────────────────────┐   ┌─────────────────────┐
│ Hierarchical        │   │ Retrieve Relevant   │
│ Summarization       │   │ Chunks              │
└─────────┬───────────┘   └─────────┬───────────┘
          │                         │
          ▼                         ▼

┌─────────────────────┐   ┌─────────────────────┐
│ Structured Meeting  │   │ Context-Aware       │
│ Notes               │   │ AI Answer           │
└─────────────────────┘   └─────────────────────┘
```

---

# 4. Why Unified Architecture Is Better

Instead of creating two separate AI systems:

- summarization system
    
- retrieval system
    

the project uses ONE shared ingestion pipeline.

This reduces:

- duplicated chunking
    
- duplicated embeddings
    
- duplicated preprocessing
    
- repeated inference cost
    
- maintenance complexity
    

---

# 5. Core Architectural Principle

# Shared Semantic Knowledge Layer

The most important asset in the system is:

```text
Transcript Chunks + Embeddings + Metadata
```

stored inside ChromaDB.

Everything in the system operates on top of this knowledge layer.

---

# 6. Unified Backend Pipeline

---

# Step 1 — Upload Transcript

Frontend uploads:

- PDF
    
- TXT
    
- DOCX
    

using:

```text
POST /api/upload
```

---

# Step 2 — Extract Transcript

The backend extracts raw transcript text.

Supported formats:

- PDF
    
- TXT
    
- DOCX
    

---

# Step 3 — Clean Transcript

The transcript is normalized and cleaned.

Cleaning includes:

- removing filler words
    
- removing repeated spaces
    
- fixing formatting
    
- preserving speaker structure
    

---

# Step 4 — Dynamic Chunking

The transcript is split dynamically.

---

## Small Transcript

```text
1200 chars
    ↓
1 chunk only
```

---

## Large Transcript

```text
50,000 chars
    ↓
multiple optimized chunks
```

---

# Why Dynamic Chunking

Benefits:

- fewer AI calls
    
- lower latency
    
- lower cost
    
- faster summarization
    

---

# Step 5 — Generate Embeddings

Each chunk is converted into vector embeddings using:

```text
OpenAI Embeddings
```

Recommended model:

```text
text-embedding-3-small
```

---

# Step 6 — Store In ChromaDB

Each transcript chunk is stored with:

- chunk text
    
- embeddings
    
- metadata
    

---

# Example Stored Metadata

```json
{
  "meeting_id": "meeting_001",
  "chunk_index": 4,
  "speaker": "Ali",
  "date": "2026-05-22"
}
```

---

# IMPORTANT

Embeddings are generated ONLY ONCE.

This is a major production optimization.

The system never needs to:

- reprocess transcript
    
- regenerate embeddings
    
- rechunk old meetings
    

---

# Step 7 — Unified Retrieval Layer

The retrieval layer becomes the central semantic engine.

Responsibilities:

- semantic search
    
- similarity matching
    
- historical retrieval
    
- contextual retrieval
    

---

# 7. Summarization Mode

Used immediately after transcript upload.

---

# Flow

```text
Transcript Chunks
       ↓
Hierarchical Summarization
       ↓
Structured Meeting Notes
```

---

# Hierarchical Summarization

Instead of one massive reduce step:

```text
20 chunks
    ↓
20 summaries
    ↓
5 grouped summaries
    ↓
Final summary
```

Benefits:

- smaller contexts
    
- faster inference
    
- scalable summarization
    
- lower model overload
    

---

# Final Generated Output

The AI generates:

- agendas
    
- summaries
    
- action plans
    
- deadlines
    
- decisions
    

using structured Pydantic outputs.

---

# Example Output

```json
{
  "agendas": [],
  "summary": [],
  "action_plan": [],
  "deadlines": []
}
```

---

# 8. Retrieval / Chat Mode

Used when users ask questions about historical meetings.

---

# Example Queries

```text
"What deployment decisions were made?"

"What action items are pending?"

"Show frontend-related discussions."
```

---

# Query Flow

```text
User Query
      ↓
Generate Query Embedding
      ↓
ChromaDB Similarity Search
      ↓
Retrieve Relevant Chunks
      ↓
Inject Context Into Prompt
      ↓
LLM Generates Final Answer
```

---

# Important RAG Principle

The AI does NOT memorize meetings.

Instead:

- retrieve relevant chunks
    
- inject context dynamically
    
- generate context-aware answers
    

---

# 9. Folder Structure

```text
meeting-ai/
│
├── app/
│   ├── ai/
│   │   ├── chunker.py
│   │   ├── cleaner.py
│   │   ├── prompts.py
│   │   ├── chains.py
│   │   ├── schemas.py
│   │   ├── summarizer.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   └── rag_chain.py
│   │
│   ├── api/
│   │   └── routes.py
│   │
│   ├── utils/
│   │   └── file_loader.py
│   │
│   ├── core/
│   │   └── logger.py
│   │
│   └── main.py
│
├── frontend/
│   └── index.html
│
├── chroma_db/
│
├── .env
├── requirements.txt
└── README.md
```

---

# 10. Responsibilities of New Components

---

# embeddings.py

Generate vector embeddings for transcript chunks.

---

# vector_store.py

Store:

- embeddings
    
- transcript chunks
    
- metadata
    

inside ChromaDB.

---

# retriever.py

Retrieve semantically relevant transcript chunks.

---

# rag_chain.py

Build Retrieval-Augmented Generation workflows.

---

# 11. AI Models

---

# LLM Models

Used for:

- summarization
    
- question answering
    
- structured outputs
    

Recommended:

- gpt-4.1-mini
    
- gpt-5.4-mini
    
- qwen2.5:7b
    

---

# Embedding Models

Used ONLY for vector generation.

Recommended:

```text
text-embedding-3-small
```

---

# 12. Frontend Responsibilities

The frontend supports:

- transcript upload
    
- summary visualization
    
- agenda rendering
    
- action item rendering
    
- historical query interface
    
- semantic search UI
    

---

# 13. Future Scalability

The architecture supports future enterprise features:

- Slack integrations
    
- Jira integrations
    
- LangGraph workflows
    
- Multi-agent systems
    
- Cloud storage
    
- User authentication
    
- Team workspaces
    
- Meeting analytics
    

---

# 14. Final Mental Model

```text
Upload Transcript
        ↓
Extract & Clean Transcript
        ↓
Dynamic Chunking
        ↓
Generate Embeddings
        ↓
Store Semantic Knowledge
        ↓
Summarize Meeting
        ↓
Enable Future Retrieval
        ↓
Context-Aware AI Assistant
```

---

# 15. Final Architecture Summary

The final architecture combines:

```text
FastAPI
    +
LangChain
    +
Dynamic Chunking
    +
Hierarchical Summarization
    +
OpenAI Embeddings
    +
ChromaDB
    +
RAG
    +
Structured Outputs
    +
Semantic Retrieval
    +
Frontend Dashboard
```

The system is modular, scalable, cost-efficient, RAG-enabled, and production-oriented.