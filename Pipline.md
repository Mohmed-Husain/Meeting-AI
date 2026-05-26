

# AI Meeting Assistant — Complete System Architecture & Pipeline

---

# 1. Project Overview

The AI Meeting Assistant is a production-style AI system designed to process long meeting transcripts and generate structured meeting notes automatically.

The system is built using:

- FastAPI
- LangChain
- Ollama / OpenAI
- Map-Reduce Summarization Pipeline

The purpose of the system is to:

- accept meeting transcripts
- process very large transcripts efficiently
- summarize discussions
- generate agendas
- extract action items
- identify deadlines
- return structured JSON output
- display results in a frontend dashboard

---

# 2. Problem Statement

Large meeting transcripts create several problems:

- transcripts become too large for direct AI processing
- AI context windows become overloaded
- inference becomes slow
- API cost increases
- large prompts become unstable

Example:

```text
3-hour meeting
    ↓
50,000+ characters transcript
````

Sending this directly to an LLM is inefficient.

---

# 3. Solution Architecture

The system solves this using a:

# Map-Reduce AI Pipeline

Instead of sending the entire transcript to the AI at once, the system:

1. cleans the transcript
    
2. splits it into chunks
    
3. summarizes each chunk independently
    
4. combines partial summaries
    
5. generates final structured meeting notes
    

---

# 4. High-Level System Flow

```text
User Uploads Transcript
        ↓
FastAPI API Endpoint
        ↓
File Loader
        ↓
Transcript Extraction
        ↓
Transcript Cleaning
        ↓
Transcript Chunking
        ↓
MAP STEP
(Chunk Summarization)
        ↓
Partial Summaries
        ↓
REDUCE STEP
(Final Summarization)
        ↓
Structured JSON Output
        ↓
Frontend Dashboard
```

---

# 5. Folder Structure

```text
meeting-ai/
│
├── app/
│   ├── ai/
│   │   ├── chunker.py
│   │   ├── prompts.py
│   │   ├── chains.py
│   │   ├── schemas.py
│   │   └── summarizer.py
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
├── .env
├── requirements.txt
└── README.md
```

---

# 6. Responsibilities of Each File

---

# 6.1 routes.py

## Purpose

Acts as the API entry point.

Receives uploaded transcript files from frontend.

---

## Responsibilities

- receive transcript uploads
    
- validate request
    
- call summarizer service
    
- return structured JSON response
    

---

## Flow

```text
Frontend
    ↓
POST /api/summarize
    ↓
routes.py
```

---

# 6.2 file_loader.py

## Purpose

Extract transcript text from uploaded files.

---

## Supported File Types

- TXT
    
- PDF
    
- DOCX
    

---

## Responsibilities

- open uploaded file
    
- read file contents
    
- convert file into plain transcript text
    

---

## Flow

```text
meeting.pdf
      ↓
file_loader.py
      ↓
plain transcript text
```

---

# 6.3 summarizer.py

# MOST IMPORTANT FILE

This file acts as the:

# AI Orchestrator

It controls the entire AI pipeline.

---

## Responsibilities

- initialize AI model
    
- clean transcript
    
- call chunker
    
- run map summarization
    
- run reduce summarization
    
- generate final output
    
- log execution timings
    

---

## High-Level Flow

```text
Transcript
    ↓
Clean Transcript
    ↓
Chunk Transcript
    ↓
Map Summarization
    ↓
Reduce Summarization
    ↓
Final Structured Output
```

---

# 6.4 chunker.py

## Purpose

Split large transcripts into smaller AI-processable chunks.

---

## Why Chunking Exists

Large transcripts exceed efficient LLM processing limits.

Example:

```text
53,000 chars
    ↓
20 smaller chunks
```

---

## Responsibilities

- split transcript
    
- preserve context overlap
    
- optimize AI input size
    

---

## Technologies Used

```python
RecursiveCharacterTextSplitter
```

from LangChain.

---

## Example

```text
Chunk 1
Chunk 2
Chunk 3
...
Chunk 20
```

---

# 6.5 prompts.py

## Purpose

Store all prompts used by AI.

---

## Responsibilities

- map prompts
    
- reduce prompts
    
- output formatting instructions
    
- structured response instructions
    

---

## Example Prompt

```text
Analyze this meeting transcript chunk.
Extract:
- agendas
- action items
- deadlines
```

---

# 6.6 chains.py

## Purpose

Create LangChain workflows.

---

## Responsibilities

- create map chain
    
- create reduce chain
    
- connect prompts with LLM
    
- connect output parsers
    

---

## Flow

```text
Prompt
   ↓
LLM
   ↓
Structured Output
```

---

# 6.7 schemas.py

## Purpose

Define structured output models.

Uses:

```python
Pydantic
```

---

## Why Important

Without schemas:

- AI outputs become inconsistent
    
- frontend parsing becomes unstable
    

Schemas force AI into predictable structure.

---

## Example Schema

```json
{
  "agendas": [],
  "summary": [],
  "action_plan": []
}
```

---

# 6.8 logger.py

## Purpose

Centralized logging system.

---

## Responsibilities

- execution timing
    
- performance debugging
    
- AI inference monitoring
    
- error logging
    

---

## Example Logs

```text
Starting chunk 1/20
Completed chunk 1/20 in 22.14s
```

---

# 6.9 main.py

## Purpose

FastAPI application entry point.

---

## Responsibilities

- initialize FastAPI app
    
- register routes
    
- configure middleware
    
- start backend server
    

---

# 7. Detailed AI Pipeline

---

# 7.1 Transcript Upload

User uploads transcript from frontend.

Example:

```text
meeting.pdf
```

Frontend sends request:

```text
POST /api/summarize
```

---

# 7.2 File Extraction

Backend extracts transcript text.

Example:

```text
PDF
    ↓
Plain Text
```

---

# 7.3 Transcript Cleaning

System removes:

- filler words
    
- repeated spaces
    
- transcription noise
    
- formatting issues
    

---

## Example

Before:

```text
umm okay so deployment maybe friday
```

After:

```text
deployment scheduled for friday
```

---

# 7.4 Chunking

Transcript split into chunks.

Example:

```text
53,000 chars
      ↓
20 chunks
```

---

# 7.5 MAP STEP

Each chunk processed independently.

---

## Internal Flow

```text
Chunk
   ↓
Prompt Template
   ↓
LLM Inference
   ↓
Partial Summary
```

---

## AI Extracts

- agendas
    
- summary points
    
- action items
    
- decisions
    
- deadlines
    

---

## Example

```text
Chunk 1
   ↓
Partial Summary 1

Chunk 2
   ↓
Partial Summary 2
```

---

# 7.6 Partial Summaries

After map step:

```text
20 transcript chunks
       ↓
20 partial summaries
```

Now transcript becomes highly compressed.

---

# 7.7 REDUCE STEP

All partial summaries combined.

---

## Flow

```text
All Partial Summaries
          ↓
Reduce Prompt
          ↓
LLM
          ↓
Final Structured Meeting Notes
```

---

# 7.8 Final Structured Output

AI generates:

- AGENDAS
    
- SUMMARY
    
- ACTION PLAN
    
- DEADLINES
    

---

## Example Output

```json
{
  "agendas": [
    {
      "title": "Deployment Planning",
      "description": "Discussion about deployment timeline."
    }
  ],

  "summary": [
    "Backend API completed.",
    "Frontend dashboard pending."
  ],

  "action_plan": [
    {
      "task": "Complete dashboard UI",
      "assigned_to": "John",
      "deadline": "Friday"
    }
  ]
}
```

---

# 8. Frontend Workflow

Frontend responsibilities:

- upload transcript
    
- show loading state
    
- display agendas
    
- display summary
    
- display action plan
    
- export JSON
    

---

# 9. Current Performance Bottleneck

Current bottleneck:

# AI Inference Latency

Especially using:

```text
qwen3-vl:235b-cloud
```

---

# Current Timing Example

```text
Map Step:
~17 minutes

Reduce Step:
~2 minutes
```

---

# Root Cause

- very large model
    
- sequential chunk processing
    
- large prompts
    

---

# 10. Future Optimizations

---

# 10.1 Smaller Faster Models

Recommended:

- qwen2.5:7b
    
- llama3:8b
    
- mistral
    

---

# 10.2 Parallel Chunk Processing

Instead of:

```text
Chunk 1 → wait
Chunk 2 → wait
Chunk 3 → wait
```

Use:

```text
Chunk 1 ┐
Chunk 2 ├── simultaneous processing
Chunk 3 ┘
```

Huge performance improvement.

---

# 10.3 RAG Integration

Future support:

- vector databases
    
- transcript search
    
- meeting memory
    

---

# 10.4 LangGraph

Future enterprise workflow orchestration.

---

# 11. Final Mental Model

```text
Big Meeting Transcript
          ↓
Clean Transcript
          ↓
Split Into Chunks
          ↓
AI Reads Each Chunk
          ↓
Generate Partial Summaries
          ↓
Combine Partial Summaries
          ↓
Generate Final Meeting Report
          ↓
Display Structured Notes
```

---

# 12. Final Architecture Summary

This system is essentially:

```text
FastAPI
    +
LangChain
    +
Map-Reduce AI Pipeline
    +
Structured Outputs
    +
Frontend Dashboard
```

The architecture is scalable, modular, and production-oriented.



