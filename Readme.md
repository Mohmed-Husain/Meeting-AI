# Meeting AI Assistant

FastAPI + LangChain meeting assistant that ingests TXT/PDF/DOCX transcripts and produces
structured agendas, summaries, and action plans using a map-reduce summarization pipeline.

## Features
- Upload transcript files (TXT, PDF, DOCX)
- Clean transcript text while preserving speaker lines
- Map-reduce summarization for large transcripts
- Structured JSON output via Pydantic
- Simple frontend for upload + results + export

## Project Structure
```
meeting-ai/
├── app/
│   ├── ai/
│   │   ├── chunker.py
│   │   ├── prompts.py
│   │   ├── chains.py
│   │   ├── schemas.py
│   │   └── summarizer.py
│   ├── api/
│   │   └── routes.py
│   ├── utils/
│   │   └── file_loader.py
│   └── main.py
├── frontend/
│   └── index.html
├── .env
├── requirements.txt
└── Readme.md
```

## Environment Variables
- `LLM_PROVIDER`: `openai` (default) or `ollama`
- `OPENAI_API_KEY`: required for OpenAI
- `OPENAI_MODEL`: default `gpt-4o-mini`
- `OLLAMA_BASE_URL`: default `http://localhost:11434`
- `OLLAMA_MODEL`: default `llama3`
- `CHUNK_SIZE`: default `3000`
- `CHUNK_OVERLAP`: default `300`
- `MAP_CONCURRENCY`: default `4`
- `UPLOAD_DIR`: default `uploads`
- `STORAGE_BACKEND`: `local` (default), `s3` reserved for future use

## Run Locally
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Set your `.env` or environment variables, then run:
```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/` for the frontend.

## API
`POST /api/summarize`
- Form-data: `file` (TXT/PDF/DOCX)
- Response: structured JSON with agendas, summary, action plan

`GET /api/health`
- Response: `{ "status": "ok" }`

## Notes
S3 support is intentionally left as a future extension. The current version saves
uploads locally and processes them via the map-reduce pipeline.
