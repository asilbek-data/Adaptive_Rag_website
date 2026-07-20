# Adaptive Agentic RAG — Website

A web version of the agent built in `Agentic_RAG_ready.ipynb`: a document
assistant that retrieves, **grades its own evidence**, falls back to web
search when the documents are weak, and **checks its own answer** for
hallucination before replying. Same LangGraph decision graph as the
notebook — just moved out of Colab into a deployable `backend/` (FastAPI)
and `frontend/` (Next.js).

Reference architecture: the project guide (`Agentic_RAG_Project_Guide_EN.html`)
and github.com/MazadiaS/adaptive-rag-assistant.

## Architecture

```
question -> retrieve -> grade_documents --yes--> generate -> grade_generation -> useful -> answer
                              |no                                |not grounded (regenerate)
                              v                                  |not useful (back to web)
                          web_search  ------------------------> generate
```

- `retrieve`: top-4 chunks from Qdrant.
- `grade_documents`: an LLM yes/no check per chunk; irrelevant chunks are dropped.
- `web_search`: Tavily fallback, only triggered when grading drops every chunk.
- `generate`: writes the answer from the surviving context.
- `grade_generation` (routing after `generate`): checks groundedness, then
  whether the answer resolves the question. A retry cap (`MAX_RETRIES`, default
  3) guarantees the loop always terminates.

## Project layout

```
backend/     FastAPI app wrapping the LangGraph agent — /chat, /ingest, /health
frontend/    Next.js chat UI — shows the agent's steps and citations
```

## Run locally

**Backend**
```bash
cd backend
cp .env.example .env      # fill in OPENAI_API_KEY (and TAVILY_API_KEY for web fallback)
pip install -r requirements.txt
uvicorn app.main:app --reload --port 7860
```

Ingest a document (equivalent to the notebook's `run_ingest(...)`):
```bash
curl -F "file=@/path/to/your.pdf" http://localhost:7860/ingest
```

**Frontend**
```bash
cd frontend
cp .env.example .env.local   # NEXT_PUBLIC_API_URL=http://localhost:7860
npm install
npm run dev
```
Open http://localhost:3000.

## Deploy (free tier, matching the guide)

**Backend -> Hugging Face Spaces (Docker)**
1. Create a Space, SDK = Docker.
2. Push the contents of `backend/` to the Space repo (the `Dockerfile` listens on port 7860).
3. In Space Settings -> Variables and secrets, set `OPENAI_API_KEY`, `TAVILY_API_KEY`,
   and — recommended, since Space storage isn't guaranteed to persist — a
   free Qdrant Cloud cluster's `QDRANT_URL` / `QDRANT_API_KEY`.
4. Confirm `https://<your-space>.hf.space/health` returns `{"status":"ok"}`.

**Frontend -> Vercel**
1. Import `frontend/` as a new Vercel project.
2. Set env var `NEXT_PUBLIC_API_URL` to your Space's URL.
3. Deploy.

## Evaluation — carried over from the notebook's "next steps"

Metrics to compute against a labeled question set (see the notebook for your
existing 12-question set from the BePro RAG work — reuse it here):

| Metric | How to measure |
|---|---|
| Retrieval hit rate | % of questions where a relevant chunk is in top-K |
| Groundedness | % of answers fully supported by cited context |
| Answer relevance | % of answers that actually resolve the question |
| Refusal correctness | Does it say "I don't know" when the answer isn't there? |
| Latency / cost | Avg time per query; tokens per query |

Experiments worth running and writing up: chunk size 500 vs 1000 vs 2000;
top-K = 2 vs 4 vs 8; agent with vs without `grade_documents`; with vs without
the web fallback.

## Notes

- The chat and vision model, embedding model, chunk size/overlap, and
  `MAX_RETRIES` all match the notebook's defaults (`gpt-4o-mini`,
  `text-embedding-3-small`, 1000/150, 3) — override via env vars in
  `backend/.env` if you want to run the chunk-size/top-K experiments above.
- Multimodal ingest (PDF images captioned by the vision model) is ported as-is
  from the notebook — no separate endpoint needed, it's part of `/ingest`.
"# Adaptive_Rag_website" 
