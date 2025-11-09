## RAG App (Streamlit): PDFs + Hybrid Retrieval + Local LLM + Web Fallback

Production‑ready, local‑first RAG with:

- Streamlit UI: multi‑PDF upload, QA, diagnostics, monitoring
- Hybrid chunking (paragraph + sentence windows)
- Hybrid retrieval (dense + TF‑IDF) with RRF, plus MMR / Cross‑Encoder rerank
- Local LLM (GPT4All / llama.cpp) with anti‑repetition and “detailed/concise” controls
- Optional Internet fallback (DuckDuckGo + requests + BeautifulSoup)
- Caching (diskcache) and structured logging with clear/reset


### Quickstart (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```


### Project Structure

```
.
├─ app.py                       # Streamlit UI and orchestration
├─ rag/
│  ├─ ingestion.py              # PDF extraction (PyMuPDF)
│  ├─ chunking.py               # Hybrid chunking
│  ├─ embeddings.py             # Sentence-Transformers
│  ├─ vectorstore.py            # Chroma or numpy+TF‑IDF store
│  ├─ retriever.py              # Hybrid retrieval + RRF + rerank
│  ├─ llm.py                    # GPT4All/llama.cpp + extractive fallback
│  ├─ web_search.py             # DuckDuckGo + requests + BeautifulSoup
│  ├─ cache.py, logger.py, types.py
├─ data/                        # uploads/chroma (created at runtime)
├─ logs/                        # rotating logs
├─ docs/                        # Architecture, config, performance, etc.
├─ .streamlit/config.toml       # server config
├─ requirements.txt, README.md, LICENSE, CHANGELOG.md
├─ CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md
└─ .gitignore
```


### How to Use

1) Upload PDFs → “Upload and Ingest PDFs”
2) Ask your question → choose Answer style (detailed/concise) and Target sentences
3) Retrieval settings → pick “hybrid”, candidate pool (20–40), rerank (MMR or cross‑encoder)
4) View “Retrieved Context and Citations” and the “Top sources” summary
5) Troubleshoot via Diagnostics (extraction, chunking preview, test retrieval) and Monitoring


### Configuration & Docs

- docs/Architecture.md — system overview
- docs/Configuration.md — sidebar options, envs
- docs/Performance.md — tuning guidance
- docs/Troubleshooting.md — common issues
- docs/Deployment.md — Windows/macOS/Linux
- docs/OCR.md — handling scanned PDFs


### Models

- Embeddings: sentence-transformers `all-MiniLM-L6-v2` (CPU friendly)
- GPT4All: place model under `%USERPROFILE%\.cache\gpt4all\` (e.g., `Mistral-7B-Instruct.Q4_0.gguf`)
- llama.cpp: set an absolute GGUF path (optional)


### Notes

- Vector store defaults to numpy+TF‑IDF for broad compatibility; Chroma can be enabled later.
- Anti‑repetition cleaning and decoding controls reduce duplicated lines.
- Internet fallback is optional; respect website terms/robots.


### Contributing
See CONTRIBUTING.md and CODE_OF_CONDUCT.md. Please open an issue before large changes.


### License
MIT — see LICENSE.


