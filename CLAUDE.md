# DeepSeek Coder API — JC1 Chat Assistant

Self-hosted LLM chat API built with FastAPI and HuggingFace Transformers. Powers **JC1**, a custom-branded AI assistant for Aryan Tech World, deployed at jerdonai.com.

## System Configuration

| Component | Details |
|---|---|
| **CPU** | Intel Core i5-12400F (12th Gen) — 6 cores / 12 threads, max 5.6 GHz |
| **CPU ISA** | AVX2 — no AVX512, **no bfloat16 on CPU** |
| **RAM** | 32 GB total — ~23 GB available |
| **Swap** | 8 GB — fully exhausted; large models risk OOM |
| **Disk** | 457 GB NVMe SSD — 222 GB free |
| **GPU** | AMD Radeon RX 580 2048SP — **no CUDA, no ROCm configured** |
| **Inference device** | CPU only (`device_map="auto"` resolves to CPU) |
| **OS / Kernel** | Ubuntu, Linux 6.17.0-35-generic |
| **Python** | 3.12.3 |
| **PyTorch** | 2.7.1+cu126 (CUDA build, but CUDA unavailable — AMD GPU) |
| **Transformers** | 4.39.3 |
| **FastAPI** | 0.115.14 |
| **SQLAlchemy** | 2.0.41 |
| **Pydantic** | 2.11.7 |

### Important constraints from this hardware

- **Always use `torch_dtype=torch.float32`** — bfloat16/float16 require CUDA or AVX512, neither present.
- **Swap is fully used** — loading the 7B model (deepseek-llm-7b-chat) in float32 (~28 GB) will OOM. Use the 1.3B model for CPU-only work.
- **PyTorch CUDA build on AMD GPU** — `torch.cuda.is_available()` always returns `False`. Do not check CUDA availability to decide dtype.
- **No flash-attention** — not installed; safe to ignore the warning.

## Architecture

Local models are loaded at startup via `device_map="auto"` (resolves to CPU on this machine). All inference is synchronous (runs in a thread executor in `chat_test.py`). Chat history is persisted to SQLite (`chat_test.py`) or PostgreSQL (`main_1.py`).

## Entry Points

| File | Model | DB | Notes |
|---|---|---|---|
| `chat_test.py` | deepseek-llm-7b-chat | SQLite (`chat_ds.db`) | **Most complete** — multi-user, multi-session, serves `chat.html` UI |
| `main_1.py` | deepseek-llm-7b-chat | PostgreSQL | Per-user history API, `/history` endpoint |
| `main.py` | deepseek-llm-7b-chat | None | Stateless, JC1 branding, CORS for jerdonai.com |
| `app.py` | deepseek-coder-1.3b-base | None | Minimal code assistant |
| `app_1b.py` | deepseek-coder-1.3b-instruct | None | 1.3B instruct variant |

## Running the Server

```bash
# Recommended: full-featured chat with SQLite and built-in UI
uvicorn chat_test:app --host 0.0.0.0 --port 8000

# Production variant with PostgreSQL history
uvicorn main_1:app --host 0.0.0.0 --port 8000

# Lightweight stateless variant
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` — `chat_test.py` serves `chat.html` directly at `/`.

## API Endpoints (`chat_test.py`)

```
POST /chat                          → send a message, returns session_id + response
GET  /sessions/{username}           → list all sessions for a user
GET  /history/{username}/{session}  → full message history for a session
DELETE /sessions/{username}/{session} → delete a session
```

### Chat request body
```json
{ "username": "alice", "message": "Hello!", "session_id": null }
```
Omit `session_id` to start a new session; include it to continue an existing one.

## Local Models

Models are stored locally under `./deepseek-ai/`, `./microsoft/`, `./meta-llama/`.

| Path | Model | Size |
|---|---|---|
| `./deepseek-ai/deepseek-llm-7b-chat` | DeepSeek LLM 7B Chat | ~14 GB (bfloat16) |
| `./deepseek-ai/deepseek-coder-1.3b-instruct` | DeepSeek Coder 1.3B | ~2.6 GB |
| `./microsoft/Phi-3-mini-128k-instruct` | Microsoft Phi-3 Mini | ~7 GB |
| `./meta-llama/Meta-Llama-3-8B-Instruct` | Meta Llama 3 8B | ~16 GB |

### Downloading a model
```bash
python down_huggies.py   # edit repo_id inside the file before running
```

## Database

**SQLite** (used in `chat_test.py`):
- File: `chat_ds.db`
- Table: `chat_history(id, user, session, role, content, ts)`
- No migration needed — table is auto-created on startup

**PostgreSQL** (used in `main_1.py`):
- URL: `postgresql://jc1_user:jc1_pass@localhost/jc1_chat`
- Table: `chat_messages` — defined in `models.py`, created via `Base.metadata.create_all()`
- To initialize: uncomment the `create_all` lines at the bottom of `database.py`

## JC1 Branding

The assistant is branded as **JC1** for Aryan Tech World. System prompts must never leak the underlying model name. The canonical system prompt is in `main.py` and `main_1.py`:

> "You are a helpful assistant named JC1. You are a prototype AI created by Aryan Tech World. Never mention DeepSeek or any other model name. Only say you are JC1."

`chat_test.py` uses a different persona (`SYSTEM_PROMPT` constant at top of file — currently set to Hinglish mode).

## CORS

`main.py` and `main_1.py` restrict origins to `https://jerdonai.com` and `https://www.jerdonai.com`.
`chat_test.py` allows `*` (open) — tighten before production.

## Sub-projects

- `wiki_ai/` — Separate RAG-style wiki chatbot. Uses Jinja2 templates + static files. URL/PDF ingestion via `url_text.py` and `pdf_test.py`. Run from inside that directory.
- `aisquared/` — Separate project, check its own files.

## Test Scripts

```bash
python test_api.py       # basic POST /generate test
python test_phi.py       # run Phi-3 locally (no server)
python test_meta_liama.py  # run Llama 3 locally (no server)
python phi.py            # quick Phi-3 CPU inference test
```

## Stack

- Python 3.12
- FastAPI + Uvicorn
- PyTorch (GPU via `device_map="auto"`)
- HuggingFace Transformers
- SQLAlchemy (PostgreSQL path) / sqlite3 (SQLite path)
- Pydantic v2
- Virtual env: `/home/atw-server/Desktop/deepseek_coder_api/.venv`

## Virtual Environment

```bash
# Activate
source /home/atw-server/Desktop/deepseek_coder_api/.venv/bin/activate

# Deactivate
deactivate

# Run a script without activating
/home/atw-server/Desktop/deepseek_coder_api/.venv/bin/python test_phi.py

# Install a package into the venv
/home/atw-server/Desktop/deepseek_coder_api/.venv/bin/pip install <package>
```

Always activate the venv (or use its Python directly) before running any script in this project.

## Key Constraints

- Models load at **startup** — first request is not slow, but startup takes 30–90 seconds depending on GPU VRAM.
- Only one inference thread (`ThreadPoolExecutor(max_workers=1)`) — concurrent requests queue up.
- `max_new_tokens=300` is the generation cap across all entry points; increase carefully as it directly affects latency.
- `torch_dtype=torch.bfloat16` is used for 7B model (requires GPU with bfloat16 support); `float16` for 1.3B; `float32` for Phi-3 CPU fallback.
