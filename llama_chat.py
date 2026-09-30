import torch
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

DB_PATH = "chat_llama.db"
MODEL_NAME = "./meta-llama/Meta-Llama-3-8B-Instruct"
SYSTEM_PROMPT = (
    "You are a helpful assistant named JC1. "
    "You are a prototype AI created by Aryan Tech World. "
    "Never mention Meta, Llama, or any other model name. Only say you are JC1."
)

# --- DB ---

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user      TEXT NOT NULL,
            session   TEXT NOT NULL,
            role      TEXT NOT NULL,
            content   TEXT NOT NULL,
            ts        DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

def save_message(conn, user, session, role, content):
    conn.execute(
        "INSERT INTO chat_history (user, session, role, content) VALUES (?, ?, ?, ?)",
        (user, session, role, content),
    )
    conn.commit()

def load_history(conn, user, session):
    rows = conn.execute(
        "SELECT role, content FROM chat_history WHERE user = ? AND session = ? ORDER BY id",
        (user, session),
    ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]

# --- App lifespan ---

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    state["conn"] = get_conn()
    init_db(state["conn"])
    state["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_NAME)
    state["model"] = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    # Llama 3 uses <|eot_id|> as the end-of-turn terminator
    eot_id = state["tokenizer"].convert_tokens_to_ids("<|eot_id|>")
    state["terminators"] = [state["tokenizer"].eos_token_id, eot_id]
    state["executor"] = ThreadPoolExecutor(max_workers=1)
    yield
    state["executor"].shutdown(wait=False)
    state["conn"].close()

app = FastAPI(title="Llama 3 Chat API (JC1)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def serve_ui():
    return FileResponse(Path(__file__).parent / "chat.html")

# --- Schemas ---

class ChatRequest(BaseModel):
    username: str
    message: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    session_id: str
    response: str

class SessionInfo(BaseModel):
    session_id: str
    started: str
    message_count: int

class HistoryMessage(BaseModel):
    role: str
    content: str

# --- Inference ---

def run_inference(messages: list[dict]) -> str:
    tokenizer = state["tokenizer"]
    model = state["model"]
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    input_tensor = tokenizer.apply_chat_template(
        full_messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(
        input_tensor,
        max_new_tokens=300,
        eos_token_id=state["terminators"],
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)

# --- Endpoints ---

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    conn = state["conn"]
    session_id = req.session_id or str(uuid.uuid4())

    save_message(conn, req.username, session_id, "user", req.message)
    messages = load_history(conn, req.username, session_id)

    loop = __import__("asyncio").get_event_loop()
    response = await loop.run_in_executor(state["executor"], run_inference, messages)

    save_message(conn, req.username, session_id, "assistant", response)
    return ChatResponse(session_id=session_id, response=response)


@app.get("/sessions/{username}", response_model=list[SessionInfo])
def list_sessions(username: str):
    rows = state["conn"].execute(
        "SELECT session, MIN(ts) as started, COUNT(*) as cnt "
        "FROM chat_history WHERE user = ? GROUP BY session ORDER BY MIN(ts) DESC",
        (username,),
    ).fetchall()
    return [SessionInfo(session_id=r["session"], started=r["started"], message_count=r["cnt"]) for r in rows]


@app.get("/history/{username}/{session_id}", response_model=list[HistoryMessage])
def get_history(username: str, session_id: str):
    messages = load_history(state["conn"], username, session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="Session not found")
    return messages


@app.delete("/sessions/{username}/{session_id}")
def delete_session(username: str, session_id: str):
    state["conn"].execute(
        "DELETE FROM chat_history WHERE user = ? AND session = ?",
        (username, session_id),
    )
    state["conn"].commit()
    return {"deleted": session_id}
