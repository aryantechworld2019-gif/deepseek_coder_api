from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import time

# --- Initialize app ---
app = FastAPI()

# --- Setup CORS for local Angular or JS apps ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use ["http://localhost:4200"] if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mount static and templates ---
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Load model ---
model_name = "../deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

# --- Simple chat memory store ---
chat_memory = {}  # user_id: List[Dict[str, str]]

# --- Schema for frontend prompt ---
class PromptRequest(BaseModel):
    prompt: str

# --- UI route ---
@app.get("/", response_class=HTMLResponse)
async def get_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- Generate response route ---
@app.post("/generate")
async def generate_response(req: PromptRequest):
    user_id = "abc"  # hardcoded for now

    # Init memory if not present
    if user_id not in chat_memory:
        chat_memory[user_id] = []

    # Add user message
    chat_memory[user_id].append({"role": "user", "content": req.prompt})

    # Construct prompt
    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful chat assistant named AL1.\n"
            "You are a prototype AI created by Argos Labs.\n"
            "Never mention DeepSeek or any other model name.\n"
            "Only say that you are AL1, a custom prototype assistant."
        )
    }
    messages = [system_prompt] + chat_memory[user_id]

    # Tokenize
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=300)

    # Decode
    generated_tokens = outputs[0][inputs.shape[-1]:]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Save assistant response
    chat_memory[user_id].append({"role": "assistant", "content": response_text})

    return {"messages": chat_memory[user_id]}
