from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from fastapi.middleware.cors import CORSMiddleware
import time

from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import re
import requests
from bs4 import BeautifulSoup



app = FastAPI()

# --- Mount static files (like CSS, JS, images) ---
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Load templates from 'templates' folder ---
templates = Jinja2Templates(directory="templates")

# --- Render index.html on root path ---
@app.get("/", response_class=HTMLResponse)
async def get_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:4200","http://localhost:4200"],  # allow Angular app
    # allow_origins=["https://41ehjhh0gi5zq7-4000.proxy.runpod.net"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_name = "./deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

class PromptRequest(BaseModel):
    prompt: str

user_id = "abc"
chat_memory = {}  # user_id: List[Dict[str, str]]
@app.post("/generate")
async def generate_response(req: PromptRequest):
    # --- Step 1: Extract URL if present in prompt ---
    # url_match = re.search(r'https?://\S+', req.prompt)
    url_match = re.search(r'https?://[^\s\"\'<>]+', req.prompt)
    web_content = ""
    if url_match:
        url = url_match.group(0)
        web_content = extract_text_from_url(url)
        req.prompt = req.prompt.replace(url, f"[Web Content]: {web_content}")

    
    print(web_content)
    print("start")
    curr = time.ctime(1627908313.717886)
    print("Current time:", curr)


    # return {"response": result}


    # Ensure memory for each user
    if user_id not in chat_memory:
        chat_memory[user_id] = []

    chat_memory[user_id].append({"role": "user", "content": req.prompt})

    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful chat assistant named AL1.\n"
            "You are a prototype AI created by Argos Labs.\n"
            "Never mention DeepSeek or any other model name.\n"
            "Only say that you are AL1, a custom prototype assistant.\n"
        )
    }
    messages = [system_prompt] + chat_memory[user_id]
    print(messages)
    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(input_tensor, max_new_tokens=300)

    generated_tokens = outputs[0][input_tensor.shape[-1]:]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Save assistant response
    chat_memory[user_id].append({"role": "assistant", "content": result})
    # return {"response": result}
    return {"messages": chat_memory[user_id]}

def extract_text_from_url(url):
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return ' '.join(text.split())[:2000]  # Limit length
    except Exception as e:
        return f"[Error fetching content: {str(e)}]"


