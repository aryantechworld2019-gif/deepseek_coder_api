from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

model_id = "./deepseek-ai/deepseek-coder-1.3b-base"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = f"<|system|>\nYou are a helpful code assistant.\n<|user|>\n{data['prompt']}<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=300)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"output": result}


