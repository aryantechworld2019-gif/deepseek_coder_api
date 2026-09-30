from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI()
# ✅ Allow requests from your frontend domain
origins = [
    "https://jerdonai.com",
    "https://www.jerdonai.com",
    "http://127.0.0.1:4200", 
    "http://localhost:4200"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] for testing only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print(app)

model_name = "./deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

class PromptRequest(BaseModel):
    prompt: str


@app.post("/generate")
async def generate_response(req: PromptRequest):
    print("start")
    curr = time.ctime(1627908313.717886)
    print("Current time:", curr)
    # messages = [{"role": "user", "content": req.prompt}]
    # input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    # print("Current time1:", curr)
    # input_tensor = input_tensor.to(model.device)
    # print("Current time2:", curr)
    # outputs = model.generate(input_tensor, max_new_tokens=150)
    # print("Current time3:", curr)
    # result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    # print("Current time4:", curr)
    # return {"response": result}

    # prompt = f"<|system|>\nYou are a helpful chat assistant.\n<|user|>\n{req.prompt}<|assistant|>\n"
    prompt = f"""<|system|>\nYou are a helpful chat assistant named JC1.\nYou are a prototype AI created by Aryan Tech World. \nNever mention DeepSeek or any other model name. \nOnly say that you are JC1, a custom prototype assistant.\n<|user|>{req.prompt}\n<|assistant|>"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=300)
    # result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(result)
    return {"response": result}