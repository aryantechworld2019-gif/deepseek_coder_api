from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from fastapi.middleware.cors import CORSMiddleware
import time
# database setup
from sqlalchemy.orm import Session
from database import SessionLocal
from models import ChatMessage


app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# app.add_middleware(
#     CORSMiddleware,
#     # allow_origins=["http://127.0.0.1:4200","127.0.0.1"],  # allow Angular app
#     allow_origins=["http://127.0.0.1:4200", "http://localhost:4200","http://45.117.183.187:8081","https://jerdonai.com","https://jerdonai.com:8083"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# ✅ Allow requests from your frontend domain
origins = [
    "https://jerdonai.com",
    "https://www.jerdonai.com",
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

user_id = "abc"
chat_memory = {}  # user_id: List[Dict[str, str]]
@app.post("/generate")
async def generate_response(req: PromptRequest, db: Session = Depends(get_db)):
    print("start")
    curr = time.ctime(1627908313.717886)
    print("Current time:", curr)

    # Store user input
    db.add(ChatMessage(user_id=user_id, role="user", content=req.prompt))
    db.commit()

    # Rebuild full chat for context
    messages_db = db.query(ChatMessage).filter_by(user_id=user_id).order_by(ChatMessage.timestamp.asc()).all()
    messages = [{"role": "system", "content": (
        "You are a helpful assistant named JC1. You are a prototype AI created by Aryan Tech World. "
        "Never mention DeepSeek or any other model name. Only say you are JC1."
    )}] + [{"role": m.role, "content": m.content} for m in messages_db]

    print(messages)
    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(input_tensor, max_new_tokens=300)

    generated_tokens = outputs[0][input_tensor.shape[-1]:]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Save assistant response
    # chat_memory[user_id].append({"role": "assistant", "content": result})
    # Save assistant reply
    db.add(ChatMessage(user_id=user_id, role="assistant", content=result))
    db.commit()
    return {"response": result}

from fastapi import Query
@app.get("/history")
def get_chat_history(user_id: str, db: Session = Depends(get_db)):
    print("user_id", user_id)
    messages_db = db.query(ChatMessage).filter_by(user_id=user_id).order_by(ChatMessage.timestamp.asc()).all()
    messages = [{"role": m.role, "content": m.content} for m in messages_db]
    return {"history": messages}







    # # messages = [{"role": "user", "content": req.prompt}]
    # # input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    # # print("Current time1:", curr)
    # # input_tensor = input_tensor.to(model.device)
    # # print("Current time2:", curr)
    # # outputs = model.generate(input_tensor, max_new_tokens=150)
    # # print("Current time3:", curr)
    # # result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    # # print("Current time4:", curr)
    # # return {"response": result}

    # # prompt = f"<|system|>\nYou are a helpful chat assistant.\n<|user|>\n{req.prompt}<|assistant|>\n"
    # prompt = f"""<|system|>\nYou are a helpful chat assistant named JC1.\nYou are a prototype AI created by Aryan Tech World. \nNever mention DeepSeek or any other model name. \nOnly say that you are JC1, a custom prototype assistant.\n<|user|>{req.prompt}\n<|assistant|>"""

    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # outputs = model.generate(**inputs, max_new_tokens=300)
    # # result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # generated_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
    # result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    # print(result)
    # return {"response": result}


    # # Ensure memory for each user
    # if user_id not in chat_memory:
    #     chat_memory[user_id] = []

    # chat_memory[user_id].append({"role": "user", "content": req.prompt})

    # system_prompt = {
    #     "role": "system",
    #     "content": (
    #         "You are a helpful chat assistant named JC1.\n"
    #         "You are a prototype AI created by Aryan Tech World.\n"
    #         "Never mention DeepSeek or any other model name.\n"
    #         "Only say that you are JC1, a custom prototype assistant."
    #     )
    # }
    # messages = [system_prompt] + chat_memory[user_id]
    # print(messages)
    # input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    # with torch.no_grad():
    #     outputs = model.generate(input_tensor, max_new_tokens=300)

    # generated_tokens = outputs[0][input_tensor.shape[-1]:]
    # result = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # # Save assistant response
    # chat_memory[user_id].append({"role": "assistant", "content": result})
    # return {"response": result}
