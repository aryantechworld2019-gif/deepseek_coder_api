from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
app = FastAPI()

# Use the chat-style instruction model
model_id = "./deepseek-ai/deepseek-coder-1.3b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # You can use float16 if you're on older GPUs
    device_map="auto"
)

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")

    # Chat template format
    messages = [
        {"role": "user", "content": prompt}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        inputs,
        max_new_tokens=512,
        do_sample=False,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    # Extract only the generated part (exclude input)
    response_text = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    code_blocks = response_text
    # re.findall(r"```python\n(.*?)```", response_text, re.DOTALL)[0]

    return {"output": code_blocks}
