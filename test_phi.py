import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "./microsoft/Phi-3-mini-128k-instruct"

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map="auto",
    attn_implementation="eager",
)
print("Model ready. Type 'quit' to exit.\n")

messages = [
    {"role": "system", "content": "You are a helpful assistant who understands Hinglish (Hindi + English mix)."},
]

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        break
    if not user_input:
        continue

    messages.append({"role": "user", "content": user_input})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    reply = tokenizer.decode(generated, skip_special_tokens=True)

    messages.append({"role": "assistant", "content": reply})
    print(f"\nPhi: {reply}\n")
