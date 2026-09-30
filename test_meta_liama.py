from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "./meta-llama/Meta-Llama-3-8B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a helpful assistant who understands Hinglish (Hindi + English mix)."},
    {"role": "user", "content": "Yaar, mujhe batao ki machine learning kya hoti hai, simple words mein samjhao."},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(device)
attention_mask = torch.ones_like(input_ids)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

with torch.no_grad():
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
    )

generated = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(generated, skip_special_tokens=True))
