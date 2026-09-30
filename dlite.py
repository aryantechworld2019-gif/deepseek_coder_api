from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "aisquared/dlite-v2-1_5b"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Create a text-generation pipeline (chat assistant style)
chat = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Sample prompt
prompt = "Human: Hi\nAI:"
response = chat(prompt, max_new_tokens=50, do_sample=True, temperature=0.7)
print(response[0]['generated_text'])
