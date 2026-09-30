from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "./microsoft/Phi-3-mini-128k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float32,  # CPU fallback
    device_map="auto"
)

prompt = """<|system|>
You are a helpful assistant.<|end|>
<|user|>
How to explain Internet for a medieval knight?<|end|>
<|assistant|> """
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))








# import torch 
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

# torch.random.manual_seed(0) 
# model = AutoModelForCausalLM.from_pretrained( 
#     "./microsoft/Phi-3-mini-128k-instruct",  
#     device_map="cuda",  
#     torch_dtype="auto",  
#     trust_remote_code=True,  
# ) 

# tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct") 

# messages = [ 
#     {"role": "system", "content": "You are a helpful AI assistant."}, 
#     {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"}, 
#     {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."}, 
#     {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"}, 
# ] 

# pipe = pipeline( 
#     "text-generation", 
#     model=model, 
#     tokenizer=tokenizer, 
# ) 

# generation_args = { 
#     "max_new_tokens": 500, 
#     "return_full_text": False, 
#     "temperature": 0.0, 
#     "do_sample": False, 
# } 

# output = pipe(messages, **generation_args) 
# print(output[0]['generated_text'])
