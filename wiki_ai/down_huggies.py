from huggingface_hub import snapshot_download

# Which model you want:
repo_id = "deepseek-ai/deepseek-llm-7b-chat"
# "microsoft/Phi-3-mini-128k-instruct"
# "deepseek-ai/deepseek-coder-1.3b-instruct"
# "deepseek-ai/deepseek-coder-6.7b-instruct"
# "AtlaAI/Selene-1-Mini-Llama-3.1-8B"
# "distilbert-base-cased-distilled-squad"

# Where to save it (e.g., a subfolder in your current code directory)
local_model_dir = f"./{repo_id}"
# "./Selene-1-Mini"

# Download the entire model repo into local_model_dir
snapshot_download(repo_id=repo_id, local_dir=local_model_dir)