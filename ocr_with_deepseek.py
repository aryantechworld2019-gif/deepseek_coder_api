import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import AutoModel, AutoTokenizer
import torch

MODEL_PATH = "./deepseek-ai/DeepSeek-OCR"
IMAGE_FILE = r"/home/atw-server/Downloads/ChatGPT Image Jun 2, 2026, 12_11_22 PM.png"
OUTPUT_DIR = "./ocr_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Loading model...")
model = AutoModel.from_pretrained(
    MODEL_PATH,
    _attn_implementation="eager",
    trust_remote_code=True,
    use_safetensors=True,
    device_map="cpu",
)
# Convert all weights/buffers to float32 for CPU compatibility (no bfloat16 on AVX2)
model = model.eval().float()
print("Model ready.\n")

print(f"Running OCR on: {IMAGE_FILE}")
result = model.infer(
    tokenizer,
    prompt="<image>\nFree OCR. ",
    image_file=IMAGE_FILE,
    output_path=OUTPUT_DIR,
    base_size=512,
    image_size=512,
    crop_mode=False,
    save_results=False,
    test_compress=False,
    eval_mode=True,
)

print("\n=== OCR RESULT ===\n")
print(result)
