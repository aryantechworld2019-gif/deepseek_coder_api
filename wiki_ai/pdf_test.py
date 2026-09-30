# import PyPDF2

# pdf_path = "/home/atw-server/Downloads/LCpy-Excel Basics-070725-085600.pdf"
# with open(pdf_path, "rb") as f:
#     reader = PyPDF2.PdfReader(f)

#     # Extract metadata
#     metadata = reader.metadata
#     print("📅 PDF Creation Date:", metadata.get("/CreationDate"))

#     # Extract text
#     all_text = ""
#     for page in reader.pages:
#         all_text += page.extract_text()

# print("📝 Extracted Text Preview:\n", all_text)
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import io

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image = Image.open(io.BytesIO(image_data_list[0]["bytes"]))
inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
