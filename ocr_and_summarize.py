import os
import requests
from pathlib import Path
import pytesseract
from PIL import Image

# Configuration
DATA_DIR = Path("/Users/gunal/Developer/ai-data-engineering-assignment/data_upload")
API_URL = "http://localhost:8000"

def ocr_image(image_path: Path) -> str:
    """Run OCR on an image and return extracted text."""
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

def save_text(text: str, output_path: Path):
    """Save OCR text to a .txt file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

def upload_file(file_path: Path) -> dict:
    """Upload file to FastAPI /upload endpoint."""
    with open(file_path, "rb") as f:
        response = requests.post(f"{API_URL}/upload", files={"file": f})
    return response.json()

def summarize_file(filename: str, model="gpt-4") -> dict:
    """Call /summarize endpoint for uploaded file."""
    response = requests.post(f"{API_URL}/summarize?filename={filename}&model={model}")
    return response.json()

def main():
    for file_path in DATA_DIR.iterdir():
        if file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            print(f"Processing {file_path.name}...")

            # 1. OCR
            text = ocr_image(file_path)

            # 2. Save OCR text
            txt_file = DATA_DIR / f"{file_path.stem}.txt"
            save_text(text, txt_file)
            print(f"Saved OCR text to {txt_file.name}")

            # 3. Upload text to API
            upload_resp = upload_file(txt_file)
            print(f"Uploaded {txt_file.name}: {upload_resp}")

            # 4. Summarize
            summary = summarize_file(txt_file.name)
            print(f"Summary for {txt_file.name}:")
            print(summary)
            print("="*50)

if __name__ == "__main__":
    main()