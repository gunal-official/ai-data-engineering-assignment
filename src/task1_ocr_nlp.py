#!/usr/bin/env python3
"""
Task 1: OCR Implementation and NLP Processing
- OCR text/image extraction with multiple methods
- Text cleaning and NLP feature extraction
- Named Entity Recognition and POS tagging
- OCR performance comparison and evaluation
"""

import os
import time
import cv2
import numpy as np
from pathlib import Path
import logging
import re
import nltk

# Optional external tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# OCR Libraries
import pytesseract
try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

try:
    import easyocr
except ImportError:
    easyocr = None

# NLP Libraries
try:
    import spacy
except ImportError:
    spacy = None

# Local imports
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRProcessor:
    """Unified OCR processor supporting multiple OCR engines"""

    def __init__(self):
        self.tesseract_config = '--oem 3 --psm 6'
        self.available_methods = ["tesseract"]

        # PaddleOCR
        if PaddleOCR is not None:
            try:
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
                self.available_methods.append("paddleocr")
            except Exception as e:
                logger.warning(f"PaddleOCR init failed: {e}")
                self.paddle_ocr = None
        else:
            self.paddle_ocr = None

        # EasyOCR
        if easyocr is not None:
            try:
                self.easy_reader = easyocr.Reader(['en'], gpu=False)
                self.available_methods.append("easyocr")
            except Exception as e:
                logger.warning(f"EasyOCR init failed: {e}")
                self.easy_reader = None
        else:
            self.easy_reader = None

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for better OCR accuracy"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 5)
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def tesseract_ocr(self, image_path: str) -> dict:
        """Extract text using Tesseract OCR safely"""
        try:
            start = time.time()
            processed_img = self.preprocess_image(image_path)
            text = pytesseract.image_to_string(processed_img, config=self.tesseract_config)

            data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
            confidences = []
            for conf in data['conf']:
                try:
                    conf_val = float(conf)
                    if conf_val > 0:
                        confidences.append(conf_val)
                except (ValueError, TypeError):
                    continue
            avg_conf = np.mean(confidences) if confidences else 0

            return {
                'method': 'tesseract',
                'text': text.strip(),
                'confidence': avg_conf / 100.0,
                'processing_time': time.time() - start,
                'success': True
            }
        except Exception as e:
            return {'method': 'tesseract', 'text': '', 'confidence': 0, 'processing_time': 0, 'success': False, 'error': str(e)}

    def paddleocr_ocr(self, image_path: str) -> dict:
        if self.paddle_ocr is None:
            return {'method': 'paddleocr', 'success': False, 'error': 'PaddleOCR not available'}
        try:
            start = time.time()
            results = self.paddle_ocr.ocr(image_path, cls=True)
            text, total_conf, count = "", 0, 0
            for line in results:
                for word_info in line:
                    _, (txt, conf) = word_info
                    text += txt + " "
                    total_conf += conf
                    count += 1
            avg_conf = total_conf / count if count > 0 else 0
            return {'method': 'paddleocr', 'text': text.strip(), 'confidence': avg_conf, 'processing_time': time.time() - start, 'success': True}
        except Exception as e:
            return {'method': 'paddleocr', 'success': False, 'error': str(e)}

    def easyocr_ocr(self, image_path: str) -> dict:
        if self.easy_reader is None:
            return {'method': 'easyocr', 'success': False, 'error': 'EasyOCR not available'}
        try:
            start = time.time()
            results = self.easy_reader.readtext(image_path)
            text = " ".join([t for (_, t, _) in results])
            avg_conf = np.mean([conf for (_, _, conf) in results]) if results else 0
            return {'method': 'easyocr', 'text': text.strip(), 'confidence': avg_conf, 'processing_time': time.time() - start, 'success': True}
        except Exception as e:
            return {'method': 'easyocr', 'success': False, 'error': str(e)}

    def extract_all_methods(self, image_path: str) -> list:
        results = []
        if "tesseract" in self.available_methods:
            results.append(self.tesseract_ocr(image_path))
        if "paddleocr" in self.available_methods:
            results.append(self.paddleocr_ocr(image_path))
        if "easyocr" in self.available_methods:
            results.append(self.easyocr_ocr(image_path))
        return results


class NLPProcessor:
    """NLP processing for text cleaning and feature extraction"""

    def __init__(self):
        if spacy is None:
            raise ImportError("spaCy not installed. Run: pip install spacy && python -m spacy download en_core_web_sm")
        try:
            self.nlp = spacy.load(Config.SPACY_MODEL)
        except OSError:
            raise OSError(f"spaCy model {Config.SPACY_MODEL} not found. Run: python -m spacy download {Config.SPACY_MODEL}")

        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
        return text.strip()


def main():
    print("🚀 Task 1: OCR + NLP Processing")
    Config.validate_config()
    Config.ensure_directories()

    ocr = OCRProcessor()
    try:
        nlp = NLPProcessor()
    except Exception as e:
        logger.error(f"NLP setup failed: {e}")
        return

    if WANDB_AVAILABLE:
        try:
            wandb.init(project=Config.WANDB_PROJECT, name="task1_experiment")
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")

    image_dir = Path(Config.SAMPLE_IMAGES_DIR)
    image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))

    if not image_files:
        print(f"⚠️ No images found in {image_dir}, add sample docs first.")
        return

    for img in image_files:
        print(f"\n🖼 Processing {img.name}")
        ocr_results = ocr.extract_all_methods(str(img))
        for res in ocr_results:
            if res['success']:
                print(f"✅ {res['method']} | Confidence: {res['confidence']:.2f} | Time: {res['processing_time']:.2f}s")
            else:
                print(f"❌ {res['method']} failed: {res.get('error')}")


if __name__ == "__main__":
    main()