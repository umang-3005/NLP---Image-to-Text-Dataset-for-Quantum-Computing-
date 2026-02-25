"""
Local Visual Extractor Module.
Uses Tesseract OCR to read text inside circuit diagrams without external APIs.
"""
import re
import pytesseract
from PIL import Image
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter
# Import existing GateVocabulary to map OCR results to canonical names
from gate_extraction import GateVocabulary
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)

class LocalVisualExtractor:
    """
    Extracts information from circuit images using local OCR.
    """
    def __init__(self):
        self.vocabulary = GateVocabulary()
        # Filter out OCR noise (single lines, dots, common artifacts in diagrams)
        self.noise_pattern = re.compile(r'^[\s\|\-\_1lI\.,:;]+$')

    def extract_gates_from_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Runs local OCR on the image to find gate labels.
        """
        try:
            if not image_path.exists():
                logger.warning(f"Image not found for OCR: {image_path}")
                return {"detected_gates": [], "gate_counts": {}, "raw_ocr_text": ""}

            img = Image.open(image_path)
            
            # Run OCR. --psm 11 is for sparse text (diagrams).
            try:
                raw_text = pytesseract.image_to_string(img, config='--psm 11')
            except Exception:
                raw_text = pytesseract.image_to_string(img)
            
            # Filter and Map OCR text to Gate Names
            found_gates = []
            words = raw_text.split()
            
            for word in words:
                word = word.strip()
                # Skip pure noise and very short tokens
                if len(word) < 1 or self.noise_pattern.match(word):
                    continue
                
                # Normalize using your existing vocabulary (e.g., "Rz" -> "RZ")
                normalized = self.vocabulary.normalize(word)
                if normalized:
                    found_gates.append(normalized)
            
            # Deduplicate and count
            gate_counts = dict(Counter(found_gates))
            detected_gates = sorted(list(gate_counts.keys()))
            
            if detected_gates:
                logger.debug(f"OCR found gates in {image_path.name}: {detected_gates}")
            
            return {
                "detected_gates": detected_gates,
                "gate_counts": gate_counts,
                "raw_ocr_text": raw_text
            }
            
        except Exception as e:
            logger.error(f"Local OCR failed for {image_path}: {e}")
            return {"detected_gates": [], "gate_counts": {}, "raw_ocr_text": ""}
