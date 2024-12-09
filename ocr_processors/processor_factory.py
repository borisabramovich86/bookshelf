from ocr_processors.easyocr_processor import EasyOCRProcessor
from ocr_processors.tesseract_processor import TesseractProcessor
from ocr_processors.ocr_processor import OCRProcessor
from ocr_processors.ollama_processor import OllamaProcessor

class OCRProcessorFactory:
    def get_processor(self, type: str) -> OCRProcessor:
        if type == 'easyocr':
            return EasyOCRProcessor
        elif type == 'tesseract':
            return TesseractProcessor
        elif type == 'ollama':
            return OllamaProcessor
        else:
            return ValueError(type)
        