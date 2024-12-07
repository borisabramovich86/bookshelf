from ocr_processors.easyocr_processor import EasyOCRProcessor
from ocr_processors.ocr_processor import OCRProcessor

class OCRProcessorFactory:
    def get_processor(self, type: str) -> OCRProcessor:
        if type == 'easyocr':
            return EasyOCRProcessor()
        else:
            return ValueError(type)
        