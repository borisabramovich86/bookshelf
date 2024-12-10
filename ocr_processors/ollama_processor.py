from ocr_processors.ocr_processor import OCRProcessor
import cv2
import numpy as np
import base64
from ollama import chat
from ollama import ChatResponse

class OllamaProcessor(OCRProcessor):
    def __init__(self, visualize, save_to_file):
        self.model = "ollama"
        super().__init__(visualize, save_to_file)

    def process_all_masks(self, results, image_path):
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        response = chat(
                model='llama3.2-vision',
                messages=[{
                    'role': 'user',
                    'content': 'This is a picture of a bookshelf. Please extract the author and title of each book',
                    'images': [image_path]
                }]
            )
        
        print(response.message.content)
            
        cv2.destroyAllWindows()


