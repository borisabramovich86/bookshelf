from ocr_processors.ocr_processor import OCRProcessor
from ollama import chat
from dtos.book_dto import DetectedBooks
import base64 
import logging

class OllamaProcessor(OCRProcessor):
    def __init__(self, visualize, save_to_file):
        self.model = "llama3.2-vision"
        super().__init__(visualize, save_to_file)
        
    def process_with_ollama(self, image_path):
        response = chat(
                model='llama3.2-vision',
                messages=[{
                    'role': 'user',
                    'content': 'This is a picture of a bookshelf. Please extract the author and title of each book',
                    'images': [image_path]
                }]
            )
        return response.message.content
    
    def process_with_pydantic_ollama(self, image):
        logging.info(f'Detecting books using {self.model}')
        image_base64 = base64.b64encode(image.read()).decode('utf-8')
        response = chat(
                model=self.model,
                format=DetectedBooks.model_json_schema(),
                messages=[{
                    'role': 'user',
                    'content': 'This is a picture of a bookshelf. Please extract the author and title of each book',
                    'images': [image_base64]
                }]
            )
        detected_books = DetectedBooks.model_validate_json(response.message.content)
        logging.info(f'Detected books: {detected_books}')
        return detected_books

    def process_image(self, image):
        detected_books_response = self.process_with_pydantic_ollama(image)
        print(detected_books_response)
        return detected_books_response
    
    def process(self, image, visualize):
        return self.process_image(image)

