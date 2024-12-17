from ocr_processors.processor_factory import OCRProcessorFactory
from recommenders.recommender_factory import RecommenderFactory
import logging
import requests

def get_books_from_google_books(detected_books):
    for book in detected_books:
        isbn, author, link = "NULL", "NULL", "NULL"
        logging.info(f"Getting book from Google Book API: {book.title}")

        response = requests.get(
            f"https://www.googleapis.com/books/v1/volumes?q={book.title}",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response = response.json()
        
        try:
            first_result = response["items"][0]
            if "volumeInfo" in first_result:
                volume_info = first_result["volumeInfo"]
                isbn = volume_info["industryIdentifiers"][0]["identifier"]
                link = volume_info["infoLink"]
                if ("authors" in volume_info):
                    author = volume_info["authors"][0]
                
        except:
            logging.error(f"Error processing google result for {book}.title")

        logging.info(f"Results from google API: {isbn}, {author}, {link}")

def analyze_and_recommend(image, processor_type, recommender_type, visualize, save_to_file):

    logging.info(f'Processing image using: {processor_type} engine')
    processor_factory = OCRProcessorFactory()
    processor = processor_factory.get_processor(processor_type)
    processor_obj = processor(visualize, save_to_file)

    detected_books = processor_obj.process(image, visualize)
    get_books_from_google_books(detected_books.books)
    
    logging.info(f'Recommending books using: {recommender_type} engine')
    recommender_factory = RecommenderFactory()
    recommender = recommender_factory.get_recommender(recommender_type)
    recommender_obj = recommender()
    return recommender_obj.recommend(detected_books.books)