from ocr_processors.processor_factory import OCRProcessorFactory
from recommenders.recommender_factory import RecommenderFactory
import logging

from dtos.book_dto import DetectedBooks

def analyze_and_recommend(image, processor_type, recommender_type, visualize, save_to_file):

    logging.info(f'Processing image using: {processor_type} engine')
    processor_factory = OCRProcessorFactory()
    processor = processor_factory.get_processor(processor_type)
    processor_obj = processor(visualize, save_to_file)

    detected_books = processor_obj.process(image, visualize)

    # sample_data = {
    #     "books": [
    #         {"title": 'How We Survived Communism and Even Laughed', "author": 'Slavenka Drakulić'},
    #         {"title": 'Wind and Truth', "author": 'SBrandon Sanderson'},
    #         {"title": 'Black Boy', "author": 'Richard Wright'},
    #         {"title": 'Throne of Glass', "author": 'Sarah J. Maas'},
    #         {"title": 'It Ends with Us', "author": 'Colleen Hoover'},
    #         {"title": 'Vita Nostra', "author": 'Marina Dyachenko'},
    #     ]
    # }

    # detected_books = DetectedBooks(**sample_data)
    
    logging.info(f'Recommending books using: {recommender_type} engine')
    recommender_factory = RecommenderFactory()
    recommender = recommender_factory.get_recommender(recommender_type)
    recommender_obj = recommender()
    return recommender_obj.recommend(detected_books.books)