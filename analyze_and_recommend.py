from ocr_processors.processor_factory import OCRProcessorFactory
from recommenders.recommender_factory import RecommenderFactory
import logging

def analyze_and_recommend(image, processor_type, recommender_type, visualize, save_to_file):

    logging.info(f'Processing image using: {processor_type} engine')
    processor_factory = OCRProcessorFactory()
    processor = processor_factory.get_processor(processor_type)
    processor_obj = processor(visualize, save_to_file)

    detected_books = processor_obj.process(image, visualize)
    
    logging.info(f'Recommending books using: {recommender_type} engine')
    recommender_factory = RecommenderFactory()
    recommender = recommender_factory.get_recommender(recommender_type)
    recommender_obj = recommender()
    return recommender_obj.recommend(detected_books.books)