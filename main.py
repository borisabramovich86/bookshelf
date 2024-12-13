import optparse
from ocr_processors.processor_factory import OCRProcessorFactory
from recommenders.recommender_factory import RecommenderFactory

def main():
    parser = optparse.OptionParser()
    parser.add_option("-i", "--image", dest="image", help="Image to process")
    parser.add_option("-v", "--visualize", help="Visualize after segmentation", default=False, action='store_true')
    parser.add_option("-s", "--save", help="Save OCR analysis to file", default=False, action='store_true')
    parser.add_option("-c", "--confidence", help="Set mininmum confidence for ocr results")
    parser.add_option("-p", "--processor", help="OCR processor type", default='ollama')
    parser.add_option("-r", "--recommender", help="Book recommender type", default='ollama')
    args, opts = parser.parse_args()

    print("Running with args:", args)

    book_images_dir = "resources/bookshelf_images"

    image_path = f"{book_images_dir}/{args.image}"
    processor_type = args.processor
    recommender_type = args.recommender
    save_to_file = args.save
    visualize = args.visualize
    min_confidence = args.confidence

    processor_factory = OCRProcessorFactory()
    processor = processor_factory.get_processor(processor_type)
    processor_obj = processor(visualize, save_to_file)

    detected_books = processor_obj.process(image_path, visualize)

    # recommender_factory = RecommenderFactory()
    # recommender = recommender_factory.get_recommender(recommender_type)
    # recommender.recommend()
    
if __name__ =="__main__":
    main()