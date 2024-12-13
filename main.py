import optparse
import cv2 
from analyze_and_recommend import analyze_and_recommend

def main():
    parser = optparse.OptionParser()
    parser.add_option("-i", "--image", dest="image", help="Image to process", default="books_1.jpg")
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

    image = cv2.imread(image_path)

    results = analyze_and_recommend(image, processor_type, recommender_type, visualize, save_to_file)
    print(results)
    
if __name__ =="__main__":
    main()