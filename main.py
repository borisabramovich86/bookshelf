
import cv2
import optparse
import os.path
import easyocr
import pickle
import numpy as np

from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
from autodistill.utils import plot

from ocr_processors.processor_factory import OCRProcessorFactory

def read_image(image):
    image = cv2.imread(image)
    return image

def analyze_image(image_name, image, visualize):
    file_path = f"{image_name}_prediction_results.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            loaded_results = pickle.load(f)
            return loaded_results
    else:
        base_model = GroundedSAM(ontology=CaptionOntology({"book spine": "book spine"}), box_threshold=0.1)
        results = base_model.predict(image)

        with open(f"{image_name}_prediction_results.pkl", "wb") as f:
            pickle.dump(results, f)

        if visualize:
            plot(
                image=image,
                classes=base_model.ontology.classes(),
                detections=results)
                
        base_model.label("./context_images", extension=".jpeg")
        return results

def main():
    parser = optparse.OptionParser()
    parser.add_option("-i", "--image", dest="image", help="Image to process")
    parser.add_option("-v", "--visualize", help="Visualize after segmentation", default=False, action='store_true')
    parser.add_option("-s", "--save", help="Save OCR analysis to file", default=False, action='store_true')
    parser.add_option("-c", "--confidence", help="Set mininmum confidence for ocr results")
    parser.add_option("-t", "--type", help="OCR processor type", default='easyocr')
    args, opts = parser.parse_args()

    print("Running with args:", args)

    image_path = args.image
    processor_type = args.type
    save_to_file = args.save
    visualize = args.visualize
    min_confidence = args.confidence

    image = read_image(image_path)
    sam_results = analyze_image(image_path, image, args.visualize)

    processor_factory = OCRProcessorFactory()
    processor = processor_factory.get_processor(processor_type)
    processor_obj = processor(visualize, save_to_file)

    processor_obj.process_all_masks(sam_results, image_path)
    
if __name__ =="__main__":
    main()