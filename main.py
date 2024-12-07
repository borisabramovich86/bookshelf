import cv2
import optparse

import os.path
import easyocr
import pickle

import numpy as np

from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
from autodistill.utils import plot

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

def process_mask(original_image, mask, reader):
    # Create a colored mask
    colored_mask = np.zeros_like(original_image)
    colored_mask[mask > 0] = [0, 255, 0]  # Green color for the segment

    # Blend the original image with the colored mask
    alpha = 0.7
    blended = cv2.addWeighted(original_image, alpha, colored_mask, 1 - alpha, 0)

    # Apply the mask to show only the selected area
    result = cv2.bitwise_and(blended, blended, mask=mask.astype(np.uint8))

    # Perform OCR on the result
    ocr_result = reader.readtext(result)

    return result, ocr_result

def process_all_masks(results, image_path, visualize):
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Load the original image
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Process each mask
    for i, mask in enumerate(results.mask):
        mask = mask.astype(np.uint8) * 255
        processed_image, ocr_result = process_mask(original_image, mask, reader)

        # Display or save the processed image
        rotated = cv2.rotate(processed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        colored = cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR)
        if visualize:
            cv2.imshow(f"Segment {i+1}", colored)
            cv2.waitKey(0)

        # Print OCR results
        print(f"OCR results for Segment {i+1}:")
        for detection in ocr_result:
            print(f"Text: {detection[1]}, Confidence: {detection[2]}")

    cv2.destroyAllWindows()

def main():
    parser = optparse.OptionParser()
    parser.add_option("-i", "--image", dest="image", help="Image to process")
    parser.add_option("-v", "--visualize", help="Visualize after segmentation", default=False)
    args = parser.parse_args()

    print(args)

    image_name = args.image
    image = read_image(image_name)
    sam_results = analyze_image(image_name, image, args.visualize)
    process_all_masks(sam_results, image_name, args.visualize)
    
if __name__ =="__main__":
    main()