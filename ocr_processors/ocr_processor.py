import cv2
import numpy as np
import os
import logging

from autodistill.utils import plot
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM

class OCRProcessor:
    def __init__(self, visualize, save_to_file):
        self.visualize = visualize
        self.save_to_file = save_to_file
        self.results_dir = "ocr_results"
        
        if save_to_file:
            if not os.path.exists(self.results_dir):
                os.mkdir(self.results_dir)
    
    def analyze_image(self, image, visualize):
        logging.info("Processing image using GroundedSAM model")
        results = None

        base_model = GroundedSAM(ontology=CaptionOntology({"book spine": "book spine"}), box_threshold=0.1)
        results = base_model.predict(image)
        
        if visualize:
            plot(
                image=image,
                classes=base_model.ontology.classes(),
                detections=results)
        
        return results
    
    def process_mask(self,original_image, mask):
        # Create a colored mask
        colored_mask = np.zeros_like(original_image)
        colored_mask[mask > 0] = [0, 255, 0]  # Green color for the segment

        # Blend the original image with the colored mask
        alpha = 0.7
        blended = cv2.addWeighted(original_image, alpha, colored_mask, 1 - alpha, 0)

        # Apply the mask to show only the selected area
        result = cv2.bitwise_and(blended, blended, mask=mask.astype(np.uint8))
        rotated = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
        colored = cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR)
        return colored
        

    def process_masks(self, results, image):
        pass

    def create_ocr_results_string():
        pass

    def process(self, image, visualize):
        pass