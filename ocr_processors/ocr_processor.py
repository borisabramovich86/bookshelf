import cv2
import numpy as np

class OCRProcessor:
    def __init__(self, visualize):
        self.visualize = visualize
    
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
    
    def print_and_write(self, file, text):
        print(text)
        file.write(f"{text}\n")


    def process_masks(self, results, image_path):
        pass

    def create_ocr_results_string():
        pass