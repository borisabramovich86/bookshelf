from ocr_processors.ocr_processor import OCRProcessor
import easyocr
import cv2
import numpy as np

class EasyOCRProcessor(OCRProcessor):
    def __init__(self):
        self.__reader = easyocr.Reader(['en'])
    
    def __process_mask(self,original_image, mask):
        # Create a colored mask
        colored_mask = np.zeros_like(original_image)
        colored_mask[mask > 0] = [0, 255, 0]  # Green color for the segment

        # Blend the original image with the colored mask
        alpha = 0.7
        blended = cv2.addWeighted(original_image, alpha, colored_mask, 1 - alpha, 0)

        # Apply the mask to show only the selected area
        result = cv2.bitwise_and(blended, blended, mask=mask.astype(np.uint8))

        # Perform OCR on the result
        ocr_result = self.__reader.readtext(result)
        return result, ocr_result
    
    def process_all_masks(self, results, image_path, visualize):
        # Load the original image
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Process each mask
        for i, mask in enumerate(results.mask):
            mask = mask.astype(np.uint8) * 255
            processed_image, ocr_result = self.__process_mask(original_image, mask)

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


