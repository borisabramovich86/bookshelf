from ocr_processors.ocr_processor import OCRProcessor
import cv2
import pytesseract
import numpy as np

class TesseractProcessor(OCRProcessor):
    def __init__(self, visualize, save_to_file):
        super().__init__(visualize, save_to_file)

    def process_all_masks(self, results, image_path):
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        scale_factor = 300 / 72
        original_image = cv2.resize(original_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding
        _, original_image = cv2.threshold(original_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Remove noise (optional)
        original_image = cv2.medianBlur(original_image, 3)

        # Deskew (if needed)
        # angle = determine_skew(denoised)  # You'd need to implement this function
        # rotated = rotate_image(denoised, angle)  # You'd need to implement this function

        for i, mask in enumerate(results.mask):
            mask = mask.astype(np.uint8) * 255
            processed_image = self.process_mask(original_image, mask)
            ocr_result = pytesseract.image_to_string(processed_image, lang='eng')

            if self.visualize:
                cv2.imshow(f"Segment {i+1}", processed_image)
                cv2.waitKey(0)

            print(f"OCR results for Segment {i+1}:")
            print(ocr_result)

        cv2.destroyAllWindows()
