from ocr_processors.ocr_processor import OCRProcessor
import cv2
import pytesseract
import numpy as np

class TesseractProcessor(OCRProcessor):
    def __init__(self, visualize):
        super().__init__(visualize)

    def process_all_masks(self, results, image_path):
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        for i, mask in enumerate(results.mask):
            mask = mask.astype(np.uint8) * 255
            processed_image = self.process_mask(original_image, mask)
            ocr_result = pytesseract.image_to_string(processed_image, lang='eng')

            # if self.__visualize:
            #     cv2.imshow(f"Segment {i+1}", processed_image)
            #     cv2.waitKey(0)

            print(f"OCR results for Segment {i+1}:")
            print(ocr_result)

        cv2.destroyAllWindows()
