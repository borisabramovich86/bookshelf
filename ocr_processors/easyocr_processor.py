from ocr_processors.ocr_processor import OCRProcessor
import easyocr
import cv2
import datetime
import numpy as np
import os

class EasyOCRProcessor(OCRProcessor):
    def __init__(self, visualize):
        results_dir = "ocr_results/easyocr"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.filename = f"{results_dir}/{timestamp}"
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)

        self.__reader = easyocr.Reader(['en'])
        super().__init__(visualize)
    
    def create_ocr_results_string(self, ocr_results):
        detection_string = ""
        for detection in ocr_results:
            text = detection[1]
            confidence = float(detection[2])
            if confidence > 0.6:
                detection_string += f"{text} "
        return detection_string


    def process_all_masks(self, results, image_path):
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        for i, mask in enumerate(results.mask):
            mask = mask.astype(np.uint8) * 255
            processed_image = self.process_mask(original_image, mask)
            ocr_results = self.__reader.readtext(processed_image)

            if self.visualize:
                cv2.imshow(f"Segment {i+1}", processed_image)
                cv2.waitKey(0)
            
            with open(self.filename, 'a+') as f:
                header_string = f"OCR results for Segment {i+1}:"
                self.print_and_write(f, header_string)
                detection_string = self.create_ocr_results_string(ocr_results)
                self.print_and_write(f, detection_string)
                self.print_and_write(f, "------------------------------")
                    
            f.close()

        cv2.destroyAllWindows()


