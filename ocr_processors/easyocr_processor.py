from ocr_processors.ocr_processor import OCRProcessor
import easyocr
import cv2
import datetime
import numpy as np
import os

class EasyOCRProcessor(OCRProcessor):
    def __init__(self, visualize, save_to_file):
        self.model = "easyocr"

        self.__reader = easyocr.Reader(['en'])
        super().__init__(visualize, save_to_file)
    
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
        detected_books = []

        for i, mask in enumerate(results.mask):
            mask = mask.astype(np.uint8) * 255
            processed_image = self.process_mask(original_image, mask)
            ocr_results = self.__reader.readtext(processed_image, rotation_info=[90, 180 ,270])

            if self.visualize:
                cv2.imshow(f"Segment {i+1}", processed_image)
                cv2.waitKey(0)
            
            
            header_string = f"OCR results for Segment {i+1}:"
            detection_string = self.create_ocr_results_string(ocr_results)

            detected_books.append(detection_string)

            print(header_string)
            print(detection_string)
            print("------------------------------")

            if self.save_to_file:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                filename = f"{self.results_dir}/{self.model}/{timestamp}"
                with open(filename, 'a+') as f:
                    header_string = f"OCR results for Segment {i+1}:"
                    f.write(header_string)
                    detection_string = self.create_ocr_results_string(ocr_results)
                    f.write(f"{detection_string}\n")
                    f.write("------------------------------")
                    
                f.close()

            return detected_books

        cv2.destroyAllWindows()

    def process(self, image_path, visualize):
        image = cv2.imread(image_path)
        sam_results = self.analyze_image(image_path, image, visualize)
        return self.process_all_masks(sam_results, image_path)