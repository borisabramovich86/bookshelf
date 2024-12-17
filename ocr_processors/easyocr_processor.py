from ocr_processors.ocr_processor import OCRProcessor
from dtos.book_dto import DetectedBook, DetectedBooks
import easyocr
import cv2
import datetime
import numpy as np
import logging

import concurrent.futures
from functools import partial

class EasyOCRProcessor(OCRProcessor):
    def __init__(self, visualize, save_to_file):
        self.model = "easyocr"
        self.confidence = 0.6
        self.__reader = easyocr.Reader(['en'])

        super().__init__(visualize, save_to_file)
    
    def create_ocr_results_string(self, ocr_results):
        detection_string = ""
        for detection in ocr_results:
            text = detection[1]
            confidence = float(detection[2])
            if confidence > self.confidence:
                detection_string += f"{text} "
        return detection_string
    
    def save_to_file(self, ocr_results):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        filename = f"{self.results_dir}/{self.model}/{timestamp}"
        with open(filename, 'a+') as f:
            header_string = f"OCR results for Segment {i+1}:"
            f.write(header_string)
            detection_string = self.create_ocr_results_string(ocr_results)
            f.write(f"{detection_string}\n")
            f.write("------------------------------")
            
        f.close()

    def process_masks(self, results, image):
        logging.info("Processing masks and getting books info")
        image_np = np.array(image)
        original_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        detected_books = DetectedBooks(books=[])

        for i, mask in enumerate(results.mask):
            mask = mask.astype(np.uint8) * 255
            processed_image = self.process_mask(original_image, mask)
            ocr_results = self.__reader.readtext(processed_image, rotation_info=[90, 180 ,270])

            if self.visualize:
                cv2.imshow(f"Segment {i+1}", processed_image)
                cv2.waitKey(0)
            
            detection_string = self.create_ocr_results_string(ocr_results)
            if detection_string != '' and detection_string != 'ARAMAGO':
                detected_books.books.append(DetectedBook(title=detection_string, author=detection_string))

                logging.info(f"OCR results for Segment {i+1}:")
                logging.info(detection_string)
                logging.info("------------------------------")

        if self.save_to_file:
            self.save_to_file(ocr_results)
        
        logging.info(f"Detected {len(detected_books.books)} out of {len(results.mask)} books")
        logging.info(detected_books)
        cv2.destroyAllWindows()

        return detected_books

    
        logging.info("Processing masks and getting books info")
        image_np = np.array(image)
        original_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        detected_books = DetectedBooks(books=[])

        # Create a partial function with fixed arguments
        process_mask_partial = partial(self.process_single_mask, original_image=original_image)

        # Use ThreadPoolExecutor for I/O-bound tasks like OCR
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all tasks to the executor
            future_to_mask = {executor.submit(process_mask_partial, mask): i for i, mask in enumerate(results.mask)}

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_mask):
                i = future_to_mask[future]
                try:
                    detection_string = future.result()
                    if detection_string:
                        detected_books.books.append(DetectedBook(title=detection_string, author=detection_string))
                        logging.info(f"OCR results for Segment {i+1}:")
                        logging.info(detection_string)
                        logging.info("------------------------------")
                except Exception as exc:
                    logging.error(f'Mask {i} generated an exception: {exc}')

        logging.info(f"Detected {len(detected_books.books)} out of {len(results.mask)} books")
        logging.info(detected_books)
        cv2.destroyAllWindows()

        return detected_books
    
    def process(self, image, visualize):
        sam_results = self.analyze_image(image, visualize)
        return self.process_masks(sam_results, image)