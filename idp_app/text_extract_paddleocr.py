from paddleocr import PaddleOCR
import cv2
import numpy as np
import os
import pdfplumber
import fitz
import tempfile
from PIL import Image, ImageFilter, ImageEnhance
from django.conf import settings

def preprocess_image(image):
    try:
        # Convert image to grayscale
        image = image.convert('L')

        # Apply image thresholding
        image = image.point(lambda x: 0 if x < 128 else 255, '1')

        # Apply Gaussian blur for noise removal
        image = image.filter(ImageFilter.GaussianBlur(radius=2))

        # Adjust contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)  # Increase contrast (adjust the factor as needed)

        # Resize image
        new_width, new_height = image.size  # Adjust dimensions as needed
        image = image.resize((new_width, new_height), Image.ANTIALIAS)
    except:
        image = image.convert('L')

    return image

def extract_text_from_file(file_path):
    file_extension = file_path.split('.')[-1]
    
    extracted_tables = []
    output_img_path = os.path.join(settings.BASE_DIR, 'idp_app', 'temp_file')
    # print(output_img_path)
    if file_extension == 'pdf':
        extracted_text = ''
        doc = fitz.open(file_path)
        i=0
        for page in doc:
            
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            # processed_image = preprocess_image(img)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
                temp_img_path = temp_img.name
                img.save(temp_img_path)
            ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
            result = ocr.ocr(temp_img_path, cls=False)
            extracted_text += "\n".join(row[1][0] for row in result[0])
            image = cv2.imread(temp_img_path)
            for row in result[0]:
                bbox = [[int(r[0]), int(r[1])] for r in row[0]]
                
                # write text output on image at each line
                cv2.putText(image, row[1][0], bbox[0], cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 0, 0), 1)
                cv2.polylines(image, [np.array(bbox)], True, (255, 0, 0), 1)
            i+=5
            # save image
            cv2.imwrite(output_img_path +'/'+'_'+ str(i) + '_'+'.jpg', image)

    elif file_extension in ['jpg', 'jpeg', 'png']:
        ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
        result = ocr.ocr(file_path, cls=False)
        extracted_text = "\n".join(row[1][0] for row in result[0])
        image = cv2.imread(file_path)
        for row in result[0]:
            bbox = [[int(r[0]), int(r[1])] for r in row[0]]
            
            # write text output on image at each line
            cv2.putText(image, row[1][0], bbox[0], cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 0, 0), 1)
            cv2.polylines(image, [np.array(bbox)], True, (255, 0, 0), 1)

        # save image
        cv2.imwrite(output_img_path+'.jpg', image)
    return extracted_text

