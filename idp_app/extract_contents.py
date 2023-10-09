from django.conf import settings
import os
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# from paddleocr import PaddleOCR, draw_ocr
from img2table.ocr import PaddleOCR
from img2table.document import Image as Image_table

from transformers import TableTransformerForObjectDetection
import torch
from PIL import Image
from transformers import DetrFeatureExtractor



def intersection(box_1, box_2):
    return [box_2[0], box_1[1],box_2[2], box_1[3]]

def iou(box_1, box_2):

  x_1 = max(box_1[0], box_2[0])
  y_1 = max(box_1[1], box_2[1])
  x_2 = min(box_1[2], box_2[2])
  y_2 = min(box_1[3], box_2[3])

  inter = abs(max((x_2 - x_1, 0)) * max((y_2 - y_1), 0))
  if inter == 0:
      return 0

  box_1_area = abs((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]))
  box_2_area = abs((box_2[2] - box_2[0]) * (box_2[3] - box_2[1]))

  return inter / float(box_1_area + box_2_area - inter)

def paddle_table_extract(folder_path):
    ocr = PaddleOCR(lang='en')
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Check if the file is an image
            image_path = os.path.join(folder_path, filename)
    
            image_cv = cv2.imread(image_path)
            image_height = image_cv.shape[0]
            image_width = image_cv.shape[1]
            output = ocr.ocr(image_path)[0]
            boxes = [line[0] for line in output]
            texts = [line[1][0] for line in output]
            probabilities = [line[1][1] for line in output]
            image_boxes = image_cv.copy()
            for box,text in zip(boxes,texts):
                cv2.rectangle(image_boxes, (int(box[0][0]),int(box[0][1])), (int(box[2][0]),int(box[2][1])),(0,0,255),1)
                cv2.putText(image_boxes, text,(int(box[0][0]),int(box[0][1])),cv2.FONT_HERSHEY_SIMPLEX,1,(222,0,0),1)
            
            #Get Horizontal and Vertical Lines
            
            im = image_cv.copy() 
            horiz_boxes = []
            vert_boxes = []

            for box in boxes:
                x_h, x_v = 0,int(box[0][0])
                y_h, y_v = int(box[0][1]),0
                width_h,width_v = image_width, int(box[2][0]-box[0][0])
                height_h,height_v = int(box[2][1]-box[0][1]),image_height

                horiz_boxes.append([x_h,y_h,x_h+width_h,y_h+height_h])
                vert_boxes.append([x_v,y_v,x_v+width_v,y_v+height_v])

                cv2.rectangle(im,(x_h,y_h), (x_h+width_h,y_h+height_h),(0,0,255),1)
                cv2.rectangle(im,(x_v,y_v), (x_v+width_v,y_v+height_v),(0,255,0),1)
        
            #non-max-supression
        
            horiz_out = tf.image.non_max_suppression(
                horiz_boxes,
                probabilities,
                max_output_size = 1000,
                iou_threshold=0.1,
                score_threshold=float('-inf'),
                name=None
            )
            horiz_lines = np.sort(np.array(horiz_out))
            im_nms = image_cv.copy()
            for val in horiz_lines:
                cv2.rectangle(im_nms, (int(horiz_boxes[val][0]),int(horiz_boxes[val][1])), (int(horiz_boxes[val][2]),int(horiz_boxes[val][3])),(0,0,255),1)
            vert_out = tf.image.non_max_suppression(
                vert_boxes,
                probabilities,
                max_output_size = 1000,
                iou_threshold=0.1,
                score_threshold=float('-inf'),
                name=None
            )
            vert_lines = np.sort(np.array(vert_out))
            for val in vert_lines:
                cv2.rectangle(im_nms, (int(vert_boxes[val][0]),int(vert_boxes[val][1])), (int(vert_boxes[val][2]),int(vert_boxes[val][3])),(255,0,0),1)
        
            #convert to csv
        
            out_array = [["" for i in range(len(vert_lines))] for j in range(len(horiz_lines))]
            unordered_boxes = []

            for i in vert_lines:
            # print(vert_boxes[i])
                unordered_boxes.append(vert_boxes[i][0])
            ordered_boxes = np.argsort(unordered_boxes)
            for i in range(len(horiz_lines)):
                for j in range(len(vert_lines)):
                    resultant = intersection(horiz_boxes[horiz_lines[i]], vert_boxes[vert_lines[ordered_boxes[j]]] )

                    for b in range(len(boxes)):
                        the_box = [boxes[b][0][0],boxes[b][0][1],boxes[b][2][0],boxes[b][2][1]]
                        if(iou(resultant,the_box)>0.1):
                            out_array[i][j] = texts[b]
            out_array=np.array(out_array)

            output_csv_path=os.path.join(settings.BASE_DIR, 'idp_app', 'static','csv_folder',f'sample{filename}.csv')

            pd.DataFrame(out_array).to_csv(output_csv_path)


def table_detect(image_path):
    image = Image.open(image_path).convert("RGB")
    feature_extractor = DetrFeatureExtractor()
    encoding = feature_extractor(image, return_tensors="pt")
    encoding.keys()
    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
    with torch.no_grad():
        outputs = model(**encoding)
    width, height = image.size
    results = feature_extractor.post_process_object_detection(outputs, threshold=0.75, target_sizes=[(height, width)])[0]
    return results


def img_2_table_extract(folder_path):


    paddle = PaddleOCR(lang="en")
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Check if the file is an image
            image_path = os.path.join(folder_path, filename)
            print("IMMAGGEEEPATTHH:::",image_path)
            img = Image_table(image_path)
            tables = img.extract_tables(ocr=paddle,implicit_rows=True, borderless_tables=True)
            for i in range(len(tables)):
                dataf = tables[i].df
                csv_folder = os.path.join(settings.BASE_DIR, 'idp_app', 'static','csv_folder',f'sample{filename}.csv')
                dataf.to_csv(csv_folder)

