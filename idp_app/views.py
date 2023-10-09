from django.shortcuts import render
from .info_extract import main_inference
from django.conf import settings
import os
import cv2
import shutil
import json
from django.template.loader import render_to_string
import json
from .extract_contents import paddle_table_extract , table_detect , img_2_table_extract
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from rest_framework.decorators import api_view
# import fitz
from PIL import Image
from pdf2image import convert_from_path
from rest_framework.response import Response
import pandas as pd
import math
from .qa_new import flan_main, process_image, process_pdf
import numpy as np
from pyzbar.pyzbar import decode
from .tick_mark import detect_tick_main
def combine_images_to_pdf(image_folder_path, output_pdf_path):
    pdf_canvas = canvas.Canvas(output_pdf_path, pagesize=letter)

    for filename in os.listdir(image_folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder_path, filename)
            pdf_canvas.drawImage(image_path, 0, 0, letter[0], letter[1])

            # Add a new page for each image
            pdf_canvas.showPage()

    pdf_canvas.save()

#chatbot 
@api_view(['POST'])
def predict(request):

    text = request.data.get('message')

    print(text)

    response = flan_main(text)

    message = {'answer': response}

    return Response(message)

#KIE Matching 
def calculate_horizontal_distance(question_points, answer_points):
    _,p2,_,_ = question_points
    r1,_,_,_ = answer_points

    distance = math.sqrt((r1[0] - p2[0])**2 + (r1[1] - p2[1])**2)
    

    return distance

def calculate_distance(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    box1_center_x = (x1 + x2) / 2
    box1_center_y = (y1 + y2) / 2
    box2_center_x = (x3 + x4) / 2
    box2_center_y = (y3 + y4) / 2

    distance = math.sqrt((box1_center_x - box2_center_x)**2 + (box1_center_y - box2_center_y)**2)
    return distance
def calculate_verti_distance(question_points,answer_points):
    _,_,_,p4 = question_points
    _,r2,_,_ = answer_points

    distance = math.sqrt((r2[0] - p4[0])**2 + (r2[1] - p4[1])**2)
    return distance 
def map_answers_to_questions(questions, answers):
    qa_pairs = {}

    for question_text, question_bbox, question_points in questions:
        min_distance = float('inf')
        closest_answer = None
        print("QUES", question_text)
        print("ques_bboc",question_bbox)
        for answer_text, answer_bbox, answer_points in answers:
            if answer_bbox[0] > question_bbox[2]:
                # print("answer_box",answer_bbox)
                print("hori_ans:", answer_text)
                answer_x = answer_points[0][1]
                question_x = question_points[0][1]
                print(answer_x,question_x)
                if abs(answer_x - question_x) < 10:  # Adjust the threshold as needed
                    hori_distance = calculate_horizontal_distance(question_points, answer_points)
                    centre_dist = calculate_distance(question_bbox,answer_bbox)
                    print( "centredist",centre_dist)
                    print("DIST:", hori_distance)
                    if hori_distance < min_distance :
                        min_distance = hori_distance
                        closest_answer = answer_text
                        # print("CLOSES", closest_answer)
            elif answer_bbox[1] > question_bbox[3]:
                print("answer_box",answer_bbox)
                print("verti_ans:", answer_text)
                answer_y = answer_points[0][0]
                question_y = question_points[0][0]
                if abs(answer_y - question_y) < 50 or calculate_distance(question_bbox,answer_bbox)<200:  # Adjust the threshold as needed
                    verti_distance = calculate_distance(question_bbox, answer_bbox)
                    print("DIST:", verti_distance)
                    if verti_distance < min_distance:
                        min_distance = verti_distance
                        closest_answer = answer_text
                        # print("CLOSES", closest_answer)
            else:
                continue

        print("CLOSES", closest_answer)
        if closest_answer:
            qa_pairs[question_text] = closest_answer
        else:
            qa_pairs[question_text] = None
    
    i = 0
    matched_ans=[]
    for answer_text, answer_bbox, answer_points in answers:
        for question_text, _, _ in questions:
            if question_text in qa_pairs.keys() and answer_text in qa_pairs.values():
                matched_ans.append([answer_text,answer_bbox,answer_points])
                continue
            else:
                qa_pairs[f'no_question_{i}'] = answer_text
                i += 1

    # for answer_text, answer_bbox, answer_points in answers:
    #     if f"no_question_{i}"  in qa_pairs.keys():
    #         for i in matched_ans:

    return qa_pairs

def extract_details(input_txt_path):
    # Read the raw text file
    with open(input_txt_path, 'r', encoding="utf-8") as file:
        file_content = file.read()

    # Find the index of the opening brace '{'
    start_index = file_content.find('{')

    # Extract the JSON data
    json_data = file_content[start_index:]

    # Parse the JSON data
    data = json.loads(json_data)

    title = []
    other_text = []
    que=[]
    ans=[]
    if 'ocr_info' in data:
            for item in data['ocr_info']:
                if 'transcription' in item and 'pred' in item:
                    text = item['transcription']
                    pred = item['pred']
                    bbox = item.get('bbox')
                    points = item.get('points')
                    if pred == 'HEADER':
                        title.append(text)

                    elif pred == 'O':
                        other_text.append(text)
                    
                    elif pred == 'QUESTION':
                        que.append((text, bbox,points))
                        
                    elif pred== 'ANSWER':
                        ans.append((text,bbox,points))

    qa_pairs = map_answers_to_questions(que, ans)
        

    return other_text, qa_pairs, title


#pages
def home(request):
    return render(request, 'Home.html')

def training(request):
    return render(request,'training.html')

def dashboard(request):
    return render(request,'dashboard.html')


def image_crop(result_dict,image_path,increase_width=30,increase_height=30):
    bounding_boxes = result_dict['boxes'].tolist()
    x=0
    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        x1 -= increase_width
        y1 -= increase_height
        x2 += increase_width
        y2 += increase_height

        image = Image.open(image_path).convert("RGB")
        cropped_image = image.crop((x1, y1, x2, y2))

        cropped_path = os.path.join(settings.BASE_DIR, 'idp_app', 'static', 'cropped_images', f'cropped_{x}_image.jpg')
        x += 1
        cropped_image.save(cropped_path)

def remove_files(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
def loop_folder_remove():
    static_cropped_image = os.path.join(settings.BASE_DIR, 'idp_app', 'static', 'cropped_images')
    static_csv_folder = os.path.join(settings.BASE_DIR, 'idp_app', 'static', 'csv_folder')
    static_in_images = os.path.join(settings.BASE_DIR, 'idp_app', 'static', 'in_images')
    static_in_pdf = os.path.join(settings.BASE_DIR, 'idp_app', 'static', 'in_pdf')
    static_in_pdf_images = os.path.join(settings.BASE_DIR, 'idp_app', 'static', 'in_pdf_images')
    static_out_pdf_images = os.path.join(settings.BASE_DIR, 'idp_app', 'static', 'out_pdf_images')
    static_output_images = os.path.join(settings.BASE_DIR, 'idp_app', 'static', 'output_images')
    static_output_tick = os.path.join(settings.BASE_DIR, 'idp_app', 'static', 'output_tick')
    folder_list = [static_output_tick,static_cropped_image, static_csv_folder, static_in_images, static_in_pdf, static_in_pdf_images, static_out_pdf_images, static_output_images]
    
    for folder in folder_list:
        remove_files(folder)
        print(f'Removed files in {folder}')


def process_documents(request):
    if request.method == 'POST':
        print('got file')
        # Handle file upload and call the extract_text function
        uploaded_file = request.FILES['file']
        loop_folder_remove()
        
        if uploaded_file.content_type.startswith('image'):
            
            extracted_text_list = []
            uploaded_image_path = os.path.join(settings.BASE_DIR, 'idp_app', 'static', 'in_images','uploaded_image.jpg' )
            with open(uploaded_image_path, 'wb') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            #process image for QA model
            process_image()
            #detect_table using microsoft tabel detector
            table_detect_dict = table_detect(uploaded_image_path)
            print("TABLEEEEE_DICT:::",table_detect_dict)
            cropped_image_path = os.path.join(settings.BASE_DIR, 'idp_app', 'static','cropped_images')
            image_crop(table_detect_dict,uploaded_image_path)
            #extract table from paddleocr
            folder_path = os.path.join(settings.BASE_DIR, 'idp_app', 'static', 'csv_folder')
            # paddle_table_extract(cropped_image_path)
            img_2_table_extract(cropped_image_path)
            #kie using paddle
            main_inference(uploaded_image_path)
            #detect tick mark
            # tick_df = detect_tick_main(uploaded_image_path)
            uploaded_text_path=os.path.join(settings.BASE_DIR, 'idp_app', 'PaddleOCR-release-2.7','output','ser','xfund_zh','res','infer_results.txt')

            extracted_text,extracted_pairs,extracted_title=extract_details(uploaded_text_path)
                  
            # List to store all the DataFrames
            dataframes = []

            # Iterate over the files in the folder
            for filename in os.listdir(folder_path):
                if filename.endswith('.csv'):  # Check if the file is a CSV file
                    csv_path = os.path.join(folder_path, filename)  # Get the complete path to the CSV file
                    
                    df = pd.read_csv(csv_path, header=0)
                    df = df.iloc[:, 1:]  # Exclude the first column
                    df.columns = df.iloc[0]  # Set the column headers from the first row
                    df = df[1:]  # Exclude the first row
                    df.fillna(value='', inplace=True)
                    for column in df.columns:
                        for value in df[column]:
                            # Remove matching keys or values from extracted_pairs
                            extracted_pairs = {k: v for k, v in extracted_pairs.items() if v != value and k != value}
                
                    # Append the DataFrame to the list
                    dataframes.append(df)

            # Check if any CSV files were found
            if dataframes:
                # Process each DataFrame individually
                extracted_tables = []
                for df in dataframes:
                    # Convert the DataFrame to HTML
                    table_html = df.to_html(header=True)

                    # Append the HTML table to the list
                    extracted_tables.append(table_html)
            else:
                extracted_tables = ["No CSV files found in the folder."]


            extracted_text_list.append({
                    'page': 1,
                    'text': extracted_text,
                    'pairs': extracted_pairs,
                    'title': extracted_title
                })
            # move image to static
            image_path= os.path.join(settings.BASE_DIR, 'idp_app', 'PaddleOCR-release-2.7','output','ser','xfund_zh','res','uploaded_image_ser.jpg')
            source_path = image_path
            destination_path = os.path.join(settings.BASE_DIR,'idp_app', 'static', 'output_images')
            # Move the file
            shutil.move(source_path, destination_path)
            image_path_dest =os.path.join(settings.BASE_DIR,'idp_app', 'static', 'output_images', 'uploaded_image_ser.jpg')
            final_extracted_title_list =[item['title'] for item in extracted_text_list]
            final_extracted_text_list = [item['text'] for item in extracted_text_list]
            code_decode = get_qr(uploaded_image_path)
            print("*********INFO[codees]***********************",code_decode)
            #tickmark
            # tick_image_folder = os.path.join(settings.BASE_DIR, 'idp_app', 'static', 'output_tick')
            # tick_image_files = os.listdir(tick_image_folder)
            # if tick_image_files:
            #     tick_image_path = os.path.join(tick_image_folder, tick_image_files[0])  # Assuming the first file in the folder
            # tick_final_df = tick_df.to_html(header=True)
            context = {
            'extracted_title': final_extracted_title_list[0],
            'extracted_text': final_extracted_text_list[0] ,
            'extracted_pairs': [item['pairs'].items() for item in extracted_text_list],
            'image_path': image_path_dest,
            'extracted_table': extracted_tables,
            'extracted_codes': code_decode
            }
        #PDF processing
        elif uploaded_file.content_type == 'application/pdf':
            uploaded_pdf_path = os.path.join(settings.BASE_DIR, 'idp_app', 'static', 'in_pdf', 'uploaded_pdf.pdf')
            with open(uploaded_pdf_path, 'wb') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            #process pdf for QA model
            process_pdf()
            images_folder = os.path.join(settings.BASE_DIR, 'idp_app', 'static', 'in_pdf_images')
            if os.path.exists(images_folder):
                shutil.rmtree(images_folder)
            os.makedirs(images_folder)
          
            images = convert_from_path(uploaded_pdf_path,poppler_path=r"D:\Ashwanth\projects\IDP\poppler-0.68.0\bin")
            
            extracted_text_list = []
            for i, image in enumerate(images):
                img_new_path = os.path.join(images_folder, f'page_{i+1}.jpg')
                image.save(img_new_path)
                #extract table from paddleocr
                table_detect_dict = table_detect(img_new_path)
                cropped_image_path = os.path.join(settings.BASE_DIR, 'idp_app', 'static','cropped_images')
                image_crop(table_detect_dict,img_new_path)
                paddle_table_extract(cropped_image_path)
                #kie using paddle
                main_inference(img_new_path)
                # # Read any Excel file from the folder
                folder_path = os.path.join(settings.BASE_DIR, 'idp_app', 'static', 'csv_folder')  # Path to the folder containing CSV files
                # List to store all the DataFrames
                dataframes = []

                # Iterate over the files in the folder
                for filename in os.listdir(folder_path):
                    if filename.endswith('.csv'):  # Check if the file is a CSV file
                        csv_path = os.path.join(folder_path, filename)  # Get the complete path to the CSV file

                        # Read the CSV file as a DataFrame
                        df = pd.read_csv(csv_path)
                        df = df.rename(columns={'Unnamed: 0': 'S.No.'})
                        # Append the DataFrame to the list
                        dataframes.append(df)

                # Check if any CSV files were found
                if dataframes:
                    # Process each DataFrame individually
                    extracted_tables = []
                    for df in dataframes:
                        # Convert the DataFrame to HTML
                        table_html = df.to_html(index=False)
                        
                        # Append the HTML table to the list
                        extracted_tables.append(table_html)
                else:
                    extracted_tables = ["No CSV files found in the folder."]

                uploaded_text_path=os.path.join(settings.BASE_DIR, 'idp_app', 'PaddleOCR-release-2.7','output','ser','xfund_zh','res','infer_results.txt')
                
                extracted_text,extracted_pairs,extracted_title=extract_details(uploaded_text_path)
                extracted_text_list.append({
                    'page': i+1,
                    'text': extracted_text,
                    'pairs': extracted_pairs,
                    'title': extracted_title,

                })

                # move image to static
                image_path= os.path.join(settings.BASE_DIR, 'idp_app', 'PaddleOCR-release-2.7','output','ser','xfund_zh','res',f'page_{i+1}_ser.jpg')
                source_path = image_path
                destination_path = os.path.join(settings.BASE_DIR,'idp_app', 'static', 'out_pdf_images',f'page_{i+1}_ser.jpg')
                # Move the file
                shutil.move(source_path, destination_path)
                # extracted_text_json = json.dumps(extracted_text)
                dest_path_image=os.path.join(settings.BASE_DIR,'idp_app', 'static', 'out_pdf_images')
                merged_pdf_path = os.path.join(settings.BASE_DIR, 'idp_app', 'static', 'output_images', 'merged_pdf.pdf')
                combine_images_to_pdf(dest_path_image, merged_pdf_path)
            dest_path_image=os.path.join(settings.BASE_DIR,'idp_app', 'static', 'out_pdf_images')               
            final_extracted_title_list = [item['title'] for item in extracted_text_list]
            final_extracted_text_list = [item['text'] for item in extracted_text_list]
            
            context = {
            'extracted_title': final_extracted_title_list[0],
            'extracted_text': final_extracted_text_list[0],
            'extracted_pairs': [item['pairs'].items() for item in extracted_text_list],
            'image_path': merged_pdf_path,
            'extracted_table': extracted_tables,

            }

        
        # Render the table HTML using a separate template
        extracted_text_table_html = render_to_string('extracted_text_table.html', context)
        context['extracted_text_table_html'] = extracted_text_table_html
    
        return render(request, 'process.html', context)

    return render(request, 'process.html')

def master_document(request):

    if request.method == 'POST':
        files = []
        for file in request.FILES.getlist('file'):
            files.append(file)
        images = [
            Image.open(f)
            for f in files
        ]

        pdf_path = os.path.join(settings.BASE_DIR,"idp_app","static","master","results.pdf")
            
        images[0].save(
            pdf_path, "PDF" ,resolution=100.0, save_all=True, append_images=images[1:]
        )
        context = {"pdf_path":pdf_path}
        return render(request, 'dashboard.html', context)
    
    return render(request, 'dashboard.html')


# Load imgae, grayscale, Gaussian blur, Otsu's threshold
def get_qr(image_path):
    image = cv2.imread(image_path)
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Morph close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours and filter for QR code
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    decoded = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        x,y,w,h = cv2.boundingRect(approx)
        area = cv2.contourArea(c)
        ar = w / float(h)
        if len(approx) == 4 and area > 1000 and (ar > .85 and ar < 1.3):
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 3)
            ROI = original[y:y+h, x:x+w]
            decoded.append(decode(ROI)[0].data.decode())
    return decoded