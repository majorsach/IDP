import math
import json

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

# def calculate_verti_distance(question_points,answer_points):
#     _,_,_,p4 = question_points
#     _,r2,_,_ = answer_points

#     distance = math.sqrt((r2[0] - p4[0])**2 + (r2[1] - p4[1])**2)
#     return distance 

def map_answers_to_questions(questions, answers):
    qa_pairs = {}

    for question_text, question_bbox, question_points in questions:
        min_distance = float('inf')
        closest_answer = None
        # print("QUES", question_text)
        # print("ques_bboc",question_bbox)
        for answer_text, answer_bbox, answer_points in answers:
            if answer_bbox[0] > question_bbox[2]:
                # print("answer_box",answer_bbox)
                # print("hori_ans:", answer_text)
                answer_x = answer_points[0][1]
                question_x = question_points[0][1]
                # print(answer_x,question_x)
                if abs(answer_x - question_x) < 50:  # Adjust the threshold as needed
                    hori_distance = calculate_horizontal_distance(question_points, answer_points)
                    # print("DIST:", hori_distance)
                    if hori_distance < min_distance:
                        min_distance = hori_distance
                        closest_answer = answer_text
                        # print("CLOSES", closest_answer)
            elif answer_bbox[1] > question_bbox[3]:
                # print("answer_box",answer_bbox)
                # print("verti_ans:", answer_text)
                answer_y = answer_points[0][0]
                question_y = question_points[0][0]
                if abs(answer_y - question_y) < 50:  # Adjust the threshold as needed
                    # verti_distance = calculate_verti_distance(question_points,answer_points) ##change and check if needed 
                    verti_distance = calculate_distance(question_bbox, answer_bbox)
                    # print("DIST:", verti_distance)
                    if verti_distance < min_distance:
                        min_distance = verti_distance
                        closest_answer = answer_text
                        # print("CLOSES", closest_answer)
            else:
                continue

        # print("CLOSES", closest_answer)
        if closest_answer:
            qa_pairs[question_text] = closest_answer
        else:
            qa_pairs[question_text] = None

    i = 0
    for answer_text, answer_bbox, _ in answers:
        for question_text, _, _ in questions:
            if question_text in qa_pairs.keys() and answer_text in qa_pairs.values():
                continue
            else:
                qa_pairs[f'no_key_{i}'] = answer_text
                i += 1
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


if __name__ =='__main__':
    text,pairs,title=extract_details(r'path')