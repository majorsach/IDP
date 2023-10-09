import cv2
import numpy as np
import os
import pandas as pd
from django.conf import settings
import pytesseract
import re

MAX_BOX_AREA = 200  # Maximum allowable area for small boxes
ASPECT_RATIO_THRESHOLD = 1.1  # Maximum allowable aspect ratio for valid tick boxes
WIDTH_HEIGHT_THRESHOLD = (
    0.9  # Minimum allowable width-height ratio for valid tick boxes
)
MIN_BOX_AREA = 50  # Minimum allowable area for valid tick boxes


def detect_box(image, line_min_width=12):
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th1, img_bin = cv2.threshold(gray_scale, 170, 250, cv2.THRESH_BINARY)
    kernal6h = np.ones((1, line_min_width), np.uint8)
    kernal6v = np.ones((line_min_width, 1), np.uint8)
    img_bin_h = cv2.morphologyEx(~img_bin, cv2.MORPH_OPEN, kernal6h)
    img_bin_v = cv2.morphologyEx(~img_bin, cv2.MORPH_OPEN, kernal6v)
    img_bin_final = img_bin_h | img_bin_v
    final_kernel = np.ones((3, 3), np.uint8)
    img_bin_final = cv2.dilate(img_bin_final, final_kernel, iterations=1)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(
        ~img_bin_final, connectivity=8, ltype=cv2.CV_32S
    )
    return stats, labels


def detect_tick(image, checkbox_region, threshold_ratio=0.1):
    checkbox_crop = image[
        checkbox_region[1] : checkbox_region[1] + checkbox_region[3],
        checkbox_region[0] : checkbox_region[0] + checkbox_region[2],
    ]
    gray = cv2.cvtColor(checkbox_crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    checkbox_area = checkbox_crop.size
    threshold_area = threshold_ratio * checkbox_area

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > threshold_area:
            return True  # Checkbox has a tick mark

    return False  # Checkbox does not have a tick mark


def clean_text(text):
    # Remove unwanted characters and newlines
    cleaned_text = re.sub(r"[\n\r\t]", " ", text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)

    # Remove "_ |"
    cleaned_text = re.sub("[|_}{]", "", cleaned_text)

    return cleaned_text.strip()


def detect_tick_main(image_path):
    image = cv2.imread(image_path)
    stats, labels = detect_box(image)
    results = []
    checkbox_regions = []

    for x, y, w, h, area in stats[2:]:
        if area > MAX_BOX_AREA or area < MIN_BOX_AREA:
            continue

        aspect_ratio = float(w) / h
        if (
            aspect_ratio > ASPECT_RATIO_THRESHOLD
            or aspect_ratio < 1 / ASPECT_RATIO_THRESHOLD
        ):
            continue

        width_height_ratio = float(w) / w
        if width_height_ratio < WIDTH_HEIGHT_THRESHOLD:
            continue

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

        checkbox_region = (x, y, w, h)
        checkbox_regions.append(checkbox_region)

        has_tick = detect_tick(image, checkbox_region)
        if has_tick:
            tick_status = "Ticked"
            cv2.putText(
                image, "Tick", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
        else:
            tick_status = "Not Ticked"
            cv2.putText(
                image,
                "No Tick",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        results.append(
            {"Checkbox Region": checkbox_region, "Checkbox Status": tick_status}
        )

    for i in range(len(results)):
        checkbox_region = results[i]["Checkbox Region"]
        text_x_start = (
            checkbox_region[0] + checkbox_region[2] + 5
        )  # Adjust this value to set the desired distance between checkbox and text
        text_x_end = image.shape[
            1
        ]  # Set the initial end of the text region to the right edge of the image

        if i < len(results) - 1:
            next_checkbox_region = results[i + 1]["Checkbox Region"]
            text_x_end = min(text_x_end, next_checkbox_region[0])

        if text_x_start >= text_x_end:
            text_x_end = text_x_start + 70

        text_y = checkbox_region[1] + int(
            checkbox_region[3] / 2
        )  # Adjust this value to align the text vertically within the checkbox

        cropped_image = image[
            text_y - 10 : text_y + 10, text_x_start:text_x_end
        ]  # Adjust the vertical crop range as needed
        if cropped_image.size == 0:
            continue  # Skip OCR if the cropped image is empty

        try:
            text = pytesseract.image_to_string(cropped_image, config="--psm 7")
            cleaned_text = clean_text(text)
            results[i]["Text"] = cleaned_text
        except pytesseract.TesseractError:
            continue  # Skip OCR if an error occurs during OCR

    out_folder = os.path.join(settings.BASE_DIR, "idp_app", "static", "output_tick")
    filename = os.path.basename(image_path)
    cv2.imwrite(os.path.join(out_folder, f"out_{filename}"), image)

    df = pd.DataFrame(results)
    df = df.drop(["Checkbox Region"], axis=1)
    return df
