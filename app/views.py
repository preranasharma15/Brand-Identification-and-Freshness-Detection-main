from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from .models import BrandResult, FreshnessResult
import cv2
import pandas as pd
import time
import os
from django.conf import settings
from django.http import StreamingHttpResponse
import csv

# Import Brand Classification Dependencies
import cvlib as cv
from paddleocr import PaddleOCR
import google.generativeai as genai

# Import Freshness Calculator Dependencies
import numpy as np
import tensorflow as tf
import joblib
import pickle
from cvlib.object_detection import draw_bbox
from .models import BrandResult
from django.utils import timezone

def home_view(request):
    return render(request, 'home.html')

def rgb_to_grayscale(x):
    return tf.image.rgb_to_grayscale(x)

def draw_bbox_without_labels(frame, bbox):
    for box in bbox:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

def detect_objects_and_text():

    genai.configure(api_key="API_Key Here")
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Initialize PaddleOCR
    ocr = PaddleOCR(lang='en')

    # Webcam capture
    cap = cv2.VideoCapture(0)

    last_frame_time = 0

    try:
        while True:
            current_time = time.time()
            ret, frame = cap.read()

            if not ret or frame is None:
                print("Failed to capture frame. Retrying...")
                continue

            try:
                bbox, labels, confidences = cv.detect_common_objects(frame)
                output_image = draw_bbox_without_labels(frame, bbox)
            except Exception as e:
                print(f"Error during object detection: {e}")
                continue

            output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

            if current_time - last_frame_time >= 2:
                last_frame_time = current_time
                cropped_objects = []

                for box in bbox:
                    x1, y1, x2, y2 = box
                    try:
                        cropped_object = frame[y1:y2, x1:x2]
                        if cropped_object.size != 0:
                            cropped_objects.append(cropped_object)
                        else:
                            print("Warning: Cropped object is empty.")
                    except Exception as e:
                        print(f"Error while cropping object: {e}")

                for idx, obj in enumerate(cropped_objects):
                    try:
                        ocr_results = []
                        result = ocr.ocr(obj, cls=True)
                        if result and result[0]:
                            detected_text = [line[1][0] for line in result[0]]
                            ocr_results.append(detected_text)

                        if ocr_results:
                            response = model.generate_content(
                                f"Extract the product name from this and return just the name and nothing else. Otherwise return a null string: {ocr_results}"
                            )
                            brand = response.text.strip()
                            if brand:
                                # Save to the database
                                BrandResult.objects.create(
                                    timestamp=timezone.now(),
                                    brand=brand
                                    )
                                print(f"Brand saved: {brand}")
                                cv2.putText(frame, brand, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    except Exception as e:
                        print(f"Error during OCR processing: {e}")

            # Encode frame for streaming
            _, jpeg = cv2.imencode('.jpg', output_image)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")
    finally:
        cap.release()
        cv2.destroyAllWindows()


# Load models for freshness calculator
def load_freshness_models():
    model_path = os.path.join(settings.BASE_DIR, 'app', 'models', 'model2.h5')
    regression_model_path = os.path.join(settings.BASE_DIR, 'app', 'models', 'regression_model.pkl')
    classification_model = tf.keras.models.load_model(model_path, custom_objects={'rgb_to_grayscale': rgb_to_grayscale})
    with open(regression_model_path, 'rb') as f:
        regression_model = pickle.load(f)
    return classification_model, regression_model

# Load models
try:
    classification_model, regression_model = load_freshness_models()
except Exception as e:
    print(f"Error loading models: {e}")
    classification_model, label_encoder, regression_model = None, None, None

def calculate_color_score(image):
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _ , saturation ,_ = cv2.split(hsv_image)
    mean_saturation = np.mean(saturation)
    color_score = (mean_saturation / 255) * 100

    return color_score

def calculate_shape_score(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0
    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    x, y, w, h = cv2.boundingRect(largest_contour)
    bounding_box_area = w * h
    extent = contour_area / bounding_box_area if bounding_box_area > 0 else 0
    shape_score = extent * 100

    return shape_score

def calculate_wrinkle_score(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 100)
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    wrinkle_score = (1 - edge_density) * 100

    return wrinkle_score

def calculate_marks_score(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    total_pixels = image.shape[0] * image.shape[1]
    dark_pixels = np.sum(thresh == 255)
    dark_ratio = dark_pixels / total_pixels
    mark_score = (1 - dark_ratio) * 100

    return mark_score

def extract_features(image):

    image = cv2.resize(image, (224, 244))

    color_score = calculate_color_score(image)
    shape_score = calculate_shape_score(image)
    wrinkle_score = calculate_wrinkle_score(image)
    mark_score = calculate_marks_score(image)

    return {
        "color_score": color_score,
        "shape_score": shape_score,
        "wrinkle_score": wrinkle_score,
        "mark_score": mark_score
    }

def freshness_calc(image):
    classes = ['Apple', 'Banana', 'Bittergourd', 'Capsicum', 'Cucumber', 'Okra', 'Oranges', 'Peaches', 'Pomergranate', 'Potato', 'Strawberry', 'Tomato']

    image_resized = cv2.resize(image, (128, 128))

    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    image_array = image_rgb.astype('int')

    # Add batch dimension to the image array
    image_array = np.expand_dims(image_array, axis=0)

    predictions = classification_model.predict(image_array)
    predicted_class = classes[np.argmax(predictions)]

    print(predicted_class)
    features = extract_features(image)

    features = list(features.values())

    freshness_score = regression_model.predict([features]) 
    return freshness_score, predicted_class

def detect_objects_and_calculate():
    cap = cv2.VideoCapture(0)

    last_time = 0

    try:
        while True:
            current_time = time.time()

            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to capture frame. Retrying...")
                continue

            try:
                bbox, labels, confidences = cv.detect_common_objects(frame)
            except Exception as e:
                print(f"Error during object detection: {e}")
                continue

            output_image = draw_bbox_without_labels(frame, bbox)

            if current_time - last_time >= 2:
                last_time = current_time
                cropped_objects = []
                for box in bbox:
                    x1, y1, x2, y2 = box
                    try:
                        cropped_object = frame[y1:y2, x1:x2]
                        if cropped_object.size != 0:
                            cropped_objects.append(cropped_object)
                        else:
                            print("Warning: Cropped object is empty.")
                    except Exception as e:
                        print(f"Error while cropping object: {e}")
                
                for idx, obj in enumerate(cropped_objects):
                        try:
                            freshness_score, item_name = freshness_calc(obj)
                            if freshness_score:
                                shelf_life = round(freshness_score[0], 2)

                                # Save to the database
                                FreshnessResult.objects.create(
                                    timestamp=timezone.now(),
                                    item_name=item_name,
                                    freshness_score=float(freshness_score),
                                    shelf_life=shelf_life
                                )
                                print(f"Freshness result saved: {item_name}, {freshness_score}, {shelf_life}")
                                cv2.putText(frame, item_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            else:
                                print(f"NA for {labels[idx]}")
                        except Exception as e:
                            print(f"Error during Freshness Score Calculation: {e}")

            # Encode frame for streaming
            _, jpeg = cv2.imencode('.jpg', output_image)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")
    finally:
        cap.release()
        cv2.destroyAllWindows()    
  

# Brand Classification View
def brand_classification_view(request):
    if request.method == 'POST':
        # Implement brand classification logic here
        result = "Brand classification result here"
        return JsonResponse({'result': result})
    return render(request, 'brand.html')

# Freshness Calculator View
def freshness_calculator_view(request):
    if request.method == 'POST':
        # Implement freshness calculator logic here
        result = "Freshness calculation result here"
        return JsonResponse({'result': result})
    return render(request, 'freshness.html')


def brand_classification_feed(request):
    return StreamingHttpResponse(detect_objects_and_text(), content_type='multipart/x-mixed-replace; boundary=frame')

def freshness_calculator_feed(request):
    return StreamingHttpResponse(detect_objects_and_calculate(), content_type='multipart/x-mixed-replace; boundary=frame')


# Export to Excel
def export_brand_results(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="brand_results.csv"'

    writer = csv.writer(response)
    
    writer.writerow(['ID', 'Timestamp', 'Brand'])
    
    brand_results = BrandResult.objects.all()
    for result in brand_results:
        writer.writerow([result.id, result.timestamp, result.brand])
    
    return response

def export_freshness_results(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="freshness_results.csv"'

    writer = csv.writer(response)
    
    writer.writerow(['ID', 'Timestamp', 'Produce', 'Freshness Score', 'Expected Life Span'])
    
    freshness_results = FreshnessResult.objects.all()
    for result in freshness_results:
        writer.writerow([result.id, result.timestamp, result.produce, result.freshness, result.expected_life_span])
    
    return response


