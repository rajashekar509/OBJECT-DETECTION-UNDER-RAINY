import torch
from utils.preprocessing import preprocess_image
from utils.visualization import visualize_results
import os

def detect_objects(image_path, model):
    """
    Perform object detection on an image.
    """
    image = preprocess_image(image_path)
    results = model(image)
    return results

if __name__ == "__main__":
    # Load YOLOv5 model
    object_detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # Detect objects in derained images
    input_folder = 'data/derained_images'
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        results = detect_objects(input_path, object_detection_model)
        visualize_results(preprocess_image(input_path), results, object_detection_model)