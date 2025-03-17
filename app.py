from flask import Flask, render_template, request, redirect, url_for
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load a pre-trained object detection model (Faster R-CNN)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()

# Define the deraining model
class PReNet(nn.Module):
    def __init__(self):
        super(PReNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Load trained derain model
def load_derain_model():
    model = PReNet()
    model.load_state_dict(torch.load('models/derain_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Function to preprocess image before deraining
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize for uniform input
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Function to detect objects and draw bounding boxes
from PIL import ImageDraw

def detect_and_draw_boxes(image_path, output_path):
    image = Image.open(image_path).convert("RGB")  # Define image here
    image_tensor = preprocess_image(image_path)

    # Run object detection
    with torch.no_grad():
        predictions = model(image_tensor)

    # Extract predictions
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()

    print("Number of objects detected:", len(scores))
    print("Boxes:", boxes)
    print("Scores:", scores)
    print("Labels:", labels)

    # Ensure image is defined before drawing
    draw = ImageDraw.Draw(image)

    # If no objects detected, draw a test box
    if len(scores) == 0:
        print("No objects detected! Drawing a test box.")
        draw.rectangle([50, 50, 200, 200], outline="blue", width=3)
    else:
        for i in range(len(scores)):
            if scores[i] > 0.3:
                box = [int(coord) for coord in boxes[i]]
                draw.rectangle(box, outline="red", width=3)

    image.save(output_path)



    

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Derain image
        derained_filename = "derained_" + file.filename
        derained_path = os.path.join(UPLOAD_FOLDER, derained_filename)

        derain_model = load_derain_model()
        derained_image = preprocess_image(file_path)

        with torch.no_grad():
            output = derain_model(derained_image)

        output = output.squeeze(0).permute(1, 2, 0).numpy()
        output = (output * 255).astype(np.uint8)
        Image.fromarray(output).save(derained_path)

        # Detect objects and draw bounding boxes
        detected_filename = "detected_" + file.filename
        detected_path = os.path.join(UPLOAD_FOLDER, detected_filename)
        detect_and_draw_boxes(derained_path, detected_path)

        return render_template("index.html", original_image=file.filename, derained_image=derained_filename, detected_image=detected_filename)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
