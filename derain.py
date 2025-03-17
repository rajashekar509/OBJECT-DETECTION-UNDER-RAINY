import torch
from utils.preprocessing import preprocess_image
from PIL import Image
import os
import numpy as np
class PReNet(torch.nn.Module):
    def __init__(self):
        super(PReNet, self).__init__()
        # Define the model architecture here
        pass

    def forward(self, x):
        # Forward pass
        return x

def derain_image(image_path, output_path, model):
    """
    Derain a single image and save the result.
    """
    image = preprocess_image(image_path)
    with torch.no_grad():
        derained_image = model(image)
    derained_image = derained_image.squeeze(0).permute(1, 2, 0).numpy()
    derained_image = (derained_image * 255).astype(np.uint8)
    Image.fromarray(derained_image).save(output_path)

if __name__ == "__main__":
    # Load deraining model
    derain_model = PReNet()
    model_path = 'models/derain_model.pth'

    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please ensure the file exists.")

    try:
        derain_model.load_state_dict(torch.load(model_path))
        derain_model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights from '{model_path}'. Error: {e}")

    # Derain all images in the rainy_images folder
    input_folder = 'data/rainy_images'
    output_folder = 'data/derained_images'
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        derain_image(input_path, output_path, derain_model)
        print(f"Derained {filename} and saved to {output_path}")