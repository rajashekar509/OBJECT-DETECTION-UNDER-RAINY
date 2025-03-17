from torchvision import transforms
from PIL import Image

def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

def preprocess_image(image):
    """
    Preprocess an image for model input.
    """
    if isinstance(image, str):  # If input is a file path
        image = Image.open(image).convert('RGB')
    transform = get_transform()
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image