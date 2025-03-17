import torch
from utils.dataset import RainDataset  # Ensure this module exists
from utils.preprocessing import get_transform  # Ensure this module exists
from PIL import Image
import os

# Define the PReNet model
class PReNet(torch.nn.Module):
    def __init__(self):
        super(PReNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Load the trained model
model = PReNet()
model.load_state_dict(torch.load('models/derain_model.pth', map_location=torch.device('cpu')))  # Load model weights
model.eval()  # Set the model to evaluation mode

# Test on a sample image
test_rainy_dir = 'data/rainy_images'  # Directory containing rainy images
test_clear_dir = 'data/clear_images'  # Directory containing clear images (optional)
transform = get_transform()  # Get the preprocessing transform
test_dataset = RainDataset(test_rainy_dir, test_clear_dir, transform=transform)  # Create dataset

# Create output directory
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Process each image in the dataset
for i in range(len(test_dataset)):
    rainy, clear = test_dataset[i]  # Get rainy and clear images
    output = model(rainy.unsqueeze(0))  # Add batch dimension and pass through the model
    output = output.squeeze(0).permute(1, 2, 0).detach().numpy()  # Convert to NumPy array
    output = (output * 255).astype('uint8')  # Scale to [0, 255]
    output_image = Image.fromarray(output)  # Convert to PIL image
    output_image.save(os.path.join(output_dir, f'output_{i}.png'))  # Save the output image
    print(f'Saved output_{i}.png')