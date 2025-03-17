import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import RainDataset
from utils.preprocessing import get_transform
import os
from tqdm import tqdm

# Define PReNet model
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

# Training parameters
batch_size = 8
num_epochs = 50
learning_rate = 0.001
train_rainy_dir = 'data/rainy_images'
train_clear_dir = 'data/clear_images'
model_save_path = 'models/derain_model.pth'

# Load dataset
transform = get_transform()
train_dataset = RainDataset(train_rainy_dir, train_clear_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PReNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for rainy, clear in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        rainy = rainy.to(device)
        clear = clear.to(device)

        # Forward pass
        outputs = model(rainy)
        loss = criterion(outputs, clear)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

# Save the model
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')