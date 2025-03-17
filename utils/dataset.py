import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class RainDataset(Dataset):
    def __init__(self, rainy_dir, clear_dir, transform=None):
        self.rainy_dir = rainy_dir
        self.clear_dir = clear_dir
        self.transform = transform
        self.rainy_images = sorted(os.listdir(rainy_dir))
        self.clear_images = sorted(os.listdir(clear_dir))

    def __len__(self):
        return len(self.rainy_images)

    def __getitem__(self, idx):
        rainy_image_path = os.path.join(self.rainy_dir, self.rainy_images[idx])
        clear_image_path = os.path.join(self.clear_dir, self.clear_images[idx])

        rainy_image = Image.open(rainy_image_path).convert('RGB')
        clear_image = Image.open(clear_image_path).convert('RGB')

        if self.transform:
            rainy_image = self.transform(rainy_image)
            clear_image = self.transform(clear_image)

        return rainy_image, clear_image