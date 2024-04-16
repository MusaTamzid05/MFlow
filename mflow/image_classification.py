from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch

from torch.utils.data import DataLoader
import os


class PretrainImageClassifier:
    def __init__(
            self,
            train_data_path,
            val_data_path,
            ):

        transform =  transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.485,0.485], std=[0.229, 0.224, 0.225])
            ])

        self.train_folder = ImageFolder(train_data_path, transform=transform)
        self.val_folder = ImageFolder(val_data_path, transform=transform)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        total_labels = len(os.listdir(train_data_path))
        print(f"total labels {total_labels}")


    def fit(self, batch_size=10):
        train_image_loader =  DataLoader(
                self.train_folder,
                batch_size=batch_size,
                shuffle=True
                )


        for row in  train_image_loader:
            images , labels = row
            print(f"{images.shape}, {labels}")






