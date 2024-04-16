from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import torch

from torch.utils.data import DataLoader
import torch.nn as nn
import os
import torch.optim as optim
import torch.nn.functional as F


class PreTrainedImageModel(torch.nn.Module):
    def __init__(self, total_labels):
        super(PreTrainedImageModel, self).__init__()

        self.pretrain_model = models.resnet50(pretrained=True)

        for _ , param in self.pretrain_model.named_parameters():
            param.require_grad = False

        num_features = self.pretrain_model.fc.in_features
        self.pretrain_model.fc = nn.Sequential(
                nn.Linear(num_features, 500),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(500, total_labels),
                )

    def forward(self, x):
        return self.pretrain_model(x)


    def get_optimize_parameters(self):
        return self.pretrain_model.fc.parameters()





class PreTrainedImageClassifier:
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
        self.model = PreTrainedImageModel(total_labels=total_labels).to(self.device)



    def fit(self, epochs,  batch_size=10, lr=0.001):
        train_image_loader =  DataLoader(
                self.train_folder,
                batch_size=batch_size,
                shuffle=True
                )

        val_image_loader =  DataLoader(
                self.val_folder,
                batch_size=batch_size,
                shuffle=True
                )

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.get_optimize_parameters(), lr=lr, momentum=0.9)

        
        for epoch in range(epochs):
            print(f"Epoch {epoch}")

            self.model.train()

            current_losses = []
            total_corrects = 0
            total_data = 0

            epoch_train_acc = 0.0

            epoch_val_losses = 0.0
            epoch_val_acc = 0.0

            for item in train_image_loader:
                images, labels = item
                images, labels = images.to(self.device), labels.to(self.device)

                preds = self.model(images)
                loss = loss_fn(preds, labels)
                loss.backward()
                optimizer.step()

                current_losses.append(loss.item())
                correct = torch.eq(torch.max(F.softmax(preds, dim=1), dim=1)[1],labels).view(-1)
                correct = torch.sum(correct).item()

                total_corrects += correct
                total_data += images.shape[0]

                optimizer.zero_grad()

            epoch_train_acc = total_corrects / total_data
            epoch_train_loss = sum(current_losses) / len(current_losses)

            print("Training => Acc {:.2} Loss {:.2}".format(epoch_train_acc, epoch_train_loss))






