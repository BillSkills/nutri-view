import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

import numpy as np
import matplotlib.pyplot as plt

import utils
import models

# ------------------------------------------------ Getting the Datasets ------------------------------------------------


dataset_dir = Path("D:/dataset/MAIS_hackaton/food_images_251/pytorch_dataset_256")

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((50, 50)),
    transforms.Grayscale()
])

train_data = datasets.ImageFolder(root=dataset_dir / "train",
                                  transform=transforms.ToTensor(),
                                  target_transform=None)

test_data = datasets.ImageFolder(root=dataset_dir / "test",
                                 transform=transforms.ToTensor(),
                                 target_transform=None)

# --------------------------------------------------- Hyperparameter ---------------------------------------------------

BATCH_SIZE = 64
epochs = 20
input_shape = 3  # 28*28
output_shape = len(train_data.classes)
lr = 0.010  # learning rate the rate at witch the weights are modified
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------- Setting the dataloader -----------------------------------------------

train_dataloader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    # num_workers=1,  # os.cpu_count(),
    shuffle=True
)

test_dataloader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    # num_workers=1,  # os.cpu_count(),
    shuffle=False
)


# ------------------------------------------------- Building the model -------------------------------------------------

# model = models.EAM_residual_2(input_shape, output_shape)
model = torchvision.models.resnet18(pretrained=True)

model.to(device)

# ------------------------------------------------- Loss and Optimizer -------------------------------------------------

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(lr=lr, params=model.parameters())

# --------------------------------------------------- Training loop ----------------------------------------------------

record = 0

for epoch in range(epochs):
    print(epoch)
    utils.train_step(
        model, train_dataloader, loss_fn, optimizer,
        utils.accuracy_fn,
        device=device
    )

    accuracy = utils.test_step(
        model, test_dataloader, loss_fn,
        utils.accuracy_fn,
        device=device
    )

    if accuracy > record:
        utils.save(model.state_dict(), "test_resnet_34.pth")
        #model.save()
        record = accuracy

    # Fait en sorte que le learning rate descende au fil du temps pour essayer d'améliorer la précision
    lr *= 0.95
    optimizer = torch.optim.SGD(lr=lr, params=model.parameters())

# enregistre le modèle, le nom du fichier peut être spécifié
utils.save(model.state_dict(), "test_resnet_34.pth")
