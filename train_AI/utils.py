import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# Détermine si le gpu va être utilisé
device = "cuda" if torch.cuda.is_available() else "cpu"


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    """
    Boucle d'entrainement de base pour un modèle pytorch
    :param model: Le modèle à entrainer
    :param data_loader: le dataset
    :param loss_fn: la fonction de loss
    :param optimizer: l'optimizer
    :param accuracy_fn: la fonction qui calcule l'accuracy
    :param device: cpu ou gpu en fonction de la disponibilité
    :return: None
    """

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        model.train()

        y_logits = model(X)

        loss = loss_fn(y_logits, y)

        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_logits.argmax(dim=1))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

    return train_acc


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    """
    Boucle de test de base pour un modèle pytorch
    :param model: Le modèle à tester
    :param data_loader: Le dataset
    :param loss_fn: Fonction de loss
    :param accuracy_fn: La fonction qui calcule l'accuracy
    :param device: cpu ou gpu en fonction de la disponibilité
    :return: None
    """

    test_loss, test_acc = 0, 0
    model.eval()  # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                                    y_pred=test_pred.argmax(dim=1)  # Go from logits -> pred labels
                                    )

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

        return test_acc


def accuracy_fn(y_true, y_pred):
    """
    Fonction d'accuracy
    :param y_true: L'objectif
    :param y_pred: Ce qui a été prédis par le modèle
    :return: l'accuracy
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = correct / len(y_true) * 100
    return acc


def custom_train_step(model: torch.nn.Module,
                      data_loader: torch.utils.data.DataLoader,
                      loss_fn: torch.nn.Module,
                      optimizer: torch.optim.Optimizer,
                      accuracy_fn,
                      device: torch.device = device):
    """
    Boucle d'entrainement de base pour un modèle pytorch
    :param model: Le modèle à entrainer
    :param data_loader: le dataset
    :param loss_fn: la fonction de loss
    :param optimizer: l'optimizer
    :param accuracy_fn: la fonction qui calcule l'accuracy
    :param device: cpu ou gpu en fonction de la disponibilité
    :return: None
    """

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(data_loader):
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(dim=1)

        X, y = X.to(device), y.to(device)

        model.train()

        y_logits = model(X)

        loss = loss_fn(y_logits, y)

        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=torch.round(y_logits))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")


def custom_test_step(model: torch.nn.Module,
                     data_loader: torch.utils.data.DataLoader,
                     loss_fn: torch.nn.Module,
                     accuracy_fn,
                     device: torch.device = device):
    """
    Boucle de test de base pour un modèle pytorch
    :param model: Le modèle à tester
    :param data_loader: Le dataset
    :param loss_fn: Fonction de loss
    :param accuracy_fn: La fonction qui calcule l'accuracy
    :param device: cpu ou gpu en fonction de la disponibilité
    :return: None
    """

    test_loss, test_acc = 0, 0
    model.eval()  # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        for X, y in data_loader:
            y = torch.tensor(y, dtype=torch.float32).unsqueeze(dim=1)
            # Send data to GPU
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                                    y_pred=torch.round(test_pred))

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")


def save(weights, file_name=None):
    model_folder_path = 'models_weights/'
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    if file_name is None:
        file_name = f"CNN_{len(weights) / 2}_params.pth"

    file_name = os.path.join(model_folder_path, file_name)
    torch.save(weights, file_name)
