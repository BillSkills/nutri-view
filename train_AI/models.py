import torch
from torch import nn
import datetime

from pathlib import Path

import os


class CNNV0(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=32,
                      kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32 * 8 * 8, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=output_shape),
            nn.Softmax()
        )

    def forward(self, X):
        x = self.block1(X)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.Classifier(x)
        return x

    def save(self, file_name=None):
        model_folder_path = 'models_weights/'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        if file_name is None:
            file_name = f"CNN_{len(self.state_dict()) / 2}_params.pth"

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class CNNV1(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=32,
                      kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.Classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32 * 8 * 8, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=output_shape),
            nn.Softmax()
        )

    def forward(self, X):
        x = self.block1(X)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.Classifier(x)
        return x

    def save(self, file_name=None):
        model_folder_path = 'models_weights/'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        if file_name is None:
            file_name = f"CNN_{len(self.state_dict()) / 2}_params.pth"

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class ResidualBlock2D(nn.Module):
    """
    Test for a custom torch layer for residual neural network
    This one is for 2d residual layer
    """

    def __init__(self, nb_channels: int, kernel_size):
        super(ResidualBlock2D, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=nb_channels, out_channels=nb_channels, kernel_size=kernel_size,
                               padding="same", stride=1)
        self.conv2 = nn.Conv2d(in_channels=nb_channels, out_channels=nb_channels, kernel_size=kernel_size,
                               padding="same", stride=1)

        self.activation_fn = nn.LeakyReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation_fn(out)
        out = self.conv2(out)

        out += x

        return self.activation_fn(out)


class EAM_residual_1(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(EAM_residual_1, self).__init__()

        self.output_shape = output_shape

        self.layer1 = ResidualBlock2D(input_shape, 3)

        self.layer2 = nn.Conv2d(input_shape, out_channels=16, kernel_size=3, stride=2, padding=1)

        self.layer3 = ResidualBlock2D(16, 3)

        self.layer4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)

        self.layer5 = ResidualBlock2D(32, 3)

        self.layer6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)

        self.layer7 = ResidualBlock2D(32, 3)

        self.layer8 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(32 * 8 * 8, 128)

        self.layer_final = nn.Linear(128, out_features=self.output_shape)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)

        x = self.flatten(x)

        x = self.linear1(x)

        out = self.layer_final(x)

        # out = nn.Softmax(out)

        return out

    def save(self, file_name=None):
        model_folder_path = 'models_weights/'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        if file_name is None:
            file_name = f"test_Res_1.pth"

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)



class EAM_residual_2(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(EAM_residual_2, self).__init__()

        self.output_shape = output_shape

        self.layer1 = ResidualBlock2D(input_shape, 3)

        self.layer2 = nn.Conv2d(input_shape, out_channels=16, kernel_size=3, stride=1, padding=1)

        self.layer3 = ResidualBlock2D(16, 3)

        self.layer4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.layers = nn.ModuleList()

        for i in range(4):
            self.layers.append(ResidualBlock2D(32, 3))
            self.layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1))
            self.layers.append(ResidualBlock2D(32, 3))
            self.layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1))

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(32 * 8 * 8, 128)

        self.layer_final = nn.Linear(128, out_features=self.output_shape)

        self.Classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32 * 8 * 8, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=output_shape),
            nn.Softmax()
        )


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        for layer in self.layers:
            x = layer(x)

        out = self.Classifier(x)

        # out = nn.Softmax(out)

        return out

    def save(self, file_name=None):
        model_folder_path = 'models_weights/'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        if file_name is None:
            file_name = f"test_Res_2.pth"

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
