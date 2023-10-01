import os
import numpy as np
from PIL import Image

import torch
import torchvision


def test_loading_resnet(path):
    model = torchvision.models.resnet18()
    model.load_state_dict(torch.load(path))

    return model

def prep_img(img, size=256):
    img = img.resize((size, size))
    img = torch.tensor(np.array(img), dtype=torch.float32)
    img = img.permute(2, 1, 0)
    img = img.unsqueeze(dim=0)
    # print(img.shape)

    return img

def prediction(model, img):

    model.eval()

    with torch.inference_mode():
        pred = model.forward(img)

    return pred.numpy()[0]


if __name__ == "__main__":
    m = test_loading_resnet("models_weights/test_resnet_18_256.pth")
    image = Image.open("D:/dataset/MAIS_hackaton/food_images_251/all_images_256/test_000111.jpg")

    img_prep = prep_img(image)

    # """
    pred = prediction(m, img_prep)

    print(pred)
    print(type(pred))
    # """
