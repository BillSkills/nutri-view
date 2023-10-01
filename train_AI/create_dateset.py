from pathlib import Path
from PIL import Image
import random
import os
import shutil

import torch
import numpy as np
import pandas as pd


def resize_and_move_img(path, path_save):
    SIZE = 256

    img_o = Image.open(path)
    img_dim = img_o.size
    ratio = 1

    if img_dim[0] > img_dim[1]:
        ratio = img_dim[0]/img_dim[1]
    else:
        ratio = img_dim[1]/img_dim[0]

    if img_dim[0] >= SIZE and img_dim[1] >= SIZE and ratio < 1.75:
        img_o = img_o.resize((SIZE, SIZE))
        # print(img_o.size)
        img_o.save(path_save / path.parts[-1])


def select_correct_images(paths: list) -> list:
    ls_all_paths = []

    for path in paths:
        ls_all_paths += list(path.glob("*.jpg"))

    print(len(ls_all_paths))
    return ls_all_paths


def create_train_test(data, fraction):
    """
    Function that take an iterable and return a training and a testing set in function of the fraction given in argument
    :param data: the complete dataset
    :param fraction: the fraction of the dataset that represent the test data
    :return: training and testing set
    """

    random.shuffle(data)

    length_test = int(len(data) * fraction)

    test = []

    for i in range(length_test):
        a = random.randint(0, len(data) - 1)
        test.append(data[a])
        del data[a]

    return data, test


def create_dataset(tr_paths, te_paths, annots, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i in tr_paths:
        if not "test" in i.parts[-1]:
            dir = save_dir / "train" / str(annots.loc[i.parts[-1]].number)

            if not os.path.exists(dir):
                dir.mkdir(parents=True, exist_ok=True)

            dir = dir / i.parts[-1]

            shutil.copy(i, dir)

    for i in te_paths:
        if not "test" in i.parts[-1]:
            dir = save_dir / "test" / str(annots.loc[i.parts[-1]].number)

            if not os.path.exists(dir):
                dir.mkdir(parents=True, exist_ok=True)

            dir = dir / i.parts[-1]

            shutil.copy(i, dir)


if __name__ == "__main__":
    p_save = Path("D:/dataset/MAIS_hackaton/food_images_251/all_images_256")
    dataset_dir = Path("D:/dataset/MAIS_hackaton/food_images_251/pytorch_dataset_256")
    p_train = Path("D:/dataset/MAIS_hackaton/food_images_251/pytorch_dataset_256/train")
    p_test = Path("D:/dataset/MAIS_hackaton/food_images_251/pytorch_dataset_256/test")

    # test_annot = pd.read_csv("D:/dataset/MAIS_hackaton/food_images_251/annot/test_info.csv")
    train_annot = pd.read_csv("D:/dataset/MAIS_hackaton/food_images_251/annot/train_info.csv")
    valid_annot = pd.read_csv("D:/dataset/MAIS_hackaton/food_images_251/annot/val_info.csv")

    all_annot = pd.concat([train_annot, valid_annot]).set_index("names")
    print(all_annot)
    print(all_annot.index)
    print(all_annot.loc["train_101735.jpg"].number)

    """
    p1 = Path("D:/dataset/MAIS_hackaton/food_images_251/train_set")
    p2 = Path("D:/dataset/MAIS_hackaton/food_images_251/test_set")
    p3 = Path("D:/dataset/MAIS_hackaton/food_images_251/val_set")

    all_paths = select_correct_images([p1, p2, p3])

    for i in all_paths:
        resize_and_move_img(i, p_save)
    # """

    all_paths = list(p_save.glob("*.jpg"))
    train_paths, test_paths = create_train_test(all_paths, 0.12)
    create_dataset(train_paths, test_paths, all_annot, dataset_dir)


