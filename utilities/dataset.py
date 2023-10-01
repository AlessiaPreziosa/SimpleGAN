# import libraries
import json
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import Subset

# utilities for preprocessing dataset and loading it

# root directory: change it to the directory where the Polyvore Outfit Dataset is
root_dir = '/Users/alessiapreziosa/Downloads/'


def make_dictionary(directory, file):
    df = pd.DataFrame(json.load(open(f'{root_dir}polyvore_outfits/{directory}/{file}.json')))
    df_items = [[df['items'][i][j]['item_id'] for j in range(len(df['items'][i]))] for i in range(df.shape[0])]

    D = {}
    i = 0
    for el in list(df['set_id']):
        D[el] = df_items[i]
        i += 1

    return D


def cast(data_list, data_type):
    return list(map(data_type, data_list))


def save(clothes, new_root_dir, dataset, capo):
    for i in range(len(clothes)):
        path = f'{new_root_dir}/{clothes[i]}.jpg'
        image = Image.open(path)
        image_name = str(i) + '_' + clothes[i] + '.jpg'
        image.save(f'{root_dir}outfits/{dataset}/{capo}/{image_name}')


# augment train data
train_transform = transforms.Compose([
    transforms.Resize(128),  # resize to 128x128
    transforms.RandomHorizontalFlip(),  # add a random H flip
    transforms.RandomVerticalFlip(),  # add a random V flip
    transforms.ToTensor(),  # transform to a Float tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize
])

# transform to a Float Tensor
# and normalize each channel of test data as output[channel] = (input[channel] - mean[channel]) / std[channel]
test_transform = transforms.Compose([
    transforms.Resize(128),  # resize to 128x128
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def DataLoad(loader):
    iters = []

    for subset in ['tops', 'bottoms', 'shoes', 'accessories']:
        idx = [i for i in range(len(loader)) if loader.imgs[i][1] == loader.class_to_idx[subset]]
        iters.append(torch.utils.data.DataLoader(Subset(loader, idx), batch_size=128))

    return iters


def show(iterable, indexes):
    fig = plt.figure(figsize=(20, 20))

    imgs = []
    for X, _ in iterable:
        X = X[indexes]
        imgs = X.permute(0, 2, 3, 1) / 2 + 0.5  # un-normalize
        break

    for idx in np.arange(20):
        fig.add_subplot(4, 5, idx + 1)
        plt.imshow(imgs[idx])

    plt.show()
