import re
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def get_data(input_data_path):
    """ Create train/test data set and label as type: nd-array """
    output_train_data = []
    output_train_label = []
    output_test_data = []
    output_test_label = []
    file = pd.read_csv(input_data_path)
    content = pd.DataFrame(file).values
    for item in content:
        if item[2] == 'Training':
            output_train_label.append(item[0])
            pixel_list = [int(num) for num in re.findall(r'\d+', item[1])]
            output_train_data.append(np.reshape(np.array(pixel_list), [48, 48]))
        else:
            output_test_label.append(item[0])
            pixel_list = [int(num) for num in re.findall(r'\d+', item[1])]
            output_test_data.append(np.reshape(np.array(pixel_list), [48, 48]))
    output_train_data, output_train_label = np.array(output_train_data), np.array(output_train_label)
    output_test_data, output_test_label = np.array(output_test_data), np.array(output_test_label)
    return output_train_data, output_train_label, output_test_data, output_test_label


def calculate_mean_std(input_data):
    """ Used for getting the average and standard derivation of the dataset """
    data = np.true_divide(input_data, 255)
    mu = np.sum(np.mean(data, axis=(1, 2))) / len(data)
    std = np.sum(np.std(data, axis=(1, 2))) / len(data)
    return mu, std


class custom_dataset(Dataset):
    """ Used for creating data loader """
    def __init__(self, input_data_set, input_label_set, input_transform=None):
        super(custom_dataset, self).__init__()
        self.data_set = input_data_set
        self.label_set = input_label_set
        self.transform = input_transform

    def __getitem__(self, item):
        output_data = Image.fromarray(np.uint8(self.data_set[item]))
        output_label = torch.Tensor(self.label_set[item]).long()
        if self.transform is not None:
            output_data = self.transform(output_data)
        return output_data, output_label

    def __len__(self):
        return len(self.label_set)
