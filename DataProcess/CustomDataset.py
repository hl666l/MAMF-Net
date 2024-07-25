import pydicom
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd


def dcm2array(dcm_directory):
    """
    :param dcm_directory: 图片源文件路径
    :return:无
    """
    ds = pydicom.dcmread(dcm_directory)
    dcm_data = ds.pixel_array
    dcm_data = dcm_data.astype(np.float32)
    return dcm_data


def get_data(name, csv_file):
    data = pd.read_csv(csv_file)
    if 2 < len(name.split('_')):
        n = name.split('_')[0] + "_" + name.split('_')[1]
    else:
        n = name.split('_')[0]
    row_data = data[data.iloc[:, 0] == n]
    label = int(row_data.iloc[0, 1])  # 0 表示第一行，1 表示第二列
    data = row_data.iloc[0, 2:].astype(float)
    return label, data


class CustomDataset(Dataset):

    def __init__(self, data_folder_list, label_csv, transform=None):
        """
        """
        self.data_folder = data_folder_list
        self.label = label_csv
        self.transform = transform

    def __len__(self):
        #  样本个数
        return len(self.data_folder)

    def __getitem__(self, idx):
        # 获取一个病例的所有图片（dcm）名称
        name = self.data_folder[idx].split("/")[-1]
        label, csv_data = get_data(name, csv_file=self.label)
        csv_data = torch.Tensor(csv_data)

        image_name_files = os.listdir(self.data_folder[idx])  # 获取图片名称
        merged_image = []
        for name in image_name_files:
            path = os.path.join(self.data_folder[idx], name)
            image = dcm2array(path)  # 返回numpy.ndarray
            merged_image.append(image)
        data = np.stack(merged_image, axis=0)  # 列表中所有【512,512】numpy.ndarray堆叠成【50,512,512】
        data = data.transpose(1, 2, 0)  # 【50,512,512】-》【512,512,50】
        # 如果定义了transform，应用预处理操作
        if self.transform:
            data = self.transform(data)  # 需要输如[h，w，c]形式的ndarray or PILImage类型数据
        return data, csv_data, label
