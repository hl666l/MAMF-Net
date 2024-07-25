import os
from CustomDataset import CustomDataset as CuDa


def create_dataset(files_folder, table_csv, data_transform):
    train_data_name = os.listdir(files_folder)
    data_folder_list = []
    for name in train_data_name:
        data_folder_list.append(os.path.join(files_folder, name))
    custom_dataset = CuDa(data_folder_list, table_csv, transform=data_transform)
    return custom_dataset
