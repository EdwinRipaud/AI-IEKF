from scipy.spatial.transform import Rotation
from torch.utils.data.dataset import Dataset
from termcolor import cprint
from navpy import lla2ned
import numpy as np
import glob
import os


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


class BaseDataset(Dataset):
    def __init__(self, raw_data_path, processed_data_path):
        super(BaseDataset, self).__init__()
        self.raw_data_path = raw_data_path
        self.process_data_path = processed_data_path
        create_folder(self.process_data_path)
        print(f"Runs data processing function to turn raw data from ({self.raw_data_path}) into cleaned data ready to "
              f"be analyzed (saved in {self.process_data_path})")


if __name__ == '__main__':
    path_raw_data = "../data/raw"
    path_processed_data = "../data/processed"
    kitti_dataset = BaseDataset(path_raw_data, path_processed_data)
