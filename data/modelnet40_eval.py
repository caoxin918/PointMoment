import os
import h5py
import glob

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from data import data_utils as d_utils

trans = transforms.Compose(

            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudNormalize(),
            ])


def load_modelnet_data(partition):
    BASE_DIR = '../'
    DATA_DIR = os.path.join(BASE_DIR, 'datasets')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_label = all_label.squeeze()
    return all_data, all_label


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_modelnet_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        point_cloud = self.data[item][:self.num_points]
        label = self.label[item]
        point_cloud = trans(point_cloud)
        return point_cloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    dataset = ModelNet40(1024)
    test_dataset = ModelNet40(2048, partition='test')
    print(len(dataset))
    print(len(test_dataset))
    print(dataset[0][0][0].shape, dataset[0][1])
    print(test_dataset[0][0].shape, test_dataset[0][1])



