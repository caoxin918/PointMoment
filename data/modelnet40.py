import os
import h5py
import glob

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from data import data_utils_strl as d_utils

trans_1 = transforms.Compose(
            # [
            #     d_utils.PointcloudToTensor(),
            #     d_utils.PointcloudNormalize(),
            #     d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
            #     d_utils.PointcloudRotate(),
            #     d_utils.PointcloudTranslate(0.5, p=1),
            #     d_utils.PointcloudJitter(p=1),
            #     d_utils.PointcloudRandomInputDropout(p=1),
            # ]
            [
            d_utils.PointcloudToTensor(),
            d_utils.PointcloudUpSampling(max_num_points=4096, centroid="random"),
            d_utils.PointcloudRandomCrop(p=0.5, min_num_points=2048),
            d_utils.PointcloudNormalize(),
            d_utils.PointcloudRandomCutout(p=0.5, min_num_points=2048),
            d_utils.PointcloudScale(p=1),
            # d_utils.PointcloudRotate(p=1, axis=[0.0, 0.0, 1.0]),
            d_utils.PointcloudRotatePerturbation(p=1),
            d_utils.PointcloudTranslate(p=1),
            d_utils.PointcloudJitter(p=1),
            d_utils.PointcloudRandomInputDropout(p=1),
            d_utils.PointcloudSample(num_pt=2048)
            ]

            # [
            #     d_utils.PointcloudToTensor(),
            #     d_utils.PointcloudNormalize(),
            #     d_utils.PointcloudRandomCrop(p=0.5, min_num_points=1024),
            #     d_utils.PointcloudTranslate(p=1),
            #     d_utils.PointcloudJitter(p=1),
            #     d_utils.PointcloudScale(p=0.5),
            #     d_utils.PointcloudSample(num_pt=1024)
            # ]
            )

trans_2 = transforms.Compose(
            # [
            #     d_utils.PointcloudToTensor(),
            #     d_utils.PointcloudNormalize(),
            #     d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
            #     d_utils.PointcloudRotate(),
            #     d_utils.PointcloudTranslate(0.5, p=1),
            #     d_utils.PointcloudJitter(p=1),
            #     d_utils.PointcloudRandomInputDropout(p=1),
            # ]
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudUpSampling(max_num_points=4096, centroid="random"),
                d_utils.PointcloudRandomCrop(p=0.5, min_num_points=2048),
                d_utils.PointcloudNormalize(),
                d_utils.PointcloudRandomCutout(p=0.5, min_num_points=2048),
                d_utils.PointcloudScale(p=1),
                # d_utils.PointcloudRotate(p=1, axis=[0.0, 0.0, 1.0]),
                d_utils.PointcloudRotatePerturbation(p=1),
                d_utils.PointcloudTranslate(p=1),
                d_utils.PointcloudJitter(p=1),
                d_utils.PointcloudRandomInputDropout(p=1),
                d_utils.PointcloudSample(num_pt=2048)
            ]

            # [
            #     d_utils.PointcloudToTensor(),
            #     d_utils.PointcloudNormalize(),
            #     d_utils.PointcloudRandomCrop(p=0.5, min_num_points=1024),
            #     d_utils.PointcloudTranslate(p=1),
            #     d_utils.PointcloudJitter(p=1),
            #     d_utils.PointcloudScale(p=0.5),
            #     d_utils.PointcloudSample(num_pt=1024)
            # ]
        )

trans_test = transforms.Compose(

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
    return all_data, all_label


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_modelnet_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        point_cloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            point_t1 = trans_1(point_cloud)
            point_t2 = trans_2(point_cloud)
            point_trans = (point_t1, point_t2)
            return point_trans, label
        else:
            point_cloud = trans_test(point_cloud)
            return point_cloud, label

    def __len__(self):
        return self.data.shape[0]


class ModelNet40SVM(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_modelnet_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    dataset = ModelNet40(1024)
    test_dataset = ModelNet40(2048, partition='test')
    print(len(dataset))
    print(len(test_dataset))
    print(dataset[0][0][0].shape, dataset[0][1])
    print(test_dataset[0][0].shape, test_dataset[0][1])



