'''
Intros：
Author: Xuechao Wang
'''
import numpy as np
import csv

def read_csv(path):
    # 读取数据(根据数据不同进行修改)
    data = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            if len(row) == 7:
                row_new = [float(i) for i in row]
                data.append(row_new)
    data = np.array(data)
    return data


def resampling(data, patch_size, stride_size):
    # 重采样增加样本量
    data_idx = np.arange(patch_size-1, data.shape[0]-1-patch_size)
    idx = data_idx[::stride_size]
    patch_dataset = []
    for p in idx:
        patch_data = data[p:p + patch_size, :]
        patch_dataset.append(patch_data)
    return patch_dataset


def get_patches_from_sequence(path, patch_size, stride_size):
    data = read_csv(path)
    patch_dataset = resampling(data, patch_size, stride_size)
    return patch_dataset


if __name__ == '__main__':
    path = './data/raw_data/HC/00026/00026__1_1.svc'

    patch_size = 128  # 重采样数据长度
    stride_size = 8   # 重采样重叠程度

    patches = get_patches_from_sequence(path, patch_size, stride_size)