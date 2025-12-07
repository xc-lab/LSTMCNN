import pandas as pd
import numpy as np
import csv
import random

def read_signal_csv(path, if_remove):
    # print('####%s' % (path))
    data = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            if len(row) == 1:
                original_data_length = int(row[0])
            if len(row) == 7:
                if if_remove:
                    if row[3] == '1':  # 这个值为0的话代表笔处于悬空，为1的话代表笔接触屏幕
                        row_new = [float(i) for i in row]
                        data.append(row_new)
                else:
                    row_new = [float(i) for i in row]
                    data.append(row_new)

    data = np.array(data)
    useful_data_lenght = len(data)
    # print('    original length:%d, useful length:%d' % (original_data_length, useful_data_lenght))
    return data


def get_column_data_scale(data):
    min_scale = np.min(data, axis=0)
    max_scale = np.max(data, axis=0)
    data_scale = np.vstack((min_scale, max_scale))
    return data_scale


def normalization_array_data(data, data_range):
    normalized_data = data.copy()
    for i in np.arange(data.shape[1]):
        min_value, max_value = data_range[:, i]
        normalized_data[:,i] = (normalized_data[:,i]-min_value)/(max_value-min_value)
    return normalized_data


def get_colomn_scalled_difference(data, order, scale, dim_id):
    if dim_id:
        data_temp = data.copy()
        for i in np.arange(data.shape[1]):
            if i in dim_id:
                data_temp[:, i] = np.diff(data[:, i], n=order, axis=-1, append=0)
                data_temp[:, i] = scale[i] * data_temp[:, i]
        return data_temp[1:-2,:]
    else:
        return data


def get_random_sampling_patches(data, patch_size, stride_size):
    data_idx = np.arange(patch_size-1, data.shape[0]-1-patch_size)
    idx = data_idx[::stride_size]
    # num_samples = int(len(data_idx)/stride_size)
    # idx = random.sample(data_idx.tolist(), num_samples)
    patch_dataset = []
    for p in idx:
        patch_data = data[p:p + patch_size, :]
        patch_dataset.append(patch_data)
    return idx, patch_dataset


def get_patches_from_sequence(full_file_name, patch_size, stride_size,  compute_gradient, process_on_stroke, scale, dim_id, if_remove):

    data = read_signal_csv(full_file_name, if_remove) #[y,x,t,b,a,l,p]
    full_data_array  = np.delete(data, [2,3,], axis=1)
    full_data_array = np.hstack((full_data_array, full_data_array[:,[-1]]))
    data_scale = get_column_data_scale(full_data_array)

    if np.isnan(np.sum(full_data_array)):
        print('    the data has Nan value!')
    else:
        if (np.count_nonzero(data_scale[0, :] - data_scale[1, :]) == 6):
            full_data_array = normalization_array_data(full_data_array, data_scale)
            if compute_gradient:
                full_data_array = get_colomn_scalled_difference(full_data_array, order=1, scale=scale, dim_id=dim_id)
            if not np.isnan(np.sum(full_data_array)):
                idxs, patch_dataset = get_random_sampling_patches(full_data_array, patch_size, stride_size)
            else:
                print('    the data has Nan value!')
        else:
            print('    there are some eigenvalues in the data that do not change.')

    return data, idxs, patch_dataset
