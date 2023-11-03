import pandas as pd
import numpy as np
import json
import random
import matplotlib.pyplot as plt


def show_data_sequence(data,index):
    X = np.arange(data.shape[0])
    for i in np.arange(data.shape[1]):
        plt.plot(X, data[:,i])

    plt.xlabel("X Axis", fontdict={'size': 16})
    plt.ylabel("Y Axis", fontdict={'size': 16})
    plt.legend(index,loc='upper right')
    plt.show()


def normalized_frame_data(df, data_range):
    result = df.copy()
    for i, feature_name in enumerate(df.columns):
        min_value, max_value = data_range[:,i]
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def normalization_array_data(data, data_range):
    normalized_data = data.copy()
    for i in np.arange(data.shape[1]):
        min_value, max_value = data_range[:, i]
        normalized_data[:,i] = (normalized_data[:,i]-min_value)/(max_value-min_value)
    return normalized_data


def get_full_data_array(frame_data, index):
    dim = len(index)
    temp_data = np.zeros((1, dim))
    for j, stroke_idx in enumerate(frame_data):
        stroke = pd.DataFrame(stroke_idx)
        stroke_frame = stroke[index]
        stroke_data = stroke_frame.to_numpy()
        temp_data = np.vstack((temp_data, stroke_data))
    data = temp_data[1:,:]
    return data


def get_column_data_scale(data):
    min_scale = np.min(data, axis=0)
    max_scale = np.max(data, axis=0)
    data_scale = np.vstack((min_scale, max_scale))
    return data_scale


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
    num_samples = int(len(data_idx)/stride_size)
    idx = random.sample(data_idx.tolist(), num_samples)
    patch_dataset = []
    for p in idx:
        patch_data = data[p:p + patch_size, :]
        patch_dataset.append(patch_data)
    return patch_dataset


def get_patches_from_sequence(full_file_name, patch_size, stride_size,  compute_gradient, process_on_stroke, scale, dim_id):

    with open(full_file_name) as curr_file:
        test_data = json.load(curr_file)
    frame_data = test_data['data']

    patch_dataset = []
    index = ['a', 'l', 'p', 'x', 'y']
    full_data_array = get_full_data_array(frame_data, index)
    data_scale = get_column_data_scale(full_data_array)

    if np.isnan(np.sum(full_data_array)):
        print('The data has Nan value (%s)!'%(full_file_name))
    else:
        if process_on_stroke:
            if (np.count_nonzero(data_scale[0,:]- data_scale[1,:])==5):
                for j, stroke_idx in enumerate(frame_data):
                    patch_data = []
                    stroke = pd.DataFrame(stroke_idx)
                    stroke_frame = stroke[index]
                    stroke_data = stroke_frame.to_numpy()
                    stroke_data = normalization_array_data(stroke_data, data_scale)
                    if compute_gradient:
                        stroke_data = get_colomn_scalled_difference(stroke_data, order=1, scale=scale, dim_id=dim_id)

                    if not np.isnan(np.sum(stroke_data)):
                        patch_data = get_random_sampling_patches(stroke_data, patch_size, stride_size)
                    else:
                        print('The data after the first difference has Nan value (%s)!' % (full_file_name))
                    patch_dataset.extend(patch_data)
            else:
                print('There are some eigenvalues in the data that do not change (%s).'%(full_file_name))
        else:
            if (np.count_nonzero(data_scale[0, :] - data_scale[1, :]) == 5):
                full_data_array = normalization_array_data(full_data_array, data_scale)
                if compute_gradient:
                    full_data_array = get_colomn_scalled_difference(full_data_array, order=1, scale=scale, dim_id=dim_id)
                if not np.isnan(np.sum(full_data_array)):
                    patch_dataset = get_random_sampling_patches(full_data_array, patch_size, stride_size)
                else:
                    print('The data has Nan value (%s)!' % (full_file_name))
            else:
                print('There are some eigenvalues in the data that do not change (%s).'%(full_file_name))

    return patch_dataset
