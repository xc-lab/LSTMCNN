#  -*- coding: utf-8 -*-
'''
KT-00, PD-01
'''

import re
import os
import shutil
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_utils.utils import *


def get_raw_vectorized_dataset(data_path, output_path, pattern_lists, compute_gradient, process_on_stroke, scale, dim_id):
    if not os.path.exists(os.path.join(output_path,'patches')):
        os.mkdir(os.path.join(output_path,'patches'))
    patches_path = os.path.join(output_path, 'patches/X{patch_size}/'.format(patch_size=patch_size))
    if os.path.exists(patches_path):
        shutil.rmtree(patches_path)
        os.mkdir(patches_path)
    else:
        os.mkdir(patches_path)

    num_pd = 0
    num_kt = 0
    dataset_files = os.listdir(data_path) # ./data/raw_data
    for l, label_path in enumerate(dataset_files):
        if label_path == 'KT':
            label_id = '00'
        elif label_path == 'PD':
            label_id = '01'
        else:
            print('There is no %s class.'%(label_path))
        test_files = os.listdir(os.path.join(data_path, label_path)) # ./data/raw_data/KT
        for t, test_path in enumerate(test_files):
            test_id = '{0:03d}'.format(t)
            files = os.listdir(os.path.join(data_path, label_path, test_path)) # ./data/raw_data/KT/KT1
            for f, file_path in enumerate(files):
                file_id = '{0:03d}'.format(f)
                for pattern in pattern_lists:
                    if re.search(pattern, file_path):
                        pattern_id = pattern
                        full_file_name = data_path + '/' + label_path + '/' + test_path + '/' + file_path  # ./data/raw_data/KT/KT1/KT-01_8D6834FB_plcontinue_2017-11-05_16_45_13___d2f076a85a38476a99cec299d2c17419.json
                        print(full_file_name)

                        with open(full_file_name) as curr_file:
                            test_data = json.load(curr_file)
                        frame_data = test_data['data']

                        index = ['a', 'l', 'p', 'x', 'y']
                        full_data_array = get_full_data_array(frame_data, index)
                        data_scale = get_column_data_scale(full_data_array)

                        if np.isnan(np.sum(full_data_array)):
                            print('The data has Nan value (%s)!' % (full_file_name))
                        else:
                            if process_on_stroke:
                                if (np.count_nonzero(data_scale[0, :] - data_scale[1, :]) == 5):
                                    for j, stroke_idx in enumerate(frame_data):
                                        patch_data = []
                                        stroke = pd.DataFrame(stroke_idx)
                                        stroke_frame = stroke[index]
                                        stroke_data = stroke_frame.to_numpy()
                                        stroke_data = normalization_array_data(stroke_data, data_scale)
                                        if compute_gradient:
                                            stroke_data = get_colomn_scalled_difference(stroke_data, order=1,
                                                                                        scale=scale, dim_id=dim_id)

                                        if not np.isnan(np.sum(stroke_data)):
                                            show_data_sequence(stroke_data, index)
                                        else:
                                            print('The data after the first difference has Nan value (%s)!' % (
                                                full_file_name))

                                else:
                                    print('There are some eigenvalues in the data that do not change (%s).' % (
                                        full_file_name))
                            else:
                                if (np.count_nonzero(data_scale[0, :] - data_scale[1, :]) == 5):
                                    full_data_array = normalization_array_data(full_data_array, data_scale)
                                    if compute_gradient:
                                        full_data_array = get_colomn_scalled_difference(full_data_array, order=1,
                                                                                        scale=scale, dim_id=dim_id)
                                    if not np.isnan(np.sum(full_data_array)):
                                        #show_data_sequence(full_data_array, index)
                                        full_csv_file_name ='../data/csv_data/' + file_path[:-5]+'.csv'
                                        np.savetxt(full_csv_file_name, full_data_array, delimiter=',', fmt='%1.6f')
                                        # with open(full_csv_file_name, 'w', newline='') as csv_file_writer:
                                        #     writer = csv.writer(csv_file_writer, delimiter=',')
                                        #     writer.writerow(full_data_array)
                                    else:
                                        print('The data has Nan value (%s)!' % (full_file_name))
                                else:
                                    print('There are some eigenvalues in the data that do not change (%s).' % (
                                        full_file_name))



def get_training_patches_dataset(data_path, output_path, pattern_lists, patch_size, stride_size, compute_gradient,
                                  process_on_stroke, scale, dim_id):
    if not os.path.exists(os.path.join(output_path,'patches')):
        os.mkdir(os.path.join(output_path,'patches'))
    patches_path = os.path.join(output_path, 'patches/X{patch_size}/'.format(patch_size=patch_size))
    if os.path.exists(patches_path):
        shutil.rmtree(patches_path)
        os.mkdir(patches_path)
    else:
        os.mkdir(patches_path)

    num_pd = 0
    num_kt = 0
    dataset_files = os.listdir(data_path) # ./data/raw_data
    for l, label_path in enumerate(dataset_files):
        if label_path == 'KT':
            label_id = '00'
        elif label_path == 'PD':
            label_id = '01'
        else:
            print('There is no %s class.'%(label_path))
        test_files = os.listdir(os.path.join(data_path, label_path)) # ./data/raw_data/KT
        for t, test_path in enumerate(test_files):
            test_id = '{0:03d}'.format(t)
            files = os.listdir(os.path.join(data_path, label_path, test_path)) # ./data/raw_data/KT/KT1
            for f, file_path in enumerate(files):
                file_id = '{0:03d}'.format(f)
                for pattern in pattern_lists:
                    if re.search(pattern, file_path):
                        pattern_id = pattern
                        json_file_name = data_path + '/' + label_path + '/' + test_path + '/' + file_path  # ./data/raw_data/KT/KT1/KT-01_8D6834FB_plcontinue_2017-11-05_16_45_13___d2f076a85a38476a99cec299d2c17419.json
                        print(json_file_name)
                        patches_data = get_patches_from_sequence(json_file_name, patch_size, stride_size[label_path], compute_gradient, process_on_stroke, scale, dim_id)
                        if label_path == 'KT':
                            num_kt += len(patches_data)
                        elif label_path == 'PD':
                            num_pd += len(patches_data)
                        else:
                            print('%s type data does not exist'%(label_path))
                        # for p, patch_data in enumerate(patches_data):
                        #     patch_id = '{0:04d}'.format(p)
                        #     patch_full_name = patches_path + 'label_' + label_id + '_' + test_path + '_file_' + file_id + '_' + pattern_id + '_patch_' + patch_id + '.npy'
                        #     np.save(patch_full_name, patch_data)

    print('KT data set:%d, PD data set:%d '%(num_kt, num_pd))



def get_training_data_from_patches(path, patch_size):

    datasets_path = os.path.join(path, 'datasets')
    if os.path.exists(datasets_path):
        shutil.rmtree(datasets_path)
        os.mkdir(datasets_path)
        os.mkdir(os.path.join(datasets_path, 'training'))
        os.mkdir(os.path.join(datasets_path, 'validation'))
    else:
        os.mkdir(datasets_path)
        os.mkdir(os.path.join(datasets_path, 'training'))
        os.mkdir(os.path.join(datasets_path, 'validation'))

    patches_path = os.path.join(path, 'patches/X{patch_size}/'.format(patch_size=patch_size))
    files = os.listdir(patches_path)
    N = len(files)
    idx = np.arange(N)
    np.random.shuffle(idx)
    for i in range(N):
        if i<(0.8*N):
            file = os.path.join(os.path.join(datasets_path, 'training'), files[idx[i]])
            shutil.copy(os.path.join(patches_path, files[idx[i]]), file)

        else:
        # if i < (0.2*N):
            file = os.path.join(os.path.join(datasets_path, 'validation'), files[idx[i]])
            shutil.copy(os.path.join(patches_path, files[idx[i]]), file)


if __name__ == '__main__':

    process_on_stroke = False  # 控制采样，在笔画上还是在整体数据上
    compute_gradient = True  # 控制是否进行一阶差分
    dim_id = [3,4]  # 控制对哪些特征进行一阶差分
    scale =  [1, 1, 1, 100, 100] #控制哪些特征，进行一阶差分后扩大的倍数

    patch_size = 128 # 数据长度

    # 重采样时，前后patch data的间隔点数；第一个控制KT 7，第二个控制PD  16  (32: 8,17)\(64: 7,16)\(128: 7,16)\(256: 6,17); PL形式数据 (32: 11,24)\(64: 11,24)\(128: 10,23)\(256: 9,23)
    stride_size = {'KT':14, 'PD':32}

    path = '../data'
    rawdata_path = '../data/raw_data'
    pattern_lists = {'pcontinue',  'pcopy',  'ptrace'}
    # get_raw_vectorized_dataset(rawdata_path, path, pattern_lists, compute_gradient, process_on_stroke, scale, dim_id)
    get_training_patches_dataset(rawdata_path, path, pattern_lists, patch_size, stride_size, compute_gradient, process_on_stroke, scale, dim_id)
    # get_training_data_from_patches(path, patch_size)




