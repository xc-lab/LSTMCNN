#  -*- coding: utf-8 -*-
'''
KT-00, PD-01
'''

import os
import shutil
from data_utils.utils import *



def get_training_patches_dataset(data_path, output_path, pattern_lists, patch_size, stride_size, compute_gradient,
                                  process_on_stroke, scale, dim_id, if_remove):
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
        if label_path == 'HC':
            label_id = '00'
        elif label_path == 'PD':
            label_id = '01'
        else:
            print('There is no %s class.'%(label_path))
        test_files = os.listdir(os.path.join(data_path, label_path)) # ./data/raw_data/HC
        for t, test_path in enumerate(test_files):
            files = os.listdir(os.path.join(data_path, label_path, test_path)) # ./data/raw_data/HC/00026
            for f, file_path in enumerate(files):
                file_id = file_path[:-4]
                pattern_id = int(file_path[7:8])
                if pattern_id in pattern_lists:
                    json_file_name = data_path + '/' + label_path + '/' + test_path + '/' + file_path  # ./data/raw_data/KT/KT1/KT-01_8D6834FB_plcontinue_2017-11-05_16_45_13___d2f076a85a38476a99cec299d2c17419.json
                    full_data, idx_list, patches_data = get_patches_from_sequence(json_file_name, patch_size, stride_size[label_path], compute_gradient, process_on_stroke, scale, dim_id, if_remove)
                    if label_path == 'HC':
                        num_kt += len(patches_data)
                    elif label_path == 'PD':
                        num_pd += len(patches_data)
                    else:
                        print('    %s type data does not exist'%(label_path))
                    for p, patch_data in enumerate(patches_data):
                        patch_id = '{0:04d}'.format(p)
                        patch_full_name = patches_path + 'label_' + label_id + '_' + file_id + '_patch_' + patch_id + '.npy'
                        np.save(patch_full_name, patch_data)

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

            file = os.path.join(os.path.join(datasets_path, 'validation'), files[idx[i]])
            shutil.copy(os.path.join(patches_path, files[idx[i]]), file)


if __name__ == '__main__':
    if_remove = True #是否将悬空的点舍弃，True:舍弃，False:保留
    process_on_stroke = False  # 控制采样，在笔画上还是在整体数据上
    compute_gradient = True  # 控制是否进行一阶差分
    dim_id = [0,1]  # 控制对哪些特征进行一阶差分
    scale =  [100, 100, 1, 1, 1, 1] #控制哪些特征，进行一阶差分后扩大的倍数

    patch_size = 128 # 数据长度

    # 重采样时，前后patch data的间隔点数；第一个控制KT 7，第二个控制PD  16  (32: 8,17)\(64: 7,16)\(128: 7,16)\(256: 6,17); PL形式数据 (32: 11,24)\(64: 11,24)\(128: 10,23)\(256: 9,23)
    stride_size = {'HC':8, 'PD':9}

    path = '../data'
    rawdata_path = '../data/new_data'
    pattern_lists = {1}
    get_training_patches_dataset(rawdata_path, path, pattern_lists, patch_size, stride_size, compute_gradient, process_on_stroke, scale, dim_id, if_remove)
    get_training_data_from_patches(path, patch_size)




