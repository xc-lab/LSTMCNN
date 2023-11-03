'''
论文中信号处理部分
'''
import os
import re
import json
import pywt
from matplotlib.font_manager import FontProperties

from data_utils.utils import *


def percent_range(dataset, min=0.20, max=0.80):
    range_max = np.percentile(dataset, max * 100)
    range_min = -np.percentile(-dataset, (1 - min) * 100)
    # 剔除前20%和后80%的数据
    new_data = []
    for value in dataset:
        if value < range_max and value > range_min:
            new_data.append(value)
    return new_data



def signal_preprocess(A, L, P, X, Y, title):
    plt.rcParams.update({'font.size': 10})
    plt.figure(dpi=100, figsize=(20, 3))
    # plt.plot(A, c='#FF8C00', label='\'a\'', linewidth=5.0)
    # plt.plot(L, c='#FFD700', label='\'l\'', linewidth=5.0)
    # plt.plot(P, c='#00BFFF', label='\'p\'', linewidth=5.0)
    plt.plot(Y, c='#006400', label='\'y\'', linewidth=5.0)
    # plt.plot(X, c='#7FFF00', label='\'x\'', linewidth=5.0)
    # plt.legend()
    plt.tick_params(labelsize=10)
    plt.xticks([])
    plt.yticks([])
    # plt.title(title)
    # plt.axis('off')
    # plt.show()
    plt.savefig("D:/Projects/Papers/Parkinson/figure1.png", dpi=500, bbox_inches='tight')



def cwt_transform(data):
    plt.figure(dpi=300,figsize=(24, 6))
    plt.specgram(data, NFFT=64, noverlap=32)
    # plt.ylabel("Frequence(dBHz)")
    # plt.xlabel("Times(s)")
    plt.colorbar()
    plt.tick_params(labelsize=15)
    # plt.clim(-1, 1)
    # plt.off()
    plt.show()



if __name__=='__main__':

    data_path = '../data/test_data'
    label_path = 'KT'
    pattern_lists = {'pcontinue', 'pcopy', 'ptrace'}
    compute_gradient = True
    dim_id = [3, 4]
    scale = [1, 1, 1, 100, 10]

    list_files = {
        # '../data/raw_data/PD/PD23/PD-23-20190506-071210-ptrace_dda09d7c-7ca1-48cb-b163-9f837a2c33cf.json',
        '../data/raw_data/KT/KT115/20190523-171459-KT115-ptrace_7e02ddc3-9be8-4570-bb28-78f28c86cb95.json',
        # '../data/test_data/KT/KT103/20190509-090044-KT103-pcontinue_1fc70d43-4638-40b9-a73d-a175fcf570bf.json'

    }
    for f, full_file_name in enumerate(list_files):
        with open(full_file_name) as curr_file:
            test_data = json.load(curr_file)
        frame_data = test_data['data']

        index = ['a', 'l', 'p', 'x', 'y']
        full_data_array = get_full_data_array(frame_data, index)
        data_scale = get_column_data_scale(full_data_array)

        if (np.count_nonzero(data_scale[0, :] - data_scale[1, :]) == 5):
            full_data_array = normalization_array_data(full_data_array, data_scale)
            if compute_gradient:
                full_data_array = get_colomn_scalled_difference(full_data_array, order=1, scale=scale, dim_id=dim_id)
            a_signal = full_data_array[:,0]+3
            l_signal = full_data_array[:,1]+2
            p_signal = full_data_array[:,2]+1
            x_signal = full_data_array[:,3]
            y_signal = full_data_array[:,4]
            x_signal = np.array(percent_range(x_signal, min=0.01, max=0.99))+0.5
            y_signal = np.array(percent_range(y_signal, min=0.01, max=0.99))
            signal_preprocess(a_signal, l_signal, p_signal, x_signal, y_signal, full_file_name)
            # cwt_transform(x_signal)































