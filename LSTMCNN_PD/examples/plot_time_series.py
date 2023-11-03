'''
将数据画成时间序列的形式
'''
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
import PIL.Image as Image
import re
from sklearn import preprocessing

from data_utils.utils import *

def matplotlib_sequence(data, data_range):
    '''画各个特征曲线图'''
    n_data = data
    # n_data = normalization_array_data(data, data_range)  # 对a,l,p三个特征进行最大最小归一化处理
    A = np.array(n_data[:, 0])
    L = np.array(n_data[:, 1])
    P = np.array(n_data[:, 2])
    X = np.array(n_data[:, 3])
    Y = np.array(n_data[:, 4])
    T = np.array(n_data[:, 5])
    # plt.rcParams.update({'font.size': 10})
    fig = plt.figure(figsize=(14, 8))

    ax1 = fig.add_subplot(321)
    ax1.plot(A, linewidth=3.0)

    ax2 = fig.add_subplot(322)
    ax2.plot(L, linewidth=3.0)

    ax3 = fig.add_subplot(323)
    ax3.plot(P, linewidth=3.0)

    ax4 = fig.add_subplot(324)
    ax4.plot(T, linewidth=3.0)

    ax5 = fig.add_subplot(325)
    ax5.plot(X, linewidth=3.0)

    ax6 = fig.add_subplot(326)
    ax6.plot(-Y, linewidth=3.0)


    # plt.plot(A, c='#FF8C00', label='\'a\'', linewidth=3.0)
    # plt.plot(L, c='#FFD700', label='\'l\'', linewidth=3.0)
    # plt.plot(P[100:450], c='#1E90FF', label='\'p\'', linewidth=2.0)
    # plt.plot(X, c='#7FFF00', label='\'x\'', linewidth=3.0)
    # plt.plot(Y, c='#006400', label='\'y\'', linewidth=3.0)
    # plt.plot(T, c='k', label='t-time', linewidth=1.0)
    # plt.legend()
    # plt.plot(P, c='#000000', label='\'p\'', linewidth=2.0)

    # plt.plot(A[100:450],   linewidth=2.0)
    # plt.plot(L[100:450],   linewidth=2.0)
    # plt.plot(P[100:450],   linewidth=2.0)
    # plt.plot(Y[100:450],   linewidth=2.0)
    # plt.plot(X[100:450],   linewidth=2.0)

    # plt.tick_params(labelsize=20)
    # plt.title('Time')
    # plt.axis('off')
    # plt.show()
    plt.savefig("D:/Projects/Papers/Parkinson/figure1.png", dpi=500, bbox_inches='tight')

def matplotlib_violin(data, data_range):
    n_data = data
    # plt.style.use('_mpl-gallery')
    plt.figure(dpi=1000)

    # make data:
    A = np.array(n_data[:, 0])
    L = np.array(n_data[:, 1])
    P = np.array(n_data[:, 2])
    color = ['#006400', '#7FFF00', '#9400D3']

    data = list([A, L, P])

    # plot:
    fig, ax = plt.subplots()

    vp = ax.violinplot(data, [1, 2, 3], widths=1,
                       showmeans=False, showmedians=False, showextrema=False)

    i = 0
    for pc in vp['bodies']:
        pc.set_facecolor(color[i])
        pc.set_edgecolor(color[i])
        i += 1

    # styling:
    for body in vp['bodies']:
        body.set_alpha(0.5)

    labels = ['','','a','', 'l','', 'p']
    ax.set_xticklabels(labels)
    plt.tick_params(labelsize=15)
    font = {
            'size': 15,
            }
    plt.xlabel('feature', font)
    plt.ylabel("value", font)

    plt.show()

def filter_extreme_3sigma(data,n,times):
    # times进行times次3sigma处理
    series = data.copy()
    for i in range(times):
        mean = series.mean()
        std = series.std()
        max_range = mean + n*std
        min_range = mean - n*std
        series = np.clip(series,min_range,max_range)
    return series

def matplotlib_backward(data):
    scale = [1, 1, 1, 1, 1, 1]
    dim_id = [ 3, 4]
    order = 1
    # n_data = get_colomn_scalled_difference(data, order, scale, dim_id)
    n_data = data
    X = np.array(n_data[5:-5, 3])
    X = filter_extreme_3sigma(X, 1.5, 3)
    Y = np.array(n_data[5:-5, 4])
    Y = filter_extreme_3sigma(Y, 3, 3)
    A = np.array(n_data[5:-5, 0])
    A = filter_extreme_3sigma(A, 1.3, 3)

    L = np.array(n_data[5:-5, 1])
    L = filter_extreme_3sigma(L, 1.3, 3)

    P = np.array(n_data[5:-5, 2])
    P = filter_extreme_3sigma(P, 1.3, 3)

    a = 100
    b = 200
    c = 2400
    bwith = 2
    x1=range(0,a)
    x2=range(a,b)
    x3=range(b,c)

    plt.figure(figsize=(30, 2.5))

    plt.plot(x1, X[0:a],   c='#1E90FF', label='\'p\'', linewidth=2.0)
    plt.plot(x2, X[a:b], c='#B22222', label='\'p\'', linewidth=2.0)
    plt.plot(x3, X[b:c], c='#1E90FF', label='\'p\'', linewidth=2.0)

    # ax = plt.gca()  # 获取边框
    # ax.spines['bottom'].set_linewidth(bwith)
    # ax.spines['left'].set_linewidth(bwith)
    # ax.spines['top'].set_linewidth(bwith)
    # ax.spines['right'].set_linewidth(bwith)

    # plt.legend()

    # plt.tick_params(labelsize=30)
    # plt.title('Time')
    # plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    # plt.savefig("D:/Projects/Papers/Parkinson/figure1.png", dpi=500, bbox_inches='tight')

def get_colomn_scalled_difference_(data, order, scale, dim_id):
    if dim_id:
        data_temp = data.copy()
        for i in np.arange(data.shape[1]):
            if i in dim_id:
                a = np.diff(data[:, i], n=order, axis=-1, append=0)
                d = np.pad(a, (0, len(data_temp[:,i])-len(a)), 'constant', constant_values=(0, 0))
                data_temp[:, i] = d
                # data_temp[:, i] = np.diff(data[:, i], n=order, axis=-1, append=0)
                data_temp[:, i] = scale[i] * data_temp[:, i]
        return data_temp[1:-2,:]
    else:
        return data

def matplotlib_competation(data_kt, data_pd):
    '''画各个特征曲线图'''

    order = 0
    X_kt = data_kt[:400]
    X_pd = data_pd[10:410]

    # X_kt = np.diff(X_kt, n=order, axis=0, append=0)[:-3]
    # X_pd = np.diff(X_pd, n=order, axis=0, append=0)[:-3]

    # for i in range(len(X_kt)):
    #     if -3>X_kt[i] or X_kt[i]>3:
    #         X_kt[i] = 0
    # for i in range(len(X_pd)):
    #     if -3>X_pd[i] or X_pd[i]>3:
    #         X_pd[i] = 0
    #
    # min_scale = np.min(X_kt, axis=0)
    # max_scale = np.max(X_kt, axis=0)
    # for i in range(len(X_kt)):
    #     X_kt[i] = (X_kt[i]-min_scale)/(max_scale-min_scale)
    #
    #
    # min_scale = np.min(X_pd, axis=0)
    # max_scale = np.max(X_pd, axis=0)
    # for i in range(len(X_pd)):
    #     X_pd[i] = (X_pd[i] - min_scale) / (max_scale - min_scale)


    plt.rcParams.update({'font.size': 25})

    plt.figure(figsize=(11, 8))
    my_y_ticks = np.arange(0, 1, 0.1)
    plt.yticks(my_y_ticks)
    plt.plot(X_kt, c='#006400', label=': HC', linewidth=5.0)
    plt.plot(X_pd, c='#FF8C00', label=': PD', linewidth=5.0)
    plt.legend(loc='upper left', fontsize=25)
    plt.xlabel('data point')
    plt.ylabel('x-coordinate')

    plt.tick_params(labelsize=30)
    # plt.title('Time')
    # plt.axis('off')
    plt.show()
    # plt.savefig("D:/Projects/Papers/Parkinson/original_signal.png", dpi=500, bbox_inches='tight')



if __name__=='__main__':

    '''empty data'''
    # ./data/test_data/KT/KT21/KT-21_F4675D6C_plcontinue_2018-04-10_12_06_18___9f7cb18fc2874450940412332a83982c.json,
    # ./data/test_data/PD/PD15/PD-15_2CEE8303_plcopy_2018-03-05_08_48_43___b2501ec53c354b45a5fd56e064a35c13.json


    '''incorrect data'''
    # ./data/test_data/PD/PD15/PD-15_2CEE8303_plcontinue_2018-03-05_08_46_16___86cb047f1ed240cda2496a92c02615c0.json

    '''seperation data'''
    # ../data/raw_data/kt/KT23/KT-23_96EC7657_pcontinue_2018-04-10_14_25_50___02947de59b4c4e4da9b37a93bca9340e.json



    list_files = {
        # '../data/test_data/PD/PD15/PD-15_2CEE8303_plcontinue_2018-03-05_08_46_16___86cb047f1ed240cda2496a92c02615c0.json',
        # '../data/test_data/KT/KT21/KT-21_F4675D6C_plcontinue_2018-04-10_12_06_18___9f7cb18fc2874450940412332a83982c.json',
        # '../data/test_data/PD/PD9/PD-09_FB1C35EA_pcontinue_2017-12-12_22_07_35___4e95c808a64b4744b5408746f79cd3cd.json',
        # '../data/raw_data/kt/KT23/KT-23_96EC7657_pcontinue_2018-04-10_14_25_50___02947de59b4c4e4da9b37a93bca9340e.json',
        # '../data/test_data/PD/PD15/PD-15_2CEE8303_plcopy_2018-03-05_08_48_43___b2501ec53c354b45a5fd56e064a35c13.json',
        # '../data/test_data/KT/KT103/20190509-090044-KT103-pcontinue_1fc70d43-4638-40b9-a73d-a175fcf570bf.json',
        '../data/raw_data/KT/KT115/20190523-171011-KT115-spiral_ef251022-7605-4430-8d16-83961cdb055a.json',
        # '../data/raw_data/PD/PD23/PD-23-20190506-071210-ptrace_dda09d7c-7ca1-48cb-b163-9f837a2c33cf.json'

    }
    for f, full_file_name in enumerate(list_files):
        with open(full_file_name) as curr_file:
            test_data = json.load(curr_file)

        frame_data = test_data['data']
        index = ['a', 'l', 'p', 'x', 'y', 't']
        full_data_array = get_full_data_array(frame_data, index)  # 得到数组
        data_scale = get_column_data_scale(full_data_array)  # 得到每一个特征的最值范围
        matplotlib_sequence(full_data_array, data_scale)
        # matplotlib_violin(full_data_array, data_scale)
        # matplotlib_backward(full_data_array, data_scale)

    # path_kt = '../data/raw_data/KT/KT115/20190523-171011-KT115-spiral_ef251022-7605-4430-8d16-83961cdb055a.json'
    # with open(path_kt) as curr_file:
    #     test_data_kt = json.load(curr_file)
    # frame_data_kt = test_data_kt['data']
    # index = ['a','l','p','x','y',]
    # full_data_array_kt = get_full_data_array(frame_data_kt, index)  # 得到数组
    # matplotlib_backward(full_data_array_kt)


    # path_pd = '../data/raw_data/PD/PD23/PD-23-20190506-071210-ptrace_dda09d7c-7ca1-48cb-b163-9f837a2c33cf.json'
    # with open(path_pd) as curr_file:
    #     test_data_pd = json.load(curr_file)
    # frame_data_pd = test_data_pd['data']
    # index = ['x']
    # full_data_array_pd = get_full_data_array(frame_data_pd, index)  # 得到数组
    #
    # matplotlib_competation(full_data_array_kt, full_data_array_pd)



