'''
采用matplotlib内置函数画图.
Drawing with built-in functions of matplotlib

KT115-ptrace
PD23-ptrace
'''

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
import PIL.Image as Image
import re

from data_utils.utils import *

def matplotlib_image_from_sequence(data, data_range, re_size, file_name):

    nalp = normalization_array_data(data[:, 0:3], data_range[:, 0:3])  # 对a,l,p三个特征进行最大最小归一化处理

    X = np.array(data[:, 3])
    Y = np.array(data[:, 4])
    colors = nalp
    press = re_size * nalp[:,2]
    # plt.scatter(X, -Y, s=press, c=colors)
    # plt.scatter(X, -Y, s=press)

    # plt.figure(figsize=(18, 3))
    plt.plot(X, Y,  linewidth=3.0)
    plt.axis('off')
    plt.title(file_name)

    # canvas = FigureCanvasAgg(plt.gcf())  # 将plt转化为numpy数据
    # canvas.draw()
    # w, h = canvas.get_width_height()
    # buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)  # 解码string 得到argb图像
    #
    # buf.shape = (w, h, 4)  # 重构成w h 4(argb)图像
    # buf = np.roll(buf, 3, axis=2) # 转换为 RGBA
    # image = Image.frombytes("RGBA", (w, h), buf.tostring())  # 得到 Image RGBA图像对象
    # image = np.asarray(image)
    # rgb_image = image[:, :, :3]

    plt.show()
    # plt.savefig("D:/Projects/Papers/Parkinson/figure1.png", dpi=500, bbox_inches='tight')


def signal_decomposition(data1, data_range):
    X = np.array(data1[:, 3])[3250:3500]
    Y = np.array(data1[:, 4])[3250:3500]
    X = X - np.min(X)
    Y = 1-Y - np.min(1-Y)



    x_range = (np.ceil(np.max(X) - np.min(X))) / 20
    y_range = (np.ceil(np.max(Y) - np.min(Y))) / 20

    # plt.figure(figsize=(x_range, y_range))
    # plt.tick_params(labelsize=20)
    # plt.plot(X, Y, color='#006400', linewidth=1.0, label='PD')
    # plt.plot(X, color='k', linewidth=5.0)
    # plt.plot(-Y, color='k', linewidth=5.0)
    # plt.axis('off')
    # plt.xlabel('x',fontsize=20)

    # plt.ylabel('y',fontsize=20)
    plt.ylim(5, 10)
    plt.xlim(260, 300)

    # plt.legend(fontsize=20)
    plt.show()

if __name__=='__main__':

    '''empty data'''
    # ./data/test_data/KT/KT21/KT-21_F4675D6C_plcontinue_2018-04-10_12_06_18___9f7cb18fc2874450940412332a83982c.json,
    # ./data/test_data/PD/PD15/PD-15_2CEE8303_plcopy_2018-03-05_08_48_43___b2501ec53c354b45a5fd56e064a35c13.json

    '''incorrect data'''
    # ./data/test_data/PD/PD15/PD-15_2CEE8303_plcontinue_2018-03-05_08_46_16___86cb047f1ed240cda2496a92c02615c0.json

    '''seperation data'''
    # ../data/raw_data/kt/KT23/KT-23_96EC7657_pcontinue_2018-04-10_14_25_50___02947de59b4c4e4da9b37a93bca9340e.json

    # list_files = {
    #     # '../data/test_data/PD/PD15/PD-15_2CEE8303_plcontinue_2018-03-05_08_46_16___86cb047f1ed240cda2496a92c02615c0.json',
    #     # '../data/test_data/KT/KT21/KT-21_F4675D6C_plcontinue_2018-04-10_12_06_18___9f7cb18fc2874450940412332a83982c.json',
    #     # '../data/test_data/PD/PD9/PD-09_FB1C35EA_pcontinue_2017-12-12_22_07_35___4e95c808a64b4744b5408746f79cd3cd.json',
    #     '../data/raw_data/kt/KT23/KT-23_96EC7657_pcontinue_2018-04-10_14_25_50___02947de59b4c4e4da9b37a93bca9340e.json',
    #     # '../data/test_data/PD/PD15/PD-15_2CEE8303_plcopy_2018-03-05_08_48_43___b2501ec53c354b45a5fd56e064a35c13.json'
    #
    # }
    # for f, full_file_name in enumerate(list_files):
    #     with open(full_file_name) as curr_file:
    #         test_data = json.load(curr_file)
    #
    #     frame_data = test_data['data']
    #     index = ['a', 'l', 'p', 'x', 'y']
    #     full_data_array = get_full_data_array(frame_data, index)  # 得到数组
    #     data_scale = get_column_data_scale(full_data_array)  # 得到每一个特征的最值范围
    #     image = matplotlib_image_from_sequence(full_data_array, data_scale, re_size=5, file_name=full_file_name)

    data_path = '../data/raw_data/'

    pattern_lists = {'pcontinue', 'pcopy', 'ptrace'}
    # pattern_lists = {'clock'}
    # pattern_lists = {'digits'}
    # pattern_lists = {'lines'}
    # pattern_lists = {'pcontinue'}
    # pattern_lists = {'pcopy'}
    # pattern_lists = {'ptrace'}
    # pattern_lists = {'plcontinue'}
    #
    # pattern_lists = {'plcopy'}
    # pattern_lists = {'pltrace'}
    # pattern_lists = {'sentence'}
    # pattern_lists = {'poppelreuter'}
    # pattern_lists = {'draw'}
    # pattern_lists = {'spiral'}
    dataset_files = os.listdir(data_path) # ./data/raw_data
    for l, label_path in enumerate(dataset_files):
        test_files = os.listdir(os.path.join(data_path, label_path))  # ./data/raw_data/KT
        for t, test_path in enumerate(test_files):
            files = os.listdir(os.path.join(data_path, label_path, test_path))  # ./data/raw_data/KT/KT1
            for f, file_path in enumerate(files):
                for pattern in pattern_lists:
                    if re.search(pattern, file_path):
                        json_file_name = data_path + '/' + label_path + '/' + test_path + '/' + file_path  # ./data/raw_data/KT/KT1/KT-01_8D6834FB_plcontinue_2017-11-05_16_45_13___d2f076a85a38476a99cec299d2c17419.json
                        print(json_file_name)

                        with open(json_file_name) as curr_file:
                            test_data = json.load(curr_file)
                            frame_data = test_data['data']
                            index = ['a', 'l', 'p', 'x', 'y']
                            full_data_array = get_full_data_array(frame_data, index)
                            print('  %d'%(len(full_data_array[:,0])))
                            # data_scale = get_column_data_scale(full_data_array)
                            #
                            # if (np.count_nonzero(data_scale[0, :] - data_scale[1, :]) == 5):
                            #     full_data_array = normalization_array_data(full_data_array, data_scale)
                            #     if not np.isnan(np.sum(full_data_array)):
                            #         image = matplotlib_image_from_sequence(full_data_array, data_scale, re_size=20,
                            #                                                file_name=json_file_name)
                            #
                            #     else:
                            #         print('The data has Nan value (%s)!' % (json_file_name))
                            # else:
                            #     print('There are some eigenvalues in the data that do not change (%s).' % (
                            #         json_file_name))
