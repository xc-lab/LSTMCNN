import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
def normalization_array_data(data, data_range):
    normalized_data = data.copy()
    for i in np.arange(data.shape[1]):
        min_value, max_value = data_range[:, i]
        normalized_data[:,i] = (normalized_data[:,i]-min_value)/(max_value-min_value)
    return normalized_data


def matplotlib_image_from_sequence(data_KT, data_PD ):
    a = 63
    b = 89
    c = 71
    d = 95
    X_KT = np.array(data_KT[:, 3])
    Y_KT = np.array(data_KT[:, 4])

    X_PD = np.array(data_PD[:, 3])
    Y_PD = np.array(data_PD[:, 4])
    plt.figure(figsize=(8, 8))
    plt.plot(X_KT[:], -Y_KT[:], c='#266A2E', linewidth=3.0)
    plt.plot(X_PD[:], -Y_PD[:], c='#f07818', linewidth=3.0)

    plt.axis('off')
    plt.show()
    # plt.savefig('../P-PL-spiral.png', dpi=1000)



if __name__=='__main__':
    # path_kt = '../data/raw_data/KT/KT115/20190523-171459-KT115-ptrace_7e02ddc3-9be8-4570-bb28-78f28c86cb95.json'
    # path_kt = '../data/raw_data/KT/KT115/20190523-171413-KT115-pltrace_21ec5036-a1be-4a21-bb29-484263068af3.json'
    path_kt = '../data/raw_data/KT/KT115/20190523-171011-KT115-spiral_ef251022-7605-4430-8d16-83961cdb055a.json'

    with open(path_kt) as curr_file:
        test_data_kt = json.load(curr_file)
    frame_data_kt = test_data_kt['data']
    index = ['a', 'l', 'p', 'x', 'y', 't']
    full_data_array_kt = get_full_data_array(frame_data_kt, index)  # 得到数组
    data_scale_kt = get_column_data_scale(full_data_array_kt)  #

    # path_pd = '../data/raw_data/PD/PD23/PD-23-20190506-071210-ptrace_dda09d7c-7ca1-48cb-b163-9f837a2c33cf.json'
    # path_pd = '../data/raw_data/PD/PD23/PD-23-20190506-071134-pltrace_68ef7089-9d2b-4404-a500-aa9d9d366542.json'
    path_pd = '../data/raw_data/PD/PD23/PD-23-20190506-070733-spiral_842ddf5d-d7b9-46c2-9229-4e0867c1bbff.json'

    with open(path_pd) as curr_file:
        test_data_pd = json.load(curr_file)
    frame_data_pd = test_data_pd['data']
    index = ['a', 'l', 'p', 'x', 'y', 't']
    full_data_array_pd = get_full_data_array(frame_data_pd, index)  # 得到数组
    data_scale_pd = get_column_data_scale(full_data_array_pd)  #

    matplotlib_image_from_sequence(full_data_array_kt, full_data_array_pd)
