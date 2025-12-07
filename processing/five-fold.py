# 划分五份，按比例8:2.
import os
import random
import shutil

def data_split(full_list, ratio, shuffle=False):

    n_total = len(full_list)

    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[int(0*ratio*n_total) : int(1*ratio*n_total)]
    sublist_2 = full_list[int(1*ratio*n_total) : int(2*ratio*n_total)]
    sublist_3 = full_list[int(2*ratio*n_total) : int(3*ratio*n_total)]
    sublist_4 = full_list[int(3*ratio*n_total) : int(4*ratio*n_total)]
    sublist_5 = full_list[int(4*ratio*n_total) : int(5*ratio*n_total)]

    return sublist_1, sublist_2, sublist_3, sublist_4, sublist_5


def sub_data_save(train_HC_sublist, train_PD_sublist, test_HC_sublist, test_PD_sublist, out_train_path, out_test_path, path):
    if os.path.exists(out_train_path):
        shutil.rmtree(out_train_path)
        os.mkdir(out_train_path)
        os.mkdir(os.path.join(out_train_path, 'HC'))
        os.mkdir(os.path.join(out_train_path, 'PD'))
    else:
        os.mkdir(out_train_path)
        os.mkdir(os.path.join(out_train_path, 'HC'))
        os.mkdir(os.path.join(out_train_path, 'PD'))
    if os.path.exists(out_test_path):
        shutil.rmtree(out_test_path)
        os.mkdir(out_test_path)
        os.mkdir(os.path.join(out_test_path, 'HC'))
        os.mkdir(os.path.join(out_test_path, 'PD'))
    else:
        os.mkdir(out_test_path)
        os.mkdir(os.path.join(out_test_path, 'HC'))
        os.mkdir(os.path.join(out_test_path, 'PD'))

    for file_name in train_HC_sublist:
        os.mkdir(os.path.join(out_train_path, 'HC', file_name))
        for svc_name in os.listdir(os.path.join(path, 'original_dataset', 'HC', file_name)):
            shutil.copy(os.path.join(path, 'original_dataset', 'HC', file_name, svc_name),
                        os.path.join(out_train_path, 'HC', file_name, svc_name))
    for file_name in train_PD_sublist:
        os.mkdir(os.path.join(out_train_path, 'PD', file_name))
        for svc_name in os.listdir(os.path.join(path, 'original_dataset', 'PD', file_name)):
            shutil.copy(os.path.join(path, 'original_dataset', 'PD', file_name, svc_name),
                        os.path.join(out_train_path, 'PD', file_name, svc_name))

    for file_name in test_HC_sublist:
        os.mkdir(os.path.join(out_test_path, 'HC', file_name))
        for svc_name in os.listdir(os.path.join(path, 'original_dataset', 'HC', file_name)):
            shutil.copy(os.path.join(path, 'original_dataset', 'HC', file_name, svc_name),
                        os.path.join(out_test_path, 'HC', file_name, svc_name))
    for file_name in test_PD_sublist:
        os.mkdir(os.path.join(out_test_path, 'PD', file_name))
        for svc_name in os.listdir(os.path.join(path, 'original_dataset', 'PD', file_name)):
            shutil.copy(os.path.join(path, 'original_dataset', 'PD', file_name, svc_name),
                        os.path.join(out_test_path, 'PD', file_name, svc_name))


if __name__ == "__main__":

    path = '../data'
    ratio = 0.2

    HC_files = os.listdir(os.path.join(path, 'original_dataset', 'HC'))
    PD_files = os.listdir(os.path.join(path, 'original_dataset', 'PD'))
    print('*****All:')
    print(HC_files)
    print(PD_files)
    HC_sublist_1, HC_sublist_2, HC_sublist_3, HC_sublist_4, HC_sublist_5 = data_split(HC_files, ratio, shuffle=True)
    PD_sublist_1, PD_sublist_2, PD_sublist_3, PD_sublist_4, PD_sublist_5 = data_split(PD_files, ratio, shuffle=True)

    train_HC_sublist_1 = HC_sublist_1 + HC_sublist_2 + HC_sublist_3 + HC_sublist_4
    train_PD_sublist_1 = PD_sublist_1 + PD_sublist_2 + PD_sublist_3 + PD_sublist_4
    test_HC_sublist_1 = HC_sublist_5
    test_PD_sublist_1 = PD_sublist_5
    out_train_1_path = os.path.join(path, 'raw_data_1')
    out_test_1_path = os.path.join(path, 'test_data_1')
    sub_data_save(train_HC_sublist_1, train_PD_sublist_1, test_HC_sublist_1, test_PD_sublist_1, out_train_1_path, out_test_1_path, path)
    print('*****sub_1:')
    print(train_HC_sublist_1)
    print(train_PD_sublist_1)
    print(test_HC_sublist_1)
    print(test_PD_sublist_1)



    train_HC_sublist_2 = HC_sublist_1 + HC_sublist_2 + HC_sublist_3 + HC_sublist_5
    train_PD_sublist_2 = PD_sublist_1 + PD_sublist_2 + PD_sublist_3 + PD_sublist_5
    test_HC_sublist_2 = HC_sublist_4
    test_PD_sublist_2 = PD_sublist_4
    out_train_2_path = os.path.join(path, 'raw_data_2')
    out_test_2_path = os.path.join(path, 'test_data_2')
    sub_data_save(train_HC_sublist_2, train_PD_sublist_2, test_HC_sublist_2, test_PD_sublist_2, out_train_2_path, out_test_2_path, path)
    print('*****sub_2:')
    print(train_HC_sublist_2)
    print(train_PD_sublist_2)
    print(test_HC_sublist_2)
    print(test_PD_sublist_2)

    #
    train_HC_sublist_3 = HC_sublist_1 + HC_sublist_2 + HC_sublist_4 + HC_sublist_5
    train_PD_sublist_3 = PD_sublist_1 + PD_sublist_2 + PD_sublist_4 + PD_sublist_5
    test_HC_sublist_3 = HC_sublist_3
    test_PD_sublist_3 = PD_sublist_3
    out_train_3_path = os.path.join(path, 'raw_data_3')
    out_test_3_path = os.path.join(path, 'test_data_3')
    sub_data_save(train_HC_sublist_3, train_PD_sublist_3, test_HC_sublist_3, test_PD_sublist_3, out_train_3_path, out_test_3_path, path)
    print('*****sub_3:')
    print(train_HC_sublist_3)
    print(train_PD_sublist_3)
    print(test_HC_sublist_3)
    print(test_PD_sublist_3)

    #
    train_HC_sublist_4 = HC_sublist_1 + HC_sublist_3 + HC_sublist_4 + HC_sublist_5
    train_PD_sublist_4 = PD_sublist_1 + PD_sublist_3 + PD_sublist_4 + PD_sublist_5
    test_HC_sublist_4 = HC_sublist_2
    test_PD_sublist_4 = PD_sublist_2
    out_train_4_path = os.path.join(path, 'raw_data_4')
    out_test_4_path = os.path.join(path, 'test_data_4')
    sub_data_save(train_HC_sublist_4, train_PD_sublist_4, test_HC_sublist_4, test_PD_sublist_4, out_train_4_path,
                  out_test_4_path, path)
    print('*****sub_4:')
    print(train_HC_sublist_4)
    print(train_PD_sublist_4)
    print(test_HC_sublist_4)
    print(test_PD_sublist_4)

    #
    train_HC_sublist_5 = HC_sublist_2 + HC_sublist_3 + HC_sublist_4 + HC_sublist_5
    train_PD_sublist_5 = PD_sublist_2 + PD_sublist_3 + PD_sublist_4 + PD_sublist_5
    test_HC_sublist_5 = HC_sublist_1
    test_PD_sublist_5 = PD_sublist_1
    out_train_5_path = os.path.join(path, 'raw_data_5')
    out_test_5_path = os.path.join(path, 'test_data_5')
    sub_data_save(train_HC_sublist_5, train_PD_sublist_5, test_HC_sublist_5, test_PD_sublist_5, out_train_5_path,
                  out_test_5_path, path)
    print('*****sub_5:')
    print(train_HC_sublist_5)
    print(train_PD_sublist_5)
    print(test_HC_sublist_5)
    print(test_PD_sublist_5)





