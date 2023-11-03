import numpy as np
import os
import matplotlib.pyplot as plt
from estimation import Performance
sub_map = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')





def result_to_threshold(acc_dataset, true_labels):
    '''
    画出各个指标随着阈值变化的曲线
    :param acc_dataset:
    :param true_labels:
    :param out_path:
    :return:
    '''

    acc_list = []
    f1_list = []
    recall_list = []
    precision_list = []
    specificity_list = []
    npv_list = []
    mcc_list = []

    threshold_list = np.arange(0, 1, 0.001)

    for threshold in threshold_list:
        pred_labels = []
        for index in np.arange(len(acc_dataset)):
            acc = acc_dataset[index]
            true_label = true_labels[index]
            if acc >= threshold:
                pred_labels.append(true_label)
            else:
                pred_labels.append(1 - true_label)

        metric = Performance(true_labels, pred_labels)
        acc_list.append(metric.accuracy())
        f1_list.append(metric.f1_score())
        recall_list.append(metric.recall())
        precision_list.append(metric.presision())
        specificity_list.append(metric.specificity())
        npv_list.append(metric.npv())
        mcc_list.append(metric.mcc())

    return acc_list, f1_list, recall_list, precision_list, specificity_list, npv_list, mcc_list



if __name__ == '__main__':
    file_path = './outputs/'

    true_label_p = np.loadtxt(os.path.join(file_path, 'tru_label_p.txt'))

    pre_acc_p_1 = np.loadtxt(os.path.join(file_path, 'pre_acc_p_1.txt'))
    acc_list_p_1, f1_list_p_1, recall_list_p_1, precision_list_p_1, specificity_list_p_1, npv_list_p_1, mcc_list_p_1 = result_to_threshold(pre_acc_p_1, true_label_p)
    pre_acc_p_2 = np.loadtxt(os.path.join(file_path, 'pre_acc_p_2.txt'))
    acc_list_p_2, f1_list_p_2, recall_list_p_2, precision_list_p_2, specificity_list_p_2, npv_list_p_2, mcc_list_p_2 = result_to_threshold(pre_acc_p_2, true_label_p)
    pre_acc_p_3 = np.loadtxt(os.path.join(file_path, 'pre_acc_p_3.txt'))
    acc_list_p_3, f1_list_p_3, recall_list_p_3, precision_list_p_3, specificity_list_p_3, npv_list_p_3, mcc_list_p_3 = result_to_threshold(pre_acc_p_3, true_label_p)
    pre_acc_p_4 = np.loadtxt(os.path.join(file_path, 'pre_acc_p_4.txt'))
    acc_list_p_4, f1_list_p_4, recall_list_p_4, precision_list_p_4, specificity_list_p_4, npv_list_p_4, mcc_list_p_4 = result_to_threshold(pre_acc_p_4, true_label_p)
    pre_acc_p_5 = np.loadtxt(os.path.join(file_path, 'pre_acc_p_5.txt'))
    acc_list_p_5, f1_list_p_5, recall_list_p_5, precision_list_p_5, specificity_list_p_5, npv_list_p_5, mcc_list_p_5 = result_to_threshold(pre_acc_p_5, true_label_p)

    acc_p = np.concatenate((np.array(acc_list_p_1).reshape(-1,1), np.array(acc_list_p_2).reshape(-1,1), np.array(acc_list_p_3).reshape(-1,1), np.array(acc_list_p_4).reshape(-1,1), np.array(acc_list_p_5).reshape(-1,1)),axis=1)
    f1_p = np.concatenate((np.array(f1_list_p_1).reshape(-1,1), np.array(f1_list_p_2).reshape(-1,1), np.array(f1_list_p_3).reshape(-1,1), np.array(f1_list_p_4).reshape(-1,1), np.array(f1_list_p_5).reshape(-1,1)),axis=1)
    recall_p = np.concatenate((np.array(recall_list_p_1).reshape(-1,1), np.array(recall_list_p_2).reshape(-1,1), np.array(recall_list_p_3).reshape(-1,1), np.array(recall_list_p_4).reshape(-1,1), np.array(recall_list_p_5).reshape(-1,1)),axis=1)
    precision_p = np.concatenate((np.array(precision_list_p_1).reshape(-1,1), np.array(precision_list_p_2).reshape(-1,1), np.array(precision_list_p_3).reshape(-1,1), np.array(precision_list_p_4).reshape(-1,1), np.array(precision_list_p_5).reshape(-1,1)),axis=1)
    specificity_p = np.concatenate((np.array(specificity_list_p_1).reshape(-1,1), np.array(specificity_list_p_2).reshape(-1,1), np.array(specificity_list_p_3).reshape(-1,1), np.array(specificity_list_p_4).reshape(-1,1), np.array(specificity_list_p_5).reshape(-1,1)),axis=1)
    npv_p = np.concatenate((np.array(npv_list_p_1).reshape(-1,1), np.array(npv_list_p_2).reshape(-1,1), np.array(npv_list_p_3).reshape(-1,1), np.array(npv_list_p_4).reshape(-1,1), np.array(npv_list_p_5).reshape(-1,1)),axis=1)
    mcc_p = np.concatenate((np.array(mcc_list_p_1).reshape(-1,1), np.array(mcc_list_p_2).reshape(-1,1), np.array(mcc_list_p_3).reshape(-1,1), np.array(mcc_list_p_4).reshape(-1,1), np.array(mcc_list_p_5).reshape(-1,1)),axis=1)

    acc_p_mean = np.mean(acc_p, axis=1)
    acc_p_min = np.min(acc_p, axis=1)
    acc_p_max = np.max(acc_p, axis=1)

    f1_p_mean = np.mean(f1_p, axis=1)
    f1_p_min = np.min(f1_p, axis=1)
    f1_p_max = np.max(f1_p, axis=1)

    recall_p_mean = np.mean(recall_p, axis=1)
    recall_p_min = np.min(recall_p, axis=1)
    recall_p_max = np.max(recall_p, axis=1)

    precision_p_mean = np.mean(precision_p, axis=1)
    precision_p_min = np.min(precision_p, axis=1)
    precision_p_max = np.max(precision_p, axis=1)

    specificity_p_mean = np.mean(specificity_p, axis=1)
    specificity_p_min = np.min(specificity_p, axis=1)
    specificity_p_max = np.max(specificity_p, axis=1)

    npv_p_mean = np.mean(npv_p, axis=1)
    npv_p_min = np.min(npv_p, axis=1)
    npv_p_max = np.max(npv_p, axis=1)

    mcc_p_mean = np.mean(mcc_p, axis=1)
    mcc_p_min = np.min(mcc_p, axis=1)
    mcc_p_max = np.max(mcc_p, axis=1)



    true_label_pl= np.loadtxt(os.path.join(file_path, 'tru_label_pl.txt'))

    pre_acc_pl_1 = np.loadtxt(os.path.join(file_path, 'pre_acc_pl_1.txt'))
    acc_list_pl_1, f1_list_pl_1, recall_list_pl_1, precision_list_pl_1, specificity_list_pl_1, npv_list_pl_1, mcc_list_pl_1 = result_to_threshold(
        pre_acc_pl_1, true_label_pl)
    pre_acc_pl_2 = np.loadtxt(os.path.join(file_path, 'pre_acc_pl_2.txt'))
    acc_list_pl_2, f1_list_pl_2, recall_list_pl_2, precision_list_pl_2, specificity_list_pl_2, npv_list_pl_2, mcc_list_pl_2 = result_to_threshold(
        pre_acc_pl_2, true_label_pl)
    pre_acc_pl_3 = np.loadtxt(os.path.join(file_path, 'pre_acc_pl_3.txt'))
    acc_list_pl_3, f1_list_pl_3, recall_list_pl_3, precision_list_pl_3, specificity_list_pl_3, npv_list_pl_3, mcc_list_pl_3 = result_to_threshold(
        pre_acc_pl_3, true_label_pl)
    pre_acc_pl_4 = np.loadtxt(os.path.join(file_path, 'pre_acc_pl_4.txt'))
    acc_list_pl_4, f1_list_pl_4, recall_list_pl_4, precision_list_pl_4, specificity_list_pl_4, npv_list_pl_4, mcc_list_pl_4 = result_to_threshold(
        pre_acc_pl_4, true_label_pl)
    pre_acc_pl_5 = np.loadtxt(os.path.join(file_path, 'pre_acc_pl_5.txt'))
    acc_list_pl_5, f1_list_pl_5, recall_list_pl_5, precision_list_pl_5, specificity_list_pl_5, npv_list_pl_5, mcc_list_pl_5 = result_to_threshold(
        pre_acc_pl_5, true_label_pl)

    acc_pl = np.concatenate((np.array(acc_list_pl_1).reshape(-1, 1), np.array(acc_list_pl_2).reshape(-1, 1),
                            np.array(acc_list_pl_3).reshape(-1, 1), np.array(acc_list_pl_4).reshape(-1, 1),
                            np.array(acc_list_pl_5).reshape(-1, 1)), axis=1)
    f1_pl = np.concatenate((np.array(f1_list_pl_1).reshape(-1, 1), np.array(f1_list_pl_2).reshape(-1, 1),
                           np.array(f1_list_pl_3).reshape(-1, 1), np.array(f1_list_pl_4).reshape(-1, 1),
                           np.array(f1_list_pl_5).reshape(-1, 1)), axis=1)
    recall_pl = np.concatenate((np.array(recall_list_pl_1).reshape(-1, 1), np.array(recall_list_pl_2).reshape(-1, 1),
                               np.array(recall_list_pl_3).reshape(-1, 1), np.array(recall_list_pl_4).reshape(-1, 1),
                               np.array(recall_list_pl_5).reshape(-1, 1)), axis=1)
    precision_pl = np.concatenate((np.array(precision_list_pl_1).reshape(-1, 1),
                                  np.array(precision_list_pl_2).reshape(-1, 1),
                                  np.array(precision_list_pl_3).reshape(-1, 1),
                                  np.array(precision_list_pl_4).reshape(-1, 1),
                                  np.array(precision_list_pl_5).reshape(-1, 1)), axis=1)
    specificity_pl = np.concatenate((np.array(specificity_list_pl_1).reshape(-1, 1),
                                    np.array(specificity_list_pl_2).reshape(-1, 1),
                                    np.array(specificity_list_pl_3).reshape(-1, 1),
                                    np.array(specificity_list_pl_4).reshape(-1, 1),
                                    np.array(specificity_list_pl_5).reshape(-1, 1)), axis=1)
    npv_pl = np.concatenate((np.array(npv_list_pl_1).reshape(-1, 1), np.array(npv_list_pl_2).reshape(-1, 1),
                            np.array(npv_list_pl_3).reshape(-1, 1), np.array(npv_list_pl_4).reshape(-1, 1),
                            np.array(npv_list_pl_5).reshape(-1, 1)), axis=1)
    mcc_pl = np.concatenate((np.array(mcc_list_pl_1).reshape(-1, 1), np.array(mcc_list_pl_2).reshape(-1, 1),
                            np.array(mcc_list_pl_3).reshape(-1, 1), np.array(mcc_list_pl_4).reshape(-1, 1),
                            np.array(mcc_list_pl_5).reshape(-1, 1)), axis=1)

    acc_pl_mean = np.mean(acc_pl, axis=1)
    acc_pl_min = np.min(acc_pl, axis=1)
    acc_pl_max = np.max(acc_pl, axis=1)

    f1_pl_mean = np.mean(f1_pl, axis=1)
    f1_pl_min = np.min(f1_pl, axis=1)
    f1_pl_max = np.max(f1_pl, axis=1)

    recall_pl_mean = np.mean(recall_pl, axis=1)
    recall_pl_min = np.min(recall_pl, axis=1)
    recall_pl_max = np.max(recall_pl, axis=1)

    precision_pl_mean = np.mean(precision_pl, axis=1)
    precision_pl_min = np.min(precision_pl, axis=1)
    precision_pl_max = np.max(precision_pl, axis=1)

    specificity_pl_mean = np.mean(specificity_pl, axis=1)
    specificity_pl_min = np.min(specificity_pl, axis=1)
    specificity_pl_max = np.max(specificity_pl, axis=1)

    npv_pl_mean = np.mean(npv_pl, axis=1)
    npv_pl_min = np.min(npv_pl, axis=1)
    npv_pl_max = np.max(npv_pl, axis=1)

    mcc_pl_mean = np.mean(mcc_pl, axis=1)
    mcc_pl_min = np.min(mcc_pl, axis=1)
    mcc_pl_max = np.max(mcc_pl, axis=1)


    threshold_list = np.arange(0, 1, 0.001)

    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.subplots()
    # plt.ylim(-1, 101)
    # ax.plot(threshold_list, recall_p_mean*100, color='#266A2E', lw=3)
    # ax.fill_between(threshold_list, recall_p_min*100, recall_p_max*100, alpha=0.2, color='#266A2E')
    # ax.plot(threshold_list, recall_pl_mean*100, color='#f07818', lw=3)
    # ax.fill_between(threshold_list, recall_pl_min*100, recall_pl_max*100, alpha=0.2, color='#f07818')
    # plt.tick_params(labelsize=20)
    # plt.ylabel('Recall(in %)'.translate(sub_map), fontsize=25)
    # plt.xlabel('Threshold'.translate(sub_map), fontsize=25)
    # plt.savefig("D:/Projects/Papers/Parkinson/thres_recall.png", dpi=500, bbox_inches='tight')
    # # plt.show()

    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.subplots()
    # plt.ylim(-1, 101)
    # ax.plot(threshold_list, acc_p_mean*100, color='#266A2E', lw=3)
    # ax.fill_between(threshold_list, acc_p_min*100, acc_p_max*100, alpha=0.2, color='#266A2E')
    # ax.plot(threshold_list, acc_pl_mean*100, color='#f07818', lw=3)
    # ax.fill_between(threshold_list, acc_pl_min*100, acc_pl_max*100, alpha=0.2, color='#f07818')
    # plt.tick_params(labelsize=20)
    # plt.ylabel('Accuracy(in %)'.translate(sub_map), fontsize=25)
    # plt.xlabel('Threshold'.translate(sub_map), fontsize=25)
    # plt.savefig("D:/Projects/Papers/Parkinson/thres_acc.png", dpi=500, bbox_inches='tight')
    # # plt.show()

    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.subplots()
    # plt.ylim(-1, 101)
    # ax.plot(threshold_list, f1_p_mean*100, color='#266A2E', lw=3)
    # ax.fill_between(threshold_list, f1_p_min*100, f1_p_max*100, alpha=0.2, color='#266A2E')
    # ax.plot(threshold_list, f1_pl_mean*100, color='#f07818', lw=3)
    # ax.fill_between(threshold_list, f1_pl_min*100, f1_pl_max*100, alpha=0.2, color='#f07818')
    # plt.tick_params(labelsize=20)
    # plt.ylabel('F1 score(in %)'.translate(sub_map), fontsize=25)
    # plt.xlabel('Threshold'.translate(sub_map), fontsize=25)
    # plt.savefig("D:/Projects/Papers/Parkinson/thres_f1.png", dpi=500, bbox_inches='tight')
    # # plt.show()

    fig = plt.figure(figsize=(11, 9))
    ax = fig.subplots()
    plt.ylim(-101, 101)
    ax.plot(threshold_list, mcc_p_mean*100, color='#266A2E', lw=3)
    ax.fill_between(threshold_list, mcc_p_min*100, mcc_p_max*100, alpha=0.2, color='#266A2E')
    ax.plot(threshold_list, mcc_pl_mean*100, color='#f07818', lw=3)
    ax.fill_between(threshold_list, mcc_pl_min*100, mcc_pl_max*100, alpha=0.2, color='#f07818')
    plt.tick_params(labelsize=20)
    plt.ylabel('MCC(in %)'.translate(sub_map), fontsize=25)
    plt.xlabel('Threshold'.translate(sub_map), fontsize=25)
    plt.savefig("D:/Projects/Papers/Parkinson/thres_mcc.png", dpi=500, bbox_inches='tight')
    # # plt.show()

    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.subplots()
    # plt.ylim(-1, 101)
    # ax.plot(threshold_list, specificity_p_mean*100, color='#266A2E', lw=3)
    # ax.fill_between(threshold_list, specificity_p_min*100, specificity_p_max*100, alpha=0.2, color='#266A2E')
    # ax.plot(threshold_list, specificity_pl_mean*100, color='#f07818', lw=3)
    # ax.fill_between(threshold_list, specificity_pl_min*100, specificity_pl_max*100, alpha=0.2, color='#f07818')
    # plt.tick_params(labelsize=20)
    # plt.ylabel('Specificity(%)'.translate(sub_map), fontsize=25)
    # plt.xlabel('Threshold'.translate(sub_map), fontsize=25)
    # plt.savefig("D:/Projects/Papers/Parkinson/thres_specificity.png", dpi=500, bbox_inches='tight')
    # # plt.show()

    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.subplots()
    # plt.ylim(-1, 101)
    # ax.plot(threshold_list, precision_p_mean*100, color='#266A2E', lw=3)
    # ax.fill_between(threshold_list, precision_p_min*100, precision_p_max*100, alpha=0.2, color='#266A2E')
    # ax.plot(threshold_list, precision_pl_mean*100, color='#f07818', lw=3)
    # ax.fill_between(threshold_list, precision_pl_min*100, precision_pl_max*100, alpha=0.2, color='#f07818')
    # plt.tick_params(labelsize=20)
    # plt.ylabel('Precision(%)'.translate(sub_map), fontsize=25)
    # plt.xlabel('Threshold'.translate(sub_map), fontsize=25)
    # plt.savefig("D:/Projects/Papers/Parkinson/thres_precision.png", dpi=500, bbox_inches='tight')
    # # plt.show()

