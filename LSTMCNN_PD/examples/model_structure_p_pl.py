'''
模型结构（结构和卷积核大小）对性能的影响
'''
import numpy as np
'''
P-type数据
'''
mnist_1_p = np.array([
    [0.884615, 0.842105, 0.727273, 1.000000, 0.833333, 1.000000, 0.778499, 0.8636],
    [0.884615, 0.842105, 0.727273, 1.000000, 0.833333, 1.000000, 0.778499, 0.8636],
    [0.884615, 0.842105, 0.727273, 1.000000, 0.833333, 1.000000, 0.778499, 0.8636],
    [0.884615, 0.842105, 0.727273, 1.000000, 0.833333, 1.000000, 0.778499, 0.8636],
    [0.923077, 0.916667, 1.000000, 0.846154, 1.000000, 0.866667, 0.856349, 0.9333],
])

mnist_p = np.array([
    [0.884615, 0.842105, 0.727273, 1.000000, 0.833333, 1.000000, 0.778499, 0.8636],
    [0.884615, 0.842105, 0.727273, 1.000000, 0.833333, 1.000000, 0.778499, 0.8636],
    [0.923077, 0.900000, 0.818182, 1.000000, 0.882353, 1.000000, 0.849662, 0.9091],
    [0.923077, 0.900000, 0.818182, 1.000000, 0.882353, 1.000000, 0.849662, 0.9091],
    [0.961538, 0.952381, 0.909091, 1.000000, 0.937500, 1.000000, 0.923186, 0.9545],
])

lstmcnn_5_5_p = np.array([
    [0.884615, 0.842105, 0.727273, 1.000000, 0.833333, 1.000000, 0.778499],
    [0.923077, 0.909091, 0.909091, 0.909091, 0.933333, 0.933333, 0.842424],
    [0.961538, 0.952381, 0.909091, 1.000000, 0.937500, 1.000000, 0.923186],
    [0.961538, 0.952381, 0.909091, 1.000000, 0.937500, 1.000000, 0.923186],
    [0.961538, 0.952381, 0.909091, 1.000000, 0.937500, 1.000000, 0.923186],
])

lstmcnn_5_3_p = np.array([
    [0.884615, 0.842105, 0.727273, 1.000000, 0.833333, 1.000000, 0.778499],
    [0.923077, 0.909091, 0.909091, 0.909091, 0.933333, 0.933333, 0.842424],
    [0.923077, 0.909091, 0.909091, 0.909091, 0.933333, 0.933333, 0.842424],
    [0.923077, 0.900000, 0.818182, 1.000000, 0.882353, 1.000000, 0.849662],
    [0.923077, 0.900000, 0.818182, 1.000000, 0.882353, 1.000000, 0.849662]
])

lstmcnn_3_3_p = np.array([
    [0.884615, 0.857143, 0.818182, 0.900000, 0.875000, 0.933333, 0.763167, 0.8758],
    [0.923077, 0.900000, 0.818182, 1.000000, 0.882353, 1.000000, 0.849662, 0.9091],
    [0.961538, 0.952381, 0.909091, 1.000000, 0.937500, 1.000000, 0.923186, 0.9545],
    [0.961538, 0.956522, 1.000000, 0.916667, 1.000000, 0.933333, 0.924962, 0.9667],
    [0.961538, 0.956522, 1.000000, 0.916667, 1.000000, 0.933333, 0.924962, 0.9667]
])

crnncnn_3_3_p = np.array([
    [0.884615, 0.842105, 0.727273, 1.000000, 0.833333, 1.000000, 0.778499, 0.8636],
    [0.884615, 0.842105, 0.727273, 1.000000, 0.833333, 1.000000, 0.778499, 0.8636],
    [0.923077, 0.909091, 0.909091, 0.909091, 0.933333, 0.933333, 0.842424, 0.9212],
    [0.961538, 0.952381, 0.909091, 1.000000, 0.937500, 1.000000, 0.923186, 0.9545],
    [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.0000]
])

cgrucnn_3_3_p = np.array([
    [0.884615, 0.842105, 0.727273, 1.000000, 0.833333, 1.000000, 0.778499, 0.8636],
    [0.961538, 0.952381, 0.909091, 1.000000, 0.937500, 1.000000, 0.923186, 0.9545],
    [0.923077, 0.909091, 0.909091, 0.909091, 0.933333, 0.933333, 0.842424, 0.9212],
    [0.961538, 0.952381, 0.909091, 1.000000, 0.937500, 1.000000, 0.923186, 0.9545],
    [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.0000]
])

clstmcnn_3_3_p = np.array([
    [0.923077, 0.909091, 0.909091, 0.909091, 0.933333, 0.933333, 0.842424, 0.9212],
    [0.961538, 0.952381, 0.909091, 1.000000, 0.937500, 1.000000, 0.923186, 0.9545],
    [0.961538, 0.956522, 1.000000, 0.916667, 1.000000, 0.933333, 0.924962, 0.9667],
    [0.961538, 0.952381, 0.909091, 1.000000, 0.937500, 1.000000, 0.923186, 0.9545],
    [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.0000]
])

clstmcnn_3_3_p_2 = np.array([
    [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.0000],
    [0.923077, 0.909091, 0.909091, 0.909091, 0.933333, 0.933333, 0.842424, 0.9212],
    [0.923077, 0.900000, 0.818182, 1.000000, 0.882353, 1.000000, 0.849662, 0.9091],
    [0.961538, 0.952381, 0.909091, 1.000000, 0.937500, 1.000000, 0.923186, 0.9545],
    [0.884615, 0.857143, 0.818182, 0.900000, 0.875000, 0.933333, 0.763167, 0.8758],

])

clstmcnn_5_5_p_2 = np.array([
    [0.961538, 0.952381, 0.909091, 1.000000, 0.937500, 1.000000, 0.923186],
    [0.923077, 0.916667, 1.000000, 0.846154, 1.000000, 0.866667, 0.856349],
    [0.961538, 0.952381, 0.909091, 1.000000, 0.937500, 1.000000, 0.923186],
    [0.961538, 0.956522, 1.000000, 0.916667, 1.000000, 0.933333, 0.924962],
    [0.961538, 0.956522, 1.000000, 0.916667, 1.000000, 0.933333, 0.924962]
])

'''
PL-type
'''

mnist_pl = np.array([
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114, 0.9091],
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114, 0.9091],
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114, 0.9091],
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114, 0.9091],
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132, 0.9545]
])

lstmcnn_5_3_pl = np.array([
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114],
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114],
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114],
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132],
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132],

])

lstmcnn_5_5_pl = np.array([
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132],
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114],
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114],
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114],
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114],
])

lstmcnn_3_3_pl = np.array([
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114, 0.9091],
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114, 0.9091],
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132, 0.9545],
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132, 0.9545],
    [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.0000]
])

clstmcnn_3_3_pl = np.array([
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114, 0.9091],
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132, 0.9545],
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132, 0.9545],
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132, 0.9545],
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132, 0.9545]
])

crnncnn_3_3_pl = np.array([
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114, 0.9091],
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114, 0.9091],
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114, 0.9091],
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114, 0.9091],
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132, 0.9545]
])

cgrucnn_3_3_pl = np.array([
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114, 0.9091],
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114, 0.9091],
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132, 0.9545],
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132, 0.9545],
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132, 0.9545]
])

clstmcnn_3_3_pl_2 = np.array([
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114, 0.9091],
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114, 0.9091],
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132, 0.9545],
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132, 0.9545],
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114, 0.9091],
])

clstmcnn_5_5_pl_2 = np.array([
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132],
    [0.920000, 0.909091, 0.909091, 0.909091, 0.928571, 0.928571, 0.837662],
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114],
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114],
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114],
])

mnist_1_pl = np.array([
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132, 0.9545],
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132, 0.9545],
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114, 0.9091],
    [0.920000, 0.909091, 0.909091, 0.909091, 0.928571, 0.928571, 0.837662, 0.9188],
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114, 0.9091],
])

lg_p = np.array([
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132, 0.9545],
    [0.923077, 0.916667, 1.000000, 0.846154, 1.000000, 0.866667, 0.856349, 0.9333],
    [0.923077, 0.916667, 1.000000, 0.846154, 1.000000, 0.866667, 0.856349, 0.9333],
    [0.923077, 0.916667, 1.000000, 0.846154, 1.000000, 0.866667, 0.856349, 0.9333],
    [0.923077, 0.916667, 1.000000, 0.846154, 1.000000, 0.866667, 0.856349, 0.9333],
])
lg_pl = np.array([
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132, 0.9545],
    [0.923077, 0.916667, 1.000000, 0.846154, 1.000000, 0.866667, 0.856349, 0.9333],
    [0.923077, 0.916667, 1.000000, 0.846154, 1.000000, 0.866667, 0.856349, 0.9333],
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132, 0.9545],
    [0.923077, 0.916667, 1.000000, 0.846154, 1.000000, 0.866667, 0.856349, 0.9333],
])

'''
取数据
'''
metric_dict = {'acc':0, 'f1':1, 'tpr':2, 'ppv':3, 'npv':4, 'tnr':5, 'mcc':6, 'auc':7}



# tpr_lg_p_mean = np.mean(lg_p[:,metric_dict['tpr']])
# tpr_lg_p_std = np.std(lg_p[:,metric_dict['tpr']])
# print('##### P-type lg tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_lg_p_mean, tpr_lg_p_std))
# tnr_lg_p_mean = np.mean(lg_p[:,metric_dict['tnr']])
# tnr_lg_p_std = np.std(lg_p[:,metric_dict['tnr']])
# print('      P-type lg tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_lg_p_mean, tnr_lg_p_std))

# fpr_lg_p_mean = np.mean(1 - lg_p[:,metric_dict['tnr']])
# fpr_lg_p_std = np.std(1 - lg_p[:,metric_dict['tnr']])
# print('      P-type lg fpr-mean:%.6f, fpr-std:%.6f.'%(fpr_lg_p_mean, fpr_lg_p_std))
# auc_lg_p_mean = np.mean(lg_p[:,metric_dict['auc']])
# auc_lg_p_std = np.std(lg_p[:,metric_dict['auc']])
# print('      P-type lg auc-mean:%.6f, auc-std:%.6f.'%(auc_lg_p_mean, auc_lg_p_std))


# ppv_lg_p_mean = np.mean(lg_p[:,metric_dict['ppv']])
# ppv_lg_p_std = np.std(lg_p[:,metric_dict['ppv']])
# print('      P-type lg ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_lg_p_mean, ppv_lg_p_std))
#
# npv_lg_p_mean = np.mean(lg_p[:,metric_dict['npv']])
# npv_lg_p_std = np.std(lg_p[:,metric_dict['npv']])
# print('      P-type lg npv-mean:%.6f, npv-std:%.6f.'%(npv_lg_p_mean, npv_lg_p_std))
#
# acc_lg_p_mean = np.mean(lg_p[:,metric_dict['acc']])
# acc_lg_p_std = np.std(lg_p[:,metric_dict['acc']])
# print('      P-type lg acc-mean:%.6f, acc-std:%.6f.'%(acc_lg_p_mean, acc_lg_p_std))
#
# f1_lg_p_mean = np.mean(lg_p[:,metric_dict['f1']])
# f1_lg_p_std = np.std(lg_p[:,metric_dict['f1']])
# print('      P-type lg f1-mean:%.6f, f1-std:%.6f.'%(f1_lg_p_mean, f1_lg_p_std))
#
# mcc_lg_p_mean = np.mean(lg_p[:,metric_dict['mcc']])
# mcc_lg_p_std = np.std(lg_p[:,metric_dict['mcc']])
# print('      P-type lg mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_lg_p_mean, mcc_lg_p_std))
#
#
# tpr_lg_pl_mean = np.mean(lg_pl[:,metric_dict['tpr']])
# tpr_lg_pl_std = np.std(lg_pl[:,metric_dict['tpr']])
# print('##### Pl-type lg tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_lg_pl_mean, tpr_lg_pl_std))
# tnr_lg_pl_mean = np.mean(lg_pl[:,metric_dict['tnr']])
# tnr_lg_pl_std = np.std(lg_pl[:,metric_dict['tnr']])
# print('      Pl-type lg tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_lg_pl_mean, tnr_lg_pl_std))

# fpr_lg_p_mean = np.mean(1 - lg_p[:,metric_dict['tnr']])
# fpr_lg_p_std = np.std(1 - lg_p[:,metric_dict['tnr']])
# print('      P-type lg fpr-mean:%.6f, fpr-std:%.6f.'%(fpr_lg_p_mean, fpr_lg_p_std))
# auc_lg_p_mean = np.mean(lg_p[:,metric_dict['auc']])
# auc_lg_p_std = np.std(lg_p[:,metric_dict['auc']])
# print('      P-type lg auc-mean:%.6f, auc-std:%.6f.'%(auc_lg_p_mean, auc_lg_p_std))


# ppv_lg_pl_mean = np.mean(lg_pl[:,metric_dict['ppv']])
# ppv_lg_pl_std = np.std(lg_pl[:,metric_dict['ppv']])
# print('      Pl-type lg ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_lg_pl_mean, ppv_lg_pl_std))
#
# npv_lg_pl_mean = np.mean(lg_pl[:,metric_dict['npv']])
# npv_lg_pl_std = np.std(lg_pl[:,metric_dict['npv']])
# print('      Pl-type lg npv-mean:%.6f, npv-std:%.6f.'%(npv_lg_pl_mean, npv_lg_pl_std))
#
# acc_lg_pl_mean = np.mean(lg_pl[:,metric_dict['acc']])
# acc_lg_pl_std = np.std(lg_pl[:,metric_dict['acc']])
# print('      Pl-type lg acc-mean:%.6f, acc-std:%.6f.'%(acc_lg_pl_mean, acc_lg_pl_std))
#
# f1_lg_pl_mean = np.mean(lg_pl[:,metric_dict['f1']])
# f1_lg_pl_std = np.std(lg_pl[:,metric_dict['f1']])
# print('      Pl-type lg f1-mean:%.6f, f1-std:%.6f.'%(f1_lg_pl_mean, f1_lg_pl_std))
#
# mcc_lg_pl_mean = np.mean(lg_pl[:,metric_dict['mcc']])
# mcc_lg_pl_std = np.std(lg_pl[:,metric_dict['mcc']])
# print('      Pl-type lg mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_lg_pl_mean, mcc_lg_pl_std))





auc_mnist_p_mean = np.mean(mnist_p[:,metric_dict['auc']])
auc_mnist_p_std = np.std(mnist_p[:,metric_dict['auc']])
print('      P-type mnist auc-mean:%.6f, auc-std:%.6f.'%(auc_mnist_p_mean, auc_mnist_p_std))


tpr_mnist_p_mean = np.mean(mnist_p[:,metric_dict['tpr']])
tpr_mnist_p_std = np.std(mnist_p[:,metric_dict['tpr']])
print('##### P-type mnist tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_mnist_p_mean, tpr_mnist_p_std))
tnr_mnist_p_mean = np.mean(mnist_p[:,metric_dict['tnr']])
tnr_mnist_p_std = np.std(mnist_p[:,metric_dict['tnr']])
print('      P-type mnist tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_mnist_p_mean, tnr_mnist_p_std))

# fpr_mnist_p_mean = np.mean(1 - mnist_p[:,metric_dict['tnr']])
# fpr_mnist_p_std = np.std(1 - mnist_p[:,metric_dict['tnr']])
# print('      P-type mnist fpr-mean:%.6f, fpr-std:%.6f.'%(fpr_mnist_p_mean, fpr_mnist_p_std))
# auc_mnist_p_mean = np.mean(mnist_p[:,metric_dict['auc']])
# auc_mnist_p_std = np.std(mnist_p[:,metric_dict['auc']])
# print('      P-type mnist auc-mean:%.6f, auc-std:%.6f.'%(auc_mnist_p_mean, auc_mnist_p_std))


ppv_mnist_p_mean = np.mean(mnist_p[:,metric_dict['ppv']])
ppv_mnist_p_std = np.std(mnist_p[:,metric_dict['ppv']])
print('      P-type mnist ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_mnist_p_mean, ppv_mnist_p_std))

npv_mnist_p_mean = np.mean(mnist_p[:,metric_dict['npv']])
npv_mnist_p_std = np.std(mnist_p[:,metric_dict['npv']])
print('      P-type mnist npv-mean:%.6f, npv-std:%.6f.'%(npv_mnist_p_mean, npv_mnist_p_std))

acc_mnist_p_mean = np.mean(mnist_p[:,metric_dict['acc']])
acc_mnist_p_std = np.std(mnist_p[:,metric_dict['acc']])
print('      P-type mnist acc-mean:%.6f, acc-std:%.6f.'%(acc_mnist_p_mean, acc_mnist_p_std))

f1_mnist_p_mean = np.mean(mnist_p[:,metric_dict['f1']])
f1_mnist_p_std = np.std(mnist_p[:,metric_dict['f1']])
print('      P-type mnist f1-mean:%.6f, f1-std:%.6f.'%(f1_mnist_p_mean, f1_mnist_p_std))

mcc_mnist_p_mean = np.mean(mnist_p[:,metric_dict['mcc']])
mcc_mnist_p_std = np.std(mnist_p[:,metric_dict['mcc']])
print('      P-type mnist mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_mnist_p_mean, mcc_mnist_p_std))





auc_cgrucnn_3_3_p_mean = np.mean(cgrucnn_3_3_p[:,metric_dict['auc']])
auc_cgrucnn_3_3_p_std = np.std(cgrucnn_3_3_p[:,metric_dict['auc']])
print('##### P-type grucnn auc-mean:%.6f, auc-std:%.6f.'%(auc_cgrucnn_3_3_p_mean, auc_cgrucnn_3_3_p_std))

tpr_cgrucnn_3_3_p_mean = np.mean(cgrucnn_3_3_p[:,metric_dict['tpr']])
tpr_cgrucnn_3_3_p_std = np.std(cgrucnn_3_3_p[:,metric_dict['tpr']])
print('      P-type grucnn tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_cgrucnn_3_3_p_mean, tpr_cgrucnn_3_3_p_std))
acc_cgrucnn_3_3_p_mean = np.mean(cgrucnn_3_3_p[:,metric_dict['acc']])
acc_cgrucnn_3_3_p_std = np.std(cgrucnn_3_3_p[:,metric_dict['acc']])
print('      P-type grucnn acc-mean:%.6f, acc-std:%.6f.'%(acc_cgrucnn_3_3_p_mean, acc_cgrucnn_3_3_p_std))

f1_cgrucnn_3_3_p_mean = np.mean(cgrucnn_3_3_p[:,metric_dict['f1']])
f1_cgrucnn_3_3_p_std = np.std(cgrucnn_3_3_p[:,metric_dict['f1']])
print('      P-type grucnn f1-mean:%.6f, f1-std:%.6f.'%(f1_cgrucnn_3_3_p_mean, f1_cgrucnn_3_3_p_std))

mcc_cgrucnn_3_3_p_mean = np.mean(cgrucnn_3_3_p[:,metric_dict['mcc']])
mcc_cgrucnn_3_3_p_std = np.std(cgrucnn_3_3_p[:,metric_dict['mcc']])
print('      P-type grucnn mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_cgrucnn_3_3_p_mean, mcc_cgrucnn_3_3_p_std))



auc_crnncnn_3_3_p_mean = np.mean(crnncnn_3_3_p[:,metric_dict['auc']])
auc_crnncnn_3_3_p_std = np.std(crnncnn_3_3_p[:,metric_dict['auc']])
print('##### P-type rnncnn auc-mean:%.6f, auc-std:%.6f.'%(auc_crnncnn_3_3_p_mean, auc_crnncnn_3_3_p_std))

tpr_crnncnn_3_3_p_mean = np.mean(crnncnn_3_3_p[:,metric_dict['tpr']])
tpr_crnncnn_3_3_p_std = np.std(crnncnn_3_3_p[:,metric_dict['tpr']])
print('      P-type rnncnn tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_crnncnn_3_3_p_mean, tpr_crnncnn_3_3_p_std))
acc_crnncnn_3_3_p_mean = np.mean(crnncnn_3_3_p[:,metric_dict['acc']])
acc_crnncnn_3_3_p_std = np.std(crnncnn_3_3_p[:,metric_dict['acc']])
print('      P-type rnncnn acc-mean:%.6f, acc-std:%.6f.'%(acc_crnncnn_3_3_p_mean, acc_crnncnn_3_3_p_std))

f1_crnncnn_3_3_p_mean = np.mean(crnncnn_3_3_p[:,metric_dict['f1']])
f1_crnncnn_3_3_p_std = np.std(crnncnn_3_3_p[:,metric_dict['f1']])
print('      P-type rnncnn f1-mean:%.6f, f1-std:%.6f.'%(f1_crnncnn_3_3_p_mean, f1_crnncnn_3_3_p_std))

mcc_crnncnn_3_3_p_mean = np.mean(crnncnn_3_3_p[:,metric_dict['mcc']])
mcc_crnncnn_3_3_p_std = np.std(crnncnn_3_3_p[:,metric_dict['mcc']])
print('      P-type rnncnn mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_crnncnn_3_3_p_mean, mcc_crnncnn_3_3_p_std))






tpr_mnist_1_p_mean = np.mean(mnist_1_p[:,metric_dict['tpr']])
tpr_mnist_1_p_std = np.std(mnist_1_p[:,metric_dict['tpr']])
print('##### P-type mnist1 tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_mnist_1_p_mean, tpr_mnist_1_p_std))

auc_mnist_1_p_mean = np.mean(mnist_1_p[:,metric_dict['auc']])
auc_mnist_1_p_std = np.std(mnist_1_p[:,metric_dict['auc']])
print('      P-type mnist1 auc-mean:%.6f, auc-std:%.6f.'%(auc_mnist_1_p_mean, auc_mnist_1_p_std))

tnr_mnist_1_p_mean = np.mean(mnist_1_p[:,metric_dict['tnr']])
tnr_mnist_1_p_std = np.std(mnist_1_p[:,metric_dict['tnr']])
print('      P-type mnist1 tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_mnist_1_p_mean, tnr_mnist_1_p_std))

# fpr_mnist_1_p_mean = np.mean(1 - mnist_1_p[:,metric_dict['tnr']])
# fpr_mnist_1_p_std = np.std(1 - mnist_1_p[:,metric_dict['tnr']])
# print('      P-type mnist1 fpr-mean:%.6f, fpr-std:%.6f.'%(fpr_mnist_1_p_mean, fpr_mnist_1_p_std))
# auc_mnist_1_p_mean = np.mean(mnist_1_p[:,metric_dict['auc']])
# auc_mnist_1_p_std = np.std(mnist_1_p[:,metric_dict['auc']])
# print('      P-type mnist1 auc-mean:%.6f, auc-std:%.6f.'%(auc_mnist_1_p_mean, auc_mnist_1_p_std))


ppv_mnist_1_p_mean = np.mean(mnist_1_p[:,metric_dict['ppv']])
ppv_mnist_1_p_std = np.std(mnist_1_p[:,metric_dict['ppv']])
print('      P-type mnist1 ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_mnist_1_p_mean, ppv_mnist_1_p_std))

npv_mnist_1_p_mean = np.mean(mnist_1_p[:,metric_dict['npv']])
npv_mnist_1_p_std = np.std(mnist_1_p[:,metric_dict['npv']])
print('      P-type mnist1 npv-mean:%.6f, npv-std:%.6f.'%(npv_mnist_1_p_mean, npv_mnist_1_p_std))

acc_mnist_1_p_mean = np.mean(mnist_1_p[:,metric_dict['acc']])
acc_mnist_1_p_std = np.std(mnist_1_p[:,metric_dict['acc']])
print('      P-type mnist1 acc-mean:%.6f, acc-std:%.6f.'%(acc_mnist_1_p_mean, acc_mnist_1_p_std))

f1_mnist_1_p_mean = np.mean(mnist_1_p[:,metric_dict['f1']])
f1_mnist_1_p_std = np.std(mnist_1_p[:,metric_dict['f1']])
print('      P-type mnist1 f1-mean:%.6f, f1-std:%.6f.'%(f1_mnist_1_p_mean, f1_mnist_1_p_std))

mcc_mnist_1_p_mean = np.mean(mnist_1_p[:,metric_dict['mcc']])
mcc_mnist_1_p_std = np.std(mnist_1_p[:,metric_dict['mcc']])
print('      P-type mnist1 mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_mnist_1_p_mean, mcc_mnist_1_p_std))

#
# tpr_lstmcnn_5_5_p_mean = np.mean(lstmcnn_5_5_p[:,metric_dict['tpr']])
# tpr_lstmcnn_5_5_p_std = np.std(lstmcnn_5_5_p[:,metric_dict['tpr']])
# print('##### P-type lstmcnn_5_5 tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_lstmcnn_5_5_p_mean, tpr_lstmcnn_5_5_p_std))
#
# tnr_lstmcnn_5_5_p_mean = np.mean(lstmcnn_5_5_p[:,metric_dict['tnr']])
# tnr_lstmcnn_5_5_p_std = np.std(lstmcnn_5_5_p[:,metric_dict['tnr']])
# print('      P-type lstmcnn_5_5 tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_lstmcnn_5_5_p_mean, tnr_lstmcnn_5_5_p_std))
#
# ppv_lstmcnn_5_5_p_mean = np.mean(lstmcnn_5_5_p[:,metric_dict['ppv']])
# ppv_lstmcnn_5_5_p_std = np.std(lstmcnn_5_5_p[:,metric_dict['ppv']])
# print('      P-type lstmcnn_5_5 ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_lstmcnn_5_5_p_mean, ppv_lstmcnn_5_5_p_std))
#
# npv_lstmcnn_5_5_p_mean = np.mean(lstmcnn_5_5_p[:,metric_dict['npv']])
# npv_lstmcnn_5_5_p_std = np.std(lstmcnn_5_5_p[:,metric_dict['npv']])
# print('      P-type lstmcnn_5_5 npv-mean:%.6f, npv-std:%.6f.'%(npv_lstmcnn_5_5_p_mean, npv_lstmcnn_5_5_p_std))
#
# acc_lstmcnn_5_5_p_mean = np.mean(lstmcnn_5_5_p[:,metric_dict['acc']])
# acc_lstmcnn_5_5_p_std = np.std(lstmcnn_5_5_p[:,metric_dict['acc']])
# print('      P-type lstmcnn_5_5 acc-mean:%.6f, acc-std:%.6f.'%(acc_lstmcnn_5_5_p_mean, acc_lstmcnn_5_5_p_std))
#
# f1_lstmcnn_5_5_p_mean = np.mean(lstmcnn_5_5_p[:,metric_dict['f1']])
# f1_lstmcnn_5_5_p_std = np.std(lstmcnn_5_5_p[:,metric_dict['f1']])
# print('      P-type lstmcnn_5_5 f1-mean:%.6f, f1-std:%.6f.'%(f1_lstmcnn_5_5_p_mean, f1_lstmcnn_5_5_p_std))
#
# mcc_lstmcnn_5_5_p_mean = np.mean(lstmcnn_5_5_p[:,metric_dict['mcc']])
# mcc_lstmcnn_5_5_p_std = np.std(lstmcnn_5_5_p[:,metric_dict['mcc']])
# print('      P-type lstmcnn_5_5 mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_lstmcnn_5_5_p_mean, mcc_lstmcnn_5_5_p_std))
#
#
# tpr_lstmcnn_5_3_p_mean = np.mean(lstmcnn_5_3_p[:,metric_dict['tpr']])
# tpr_lstmcnn_5_3_p_std = np.std(lstmcnn_5_3_p[:,metric_dict['tpr']])
# print('##### P-type lstmcnn_5_3 tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_lstmcnn_5_3_p_mean, tpr_lstmcnn_5_3_p_std))
#
# tnr_lstmcnn_5_3_p_mean = np.mean(lstmcnn_5_3_p[:,metric_dict['tnr']])
# tnr_lstmcnn_5_3_p_std = np.std(lstmcnn_5_3_p[:,metric_dict['tnr']])
# print('      P-type lstmcnn_5_3 tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_lstmcnn_5_3_p_mean, tnr_lstmcnn_5_3_p_std))
#
# ppv_lstmcnn_5_3_p_mean = np.mean(lstmcnn_5_3_p[:,metric_dict['ppv']])
# ppv_lstmcnn_5_3_p_std = np.std(lstmcnn_5_3_p[:,metric_dict['ppv']])
# print('      P-type lstmcnn_5_3 ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_lstmcnn_5_3_p_mean, ppv_lstmcnn_5_3_p_std))
#
# npv_lstmcnn_5_3_p_mean = np.mean(lstmcnn_5_3_p[:,metric_dict['npv']])
# npv_lstmcnn_5_3_p_std = np.std(lstmcnn_5_3_p[:,metric_dict['npv']])
# print('      P-type lstmcnn_5_3 npv-mean:%.6f, npv-std:%.6f.'%(npv_lstmcnn_5_3_p_mean, npv_lstmcnn_5_3_p_std))
#
# acc_lstmcnn_5_3_p_mean = np.mean(lstmcnn_5_3_p[:,metric_dict['acc']])
# acc_lstmcnn_5_3_p_std = np.std(lstmcnn_5_3_p[:,metric_dict['acc']])
# print('      P-type lstmcnn_5_3 acc-mean:%.6f, acc-std:%.6f.'%(acc_lstmcnn_5_3_p_mean, acc_lstmcnn_5_3_p_std))
#
# f1_lstmcnn_5_3_p_mean = np.mean(lstmcnn_5_3_p[:,metric_dict['f1']])
# f1_lstmcnn_5_3_p_std = np.std(lstmcnn_5_3_p[:,metric_dict['f1']])
# print('      P-type lstmcnn_5_3 f1-mean:%.6f, f1-std:%.6f.'%(f1_lstmcnn_5_3_p_mean, f1_lstmcnn_5_3_p_std))
#
# mcc_lstmcnn_5_3_p_mean = np.mean(lstmcnn_5_3_p[:,metric_dict['mcc']])
# mcc_lstmcnn_5_3_p_std = np.std(lstmcnn_5_3_p[:,metric_dict['mcc']])
# print('      P-type lstmcnn_5_3 mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_lstmcnn_5_3_p_mean, mcc_lstmcnn_5_3_p_std))
#
tpr_lstmcnn_3_3_p_mean = np.mean(lstmcnn_3_3_p[:,metric_dict['tpr']])
tpr_lstmcnn_3_3_p_std = np.std(lstmcnn_3_3_p[:,metric_dict['tpr']])
print('##### P-type lstmcnn_3_3 tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_lstmcnn_3_3_p_mean, tpr_lstmcnn_3_3_p_std))

# fpr_lstmcnn_3_3_p_mean = np.mean(1 - lstmcnn_3_3_p[:,metric_dict['tnr']])
# fpr_lstmcnn_3_3_p_std = np.std(1 - lstmcnn_3_3_p[:,metric_dict['tnr']])
# print('      P-type lstmcnn_3_3 fpr-mean:%.6f, fpr-std:%.6f.'%(fpr_lstmcnn_3_3_p_mean, fpr_lstmcnn_3_3_p_std))
#
auc_lstmcnn_3_3_p_mean = np.mean(lstmcnn_3_3_p[:,metric_dict['auc']])
auc_lstmcnn_3_3_p_std = np.std(lstmcnn_3_3_p[:,metric_dict['auc']])
print('      P-type lstmcnn_3_3 auc-mean:%.6f, auc-std:%.6f.'%(auc_lstmcnn_3_3_p_mean, auc_lstmcnn_3_3_p_std))

tnr_lstmcnn_3_3_p_mean = np.mean(lstmcnn_3_3_p[:,metric_dict['tnr']])
tnr_lstmcnn_3_3_p_std = np.std(lstmcnn_3_3_p[:,metric_dict['tnr']])
print('      P-type lstmcnn_3_3 tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_lstmcnn_3_3_p_mean, tnr_lstmcnn_3_3_p_std))

ppv_lstmcnn_3_3_p_mean = np.mean(lstmcnn_3_3_p[:,metric_dict['ppv']])
ppv_lstmcnn_3_3_p_std = np.std(lstmcnn_3_3_p[:,metric_dict['ppv']])
print('      P-type lstmcnn_3_3 ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_lstmcnn_3_3_p_mean, ppv_lstmcnn_3_3_p_std))

npv_lstmcnn_3_3_p_mean = np.mean(lstmcnn_3_3_p[:,metric_dict['npv']])
npv_lstmcnn_3_3_p_std = np.std(lstmcnn_3_3_p[:,metric_dict['npv']])
print('      P-type lstmcnn_3_3 npv-mean:%.6f, npv-std:%.6f.'%(npv_lstmcnn_3_3_p_mean, npv_lstmcnn_3_3_p_std))

acc_lstmcnn_3_3_p_mean = np.mean(lstmcnn_3_3_p[:,metric_dict['acc']])
acc_lstmcnn_3_3_p_std = np.std(lstmcnn_3_3_p[:,metric_dict['acc']])
print('      P-type lstmcnn_3_3 acc-mean:%.6f, acc-std:%.6f.'%(acc_lstmcnn_3_3_p_mean, acc_lstmcnn_3_3_p_std))

f1_lstmcnn_3_3_p_mean = np.mean(lstmcnn_3_3_p[:,metric_dict['f1']])
f1_lstmcnn_3_3_p_std = np.std(lstmcnn_3_3_p[:,metric_dict['f1']])
print('      P-type lstmcnn_3_3 f1-mean:%.6f, f1-std:%.6f.'%(f1_lstmcnn_3_3_p_mean, f1_lstmcnn_3_3_p_std))

mcc_lstmcnn_3_3_p_mean = np.mean(lstmcnn_3_3_p[:,metric_dict['mcc']])
mcc_lstmcnn_3_3_p_std = np.std(lstmcnn_3_3_p[:,metric_dict['mcc']])
print('      P-type lstmcnn_3_3 mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_lstmcnn_3_3_p_mean, mcc_lstmcnn_3_3_p_std))
#
tpr_clstmcnn_3_3_p_mean = np.mean(clstmcnn_3_3_p[:,metric_dict['tpr']])
tpr_clstmcnn_3_3_p_std = np.std(clstmcnn_3_3_p[:,metric_dict['tpr']])
print('##### P-type clstmcnn_3_3 tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_clstmcnn_3_3_p_mean, tpr_clstmcnn_3_3_p_std))

# fpr_clstmcnn_3_3_p_mean = np.mean(1 - clstmcnn_3_3_p[:,metric_dict['tnr']])
# fpr_clstmcnn_3_3_p_std = np.std(1 - clstmcnn_3_3_p[:,metric_dict['tnr']])
# print('      P-type clstmcnn_3_3 fpr-mean:%.6f, fpr-std:%.6f.'%(fpr_clstmcnn_3_3_p_mean, fpr_clstmcnn_3_3_p_std))
#
auc_clstmcnn_3_3_p_mean = np.mean(clstmcnn_3_3_p[:,metric_dict['auc']])
auc_clstmcnn_3_3_p_std = np.std(clstmcnn_3_3_p[:,metric_dict['auc']])
print('      P-type clstmcnn_3_3 auc-mean:%.6f, auc-std:%.6f.'%(auc_clstmcnn_3_3_p_mean, auc_clstmcnn_3_3_p_std))

tnr_clstmcnn_3_3_p_mean = np.mean(clstmcnn_3_3_p[:,metric_dict['tnr']])
tnr_clstmcnn_3_3_p_std = np.std(clstmcnn_3_3_p[:,metric_dict['tnr']])
print('      P-type clstmcnn_3_3 tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_clstmcnn_3_3_p_mean, tnr_clstmcnn_3_3_p_std))

ppv_clstmcnn_3_3_p_mean = np.mean(clstmcnn_3_3_p[:,metric_dict['ppv']])
ppv_clstmcnn_3_3_p_std = np.std(clstmcnn_3_3_p[:,metric_dict['ppv']])
print('      P-type clstmcnn_3_3 ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_clstmcnn_3_3_p_mean, ppv_clstmcnn_3_3_p_std))

npv_clstmcnn_3_3_p_mean = np.mean(clstmcnn_3_3_p[:,metric_dict['npv']])
npv_clstmcnn_3_3_p_std = np.std(clstmcnn_3_3_p[:,metric_dict['npv']])
print('      P-type clstmcnn_3_3 npv-mean:%.6f, npv-std:%.6f.'%(npv_clstmcnn_3_3_p_mean, npv_clstmcnn_3_3_p_std))

acc_clstmcnn_3_3_p_mean = np.mean(clstmcnn_3_3_p[:,metric_dict['acc']])
acc_clstmcnn_3_3_p_std = np.std(clstmcnn_3_3_p[:,metric_dict['acc']])
print('      P-type clstmcnn_3_3 acc-mean:%.6f, acc-std:%.6f.'%(acc_clstmcnn_3_3_p_mean, acc_clstmcnn_3_3_p_std))

f1_clstmcnn_3_3_p_mean = np.mean(clstmcnn_3_3_p[:,metric_dict['f1']])
f1_clstmcnn_3_3_p_std = np.std(clstmcnn_3_3_p[:,metric_dict['f1']])
print('      P-type clstmcnn_3_3 f1-mean:%.6f, f1-std:%.6f.'%(f1_clstmcnn_3_3_p_mean, f1_clstmcnn_3_3_p_std))

mcc_clstmcnn_3_3_p_mean = np.mean(clstmcnn_3_3_p[:,metric_dict['mcc']])
mcc_clstmcnn_3_3_p_std = np.std(clstmcnn_3_3_p[:,metric_dict['mcc']])
print('      P-type clstmcnn_3_3 mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_clstmcnn_3_3_p_mean, mcc_clstmcnn_3_3_p_std))
#
#
#
tpr_clstmcnn_3_3_p_2_mean = np.mean(clstmcnn_3_3_p_2[:,metric_dict['tpr']])
tpr_clstmcnn_3_3_p_2_std = np.std(clstmcnn_3_3_p_2[:,metric_dict['tpr']])
print('##### P-type clstmcnn_3_3_2 tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_clstmcnn_3_3_p_2_mean, tpr_clstmcnn_3_3_p_2_std))

# fpr_clstmcnn_3_3_p_2_mean = np.mean(1 - clstmcnn_3_3_p_2[:,metric_dict['tnr']])
# fpr_clstmcnn_3_3_p_2_std = np.std(1 - clstmcnn_3_3_p_2[:,metric_dict['tnr']])
# print('      P-type clstmcnn_3_3_2 fpr-mean:%.6f, fpr-std:%.6f.'%(fpr_clstmcnn_3_3_p_2_mean, fpr_clstmcnn_3_3_p_2_std))
#
auc_clstmcnn_3_3_p_2_mean = np.mean(clstmcnn_3_3_p_2[:,metric_dict['auc']])
auc_clstmcnn_3_3_p_2_std = np.std(clstmcnn_3_3_p_2[:,metric_dict['auc']])
print('      P-type clstmcnn_3_3_2 auc-mean:%.6f, auc-std:%.6f.'%(auc_clstmcnn_3_3_p_2_mean, auc_clstmcnn_3_3_p_2_std))


tnr_clstmcnn_3_3_p_2_mean = np.mean(clstmcnn_3_3_p_2[:,metric_dict['tnr']])
tnr_clstmcnn_3_3_p_2_std = np.std(clstmcnn_3_3_p_2[:,metric_dict['tnr']])
print('      P-type clstmcnn_3_3_2 tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_clstmcnn_3_3_p_2_mean, tnr_clstmcnn_3_3_p_2_std))

ppv_clstmcnn_3_3_p_2_mean = np.mean(clstmcnn_3_3_p_2[:,metric_dict['ppv']])
ppv_clstmcnn_3_3_p_2_std = np.std(clstmcnn_3_3_p_2[:,metric_dict['ppv']])
print('      P-type clstmcnn_3_3_2 ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_clstmcnn_3_3_p_2_mean, ppv_clstmcnn_3_3_p_2_std))

npv_clstmcnn_3_3_p_2_mean = np.mean(clstmcnn_3_3_p_2[:,metric_dict['npv']])
npv_clstmcnn_3_3_p_2_std = np.std(clstmcnn_3_3_p_2[:,metric_dict['npv']])
print('      P-type clstmcnn_3_3_2 npv-mean:%.6f, npv-std:%.6f.'%(npv_clstmcnn_3_3_p_2_mean, npv_clstmcnn_3_3_p_2_std))

acc_clstmcnn_3_3_p_2_mean = np.mean(clstmcnn_3_3_p_2[:,metric_dict['acc']])
acc_clstmcnn_3_3_p_2_std = np.std(clstmcnn_3_3_p_2[:,metric_dict['acc']])
print('      P-type clstmcnn_3_3_2 acc-mean:%.6f, acc-std:%.6f.'%(acc_clstmcnn_3_3_p_2_mean, acc_clstmcnn_3_3_p_2_std))

f1_clstmcnn_3_3_p_2_mean = np.mean(clstmcnn_3_3_p_2[:,metric_dict['f1']])
f1_clstmcnn_3_3_p_2_std = np.std(clstmcnn_3_3_p_2[:,metric_dict['f1']])
print('      P-type clstmcnn_3_3_2 f1-mean:%.6f, f1-std:%.6f.'%(f1_clstmcnn_3_3_p_2_mean, f1_clstmcnn_3_3_p_2_std))

mcc_clstmcnn_3_3_p_2_mean = np.mean(clstmcnn_3_3_p_2[:,metric_dict['mcc']])
mcc_clstmcnn_3_3_p_2_std = np.std(clstmcnn_3_3_p_2[:,metric_dict['mcc']])
print('      P-type clstmcnn_3_3_2 mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_clstmcnn_3_3_p_2_mean, mcc_clstmcnn_3_3_p_2_std))
#
#
# ''''''
tpr_mnist_pl_mean = np.mean(mnist_pl[:,metric_dict['tpr']])
tpr_mnist_pl_std = np.std(mnist_pl[:,metric_dict['tpr']])
print('##### Pl-type mnist tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_mnist_pl_mean, tpr_mnist_pl_std))

# fpr_mnist_pl_mean = np.mean(1 - mnist_pl[:,metric_dict['tnr']])
# fpr_mnist_pl_std = np.std(1 - mnist_pl[:,metric_dict['tnr']])
# print('      PL-type mnist fpr-mean:%.6f, fpr-std:%.6f.'%(fpr_mnist_pl_mean, fpr_mnist_pl_std))
#
auc_mnist_pl_mean = np.mean(mnist_pl[:,metric_dict['auc']])
auc_mnist_pl_std = np.std(mnist_pl[:,metric_dict['auc']])
print('      PL-type mnist auc-mean:%.6f, auc-std:%.6f.'%(auc_mnist_pl_mean, auc_mnist_pl_std))

tnr_mnist_pl_mean = np.mean(mnist_pl[:,metric_dict['tnr']])
tnr_mnist_pl_std = np.std(mnist_pl[:,metric_dict['tnr']])
print('      Pl-type mnist tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_mnist_pl_mean, tnr_mnist_pl_std))

ppv_mnist_pl_mean = np.mean(mnist_pl[:,metric_dict['ppv']])
ppv_mnist_pl_std = np.std(mnist_pl[:,metric_dict['ppv']])
print('      Pl-type mnist ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_mnist_pl_mean, ppv_mnist_pl_std))

npv_mnist_pl_mean = np.mean(mnist_pl[:,metric_dict['npv']])
npv_mnist_pl_std = np.std(mnist_pl[:,metric_dict['npv']])
print('      Pl-type mnist npv-mean:%.6f, npv-std:%.6f.'%(npv_mnist_pl_mean, npv_mnist_pl_std))

acc_mnist_pl_mean = np.mean(mnist_pl[:,metric_dict['acc']])
acc_mnist_pl_std = np.std(mnist_pl[:,metric_dict['acc']])
print('      Pl-type mnist acc-mean:%.6f, acc-std:%.6f.'%(acc_mnist_pl_mean, acc_mnist_pl_std))

f1_mnist_pl_mean = np.mean(mnist_pl[:,metric_dict['f1']])
f1_mnist_pl_std = np.std(mnist_pl[:,metric_dict['f1']])
print('      Pl-type mnist f1-mean:%.6f, f1-std:%.6f.'%(f1_mnist_pl_mean, f1_mnist_pl_std))

mcc_mnist_pl_mean = np.mean(mnist_pl[:,metric_dict['mcc']])
mcc_mnist_pl_std = np.std(mnist_pl[:,metric_dict['mcc']])
print('      Pl-type mnist mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_mnist_pl_mean, mcc_mnist_pl_std))

tpr_mnist_1_pl_mean = np.mean(mnist_1_pl[:,metric_dict['tpr']])
tpr_mnist_1_pl_std = np.std(mnist_1_pl[:,metric_dict['tpr']])
print('##### Pl-type mnist1 tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_mnist_1_pl_mean, tpr_mnist_1_pl_std))

# fpr_mnist_1_pl_mean = np.mean(1 - mnist_1_pl[:,metric_dict['tnr']])
# fpr_mnist_1_pl_std = np.std(1 - mnist_1_pl[:,metric_dict['tnr']])
# print('      PL-type mnist1 fpr-mean:%.6f, fpr-std:%.6f.'%(fpr_mnist_1_pl_mean, fpr_mnist_1_pl_std))
#
auc_mnist_1_pl_mean = np.mean(mnist_1_pl[:,metric_dict['auc']])
auc_mnist_1_pl_std = np.std(mnist_1_pl[:,metric_dict['auc']])
print('      PL-type mnist1 auc-mean:%.6f, auc-std:%.6f.'%(auc_mnist_1_pl_mean, auc_mnist_1_pl_std))

tnr_mnist_1_pl_mean = np.mean(mnist_1_pl[:,metric_dict['tnr']])
tnr_mnist_1_pl_std = np.std(mnist_1_pl[:,metric_dict['tnr']])
print('      Pl-type mnist1 tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_mnist_1_pl_mean, tnr_mnist_1_pl_std))

ppv_mnist_1_pl_mean = np.mean(mnist_1_pl[:,metric_dict['ppv']])
ppv_mnist_1_pl_std = np.std(mnist_1_pl[:,metric_dict['ppv']])
print('      Pl-type mnist1 ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_mnist_1_pl_mean, ppv_mnist_1_pl_std))

npv_mnist_1_pl_mean = np.mean(mnist_1_pl[:,metric_dict['npv']])
npv_mnist_1_pl_std = np.std(mnist_1_pl[:,metric_dict['npv']])
print('      Pl-type mnist1 npv-mean:%.6f, npv-std:%.6f.'%(npv_mnist_1_pl_mean, npv_mnist_1_pl_std))

acc_mnist_1_pl_mean = np.mean(mnist_1_pl[:,metric_dict['acc']])
acc_mnist_1_pl_std = np.std(mnist_1_pl[:,metric_dict['acc']])
print('      Pl-type mnist1 acc-mean:%.6f, acc-std:%.6f.'%(acc_mnist_1_pl_mean, acc_mnist_1_pl_std))

f1_mnist_1_pl_mean = np.mean(mnist_1_pl[:,metric_dict['f1']])
f1_mnist_1_pl_std = np.std(mnist_1_pl[:,metric_dict['f1']])
print('      Pl-type mnist1 f1-mean:%.6f, f1-std:%.6f.'%(f1_mnist_1_pl_mean, f1_mnist_1_pl_std))

mcc_mnist_1_pl_mean = np.mean(mnist_1_pl[:,metric_dict['mcc']])
mcc_mnist_1_pl_std = np.std(mnist_1_pl[:,metric_dict['mcc']])
print('      Pl-type mnist1 mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_mnist_1_pl_mean, mcc_mnist_1_pl_std))


auc_cgrucnn_3_3_pl_mean = np.mean(cgrucnn_3_3_pl[:,metric_dict['auc']])
auc_cgrucnn_3_3_pl_std = np.std(cgrucnn_3_3_pl[:,metric_dict['auc']])
print('##### PL-type grucnn auc-mean:%.6f, auc-std:%.6f.'%(auc_cgrucnn_3_3_pl_mean, auc_cgrucnn_3_3_pl_std))

tpr_cgrucnn_3_3_pl_mean = np.mean(cgrucnn_3_3_pl[:,metric_dict['tpr']])
tpr_cgrucnn_3_3_pl_std = np.std(cgrucnn_3_3_pl[:,metric_dict['tpr']])
print('      PL-type grucnn tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_cgrucnn_3_3_pl_mean, tpr_cgrucnn_3_3_pl_std))
acc_cgrucnn_3_3_pl_mean = np.mean(cgrucnn_3_3_pl[:,metric_dict['acc']])
acc_cgrucnn_3_3_pl_std = np.std(cgrucnn_3_3_pl[:,metric_dict['acc']])
print('      PL-type grucnn acc-mean:%.6f, acc-std:%.6f.'%(acc_cgrucnn_3_3_pl_mean, acc_cgrucnn_3_3_pl_std))

f1_cgrucnn_3_3_pl_mean = np.mean(cgrucnn_3_3_pl[:,metric_dict['f1']])
f1_cgrucnn_3_3_pl_std = np.std(cgrucnn_3_3_pl[:,metric_dict['f1']])
print('      PL-type grucnn f1-mean:%.6f, f1-std:%.6f.'%(f1_cgrucnn_3_3_pl_mean, f1_cgrucnn_3_3_pl_std))

mcc_cgrucnn_3_3_pl_mean = np.mean(cgrucnn_3_3_pl[:,metric_dict['mcc']])
mcc_cgrucnn_3_3_pl_std = np.std(cgrucnn_3_3_pl[:,metric_dict['mcc']])
print('      PL-type grucnn mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_cgrucnn_3_3_pl_mean, mcc_cgrucnn_3_3_pl_std))




auc_crnncnn_3_3_pl_mean = np.mean(crnncnn_3_3_pl[:,metric_dict['auc']])
auc_crnncnn_3_3_pl_std = np.std(crnncnn_3_3_pl[:,metric_dict['auc']])
print('##### Pl-type rnncnn auc-mean:%.6f, auc-std:%.6f.'%(auc_crnncnn_3_3_pl_mean, auc_crnncnn_3_3_pl_std))

tpr_crnncnn_3_3_pl_mean = np.mean(crnncnn_3_3_pl[:,metric_dict['tpr']])
tpr_crnncnn_3_3_pl_std = np.std(crnncnn_3_3_pl[:,metric_dict['tpr']])
print('      Pl-type rnncnn tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_crnncnn_3_3_pl_mean, tpr_crnncnn_3_3_pl_std))
acc_crnncnn_3_3_pl_mean = np.mean(crnncnn_3_3_pl[:,metric_dict['acc']])
acc_crnncnn_3_3_pl_std = np.std(crnncnn_3_3_pl[:,metric_dict['acc']])
print('      Pl-type rnncnn acc-mean:%.6f, acc-std:%.6f.'%(acc_crnncnn_3_3_pl_mean, acc_crnncnn_3_3_pl_std))

f1_crnncnn_3_3_pl_mean = np.mean(crnncnn_3_3_pl[:,metric_dict['f1']])
f1_crnncnn_3_3_pl_std = np.std(crnncnn_3_3_pl[:,metric_dict['f1']])
print('      Pl-type rnncnn f1-mean:%.6f, f1-std:%.6f.'%(f1_crnncnn_3_3_pl_mean, f1_crnncnn_3_3_pl_std))

mcc_crnncnn_3_3_pl_mean = np.mean(crnncnn_3_3_pl[:,metric_dict['mcc']])
mcc_crnncnn_3_3_pl_std = np.std(crnncnn_3_3_pl[:,metric_dict['mcc']])
print('      Pl-type rnncnn mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_crnncnn_3_3_pl_mean, mcc_crnncnn_3_3_pl_std))




#
#
# tpr_lstmcnn_5_3_pl_mean = np.mean(lstmcnn_5_3_pl[:,metric_dict['tpr']])
# tpr_lstmcnn_5_3_pl_std = np.std(lstmcnn_5_3_pl[:,metric_dict['tpr']])
# print('##### Pl-type lstmcnn_5_3 tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_lstmcnn_5_3_pl_mean, tpr_lstmcnn_5_3_pl_std))
#
# tnr_lstmcnn_5_3_pl_mean = np.mean(lstmcnn_5_3_pl[:,metric_dict['tnr']])
# tnr_lstmcnn_5_3_pl_std = np.std(lstmcnn_5_3_pl[:,metric_dict['tnr']])
# print('      Pl-type lstmcnn_5_3 tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_lstmcnn_5_3_pl_mean, tnr_lstmcnn_5_3_pl_std))
#
# ppv_lstmcnn_5_3_pl_mean = np.mean(lstmcnn_5_3_pl[:,metric_dict['ppv']])
# ppv_lstmcnn_5_3_pl_std = np.std(lstmcnn_5_3_pl[:,metric_dict['ppv']])
# print('      Pl-type lstmcnn_5_3 ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_lstmcnn_5_3_pl_mean, ppv_lstmcnn_5_3_pl_std))
#
# npv_lstmcnn_5_3_pl_mean = np.mean(lstmcnn_5_3_pl[:,metric_dict['npv']])
# npv_lstmcnn_5_3_pl_std = np.std(lstmcnn_5_3_pl[:,metric_dict['npv']])
# print('      Pl-type lstmcnn_5_3 npv-mean:%.6f, npv-std:%.6f.'%(npv_lstmcnn_5_3_pl_mean, npv_lstmcnn_5_3_pl_std))
#
# acc_lstmcnn_5_3_pl_mean = np.mean(lstmcnn_5_3_pl[:,metric_dict['acc']])
# acc_lstmcnn_5_3_pl_std = np.std(lstmcnn_5_3_pl[:,metric_dict['acc']])
# print('      Pl-type lstmcnn_5_3 acc-mean:%.6f, acc-std:%.6f.'%(acc_lstmcnn_5_3_pl_mean, acc_lstmcnn_5_3_pl_std))
#
# f1_lstmcnn_5_3_pl_mean = np.mean(lstmcnn_5_3_pl[:,metric_dict['f1']])
# f1_lstmcnn_5_3_pl_std = np.std(lstmcnn_5_3_pl[:,metric_dict['f1']])
# print('      Pl-type lstmcnn_5_3 f1-mean:%.6f, f1-std:%.6f.'%(f1_lstmcnn_5_3_pl_mean, f1_lstmcnn_5_3_pl_std))
#
# mcc_lstmcnn_5_3_pl_mean = np.mean(lstmcnn_5_3_pl[:,metric_dict['mcc']])
# mcc_lstmcnn_5_3_pl_std = np.std(lstmcnn_5_3_pl[:,metric_dict['mcc']])
# print('      Pl-type lstmcnn_5_3 mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_lstmcnn_5_3_pl_mean, mcc_lstmcnn_5_3_pl_std))
#
#
#
# tpr_lstmcnn_5_5_pl_mean = np.mean(lstmcnn_5_5_pl[:,metric_dict['tpr']])
# tpr_lstmcnn_5_5_pl_std = np.std(lstmcnn_5_5_pl[:,metric_dict['tpr']])
# print('##### Pl-type lstmcnn_5_5 tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_lstmcnn_5_5_pl_mean, tpr_lstmcnn_5_5_pl_std))
#
# tnr_lstmcnn_5_5_pl_mean = np.mean(lstmcnn_5_5_pl[:,metric_dict['tnr']])
# tnr_lstmcnn_5_5_pl_std = np.std(lstmcnn_5_5_pl[:,metric_dict['tnr']])
# print('      Pl-type lstmcnn_5_5 tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_lstmcnn_5_5_pl_mean, tnr_lstmcnn_5_5_pl_std))
#
# ppv_lstmcnn_5_5_pl_mean = np.mean(lstmcnn_5_5_pl[:,metric_dict['ppv']])
# ppv_lstmcnn_5_5_pl_std = np.std(lstmcnn_5_5_pl[:,metric_dict['ppv']])
# print('      Pl-type lstmcnn_5_5 ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_lstmcnn_5_5_pl_mean, ppv_lstmcnn_5_5_pl_std))
#
# npv_lstmcnn_5_5_pl_mean = np.mean(lstmcnn_5_5_pl[:,metric_dict['npv']])
# npv_lstmcnn_5_5_pl_std = np.std(lstmcnn_5_5_pl[:,metric_dict['npv']])
# print('      Pl-type lstmcnn_5_5 npv-mean:%.6f, npv-std:%.6f.'%(npv_lstmcnn_5_5_pl_mean, npv_lstmcnn_5_5_pl_std))
#
# acc_lstmcnn_5_5_pl_mean = np.mean(lstmcnn_5_5_pl[:,metric_dict['acc']])
# acc_lstmcnn_5_5_pl_std = np.std(lstmcnn_5_5_pl[:,metric_dict['acc']])
# print('      Pl-type lstmcnn_5_5 acc-mean:%.6f, acc-std:%.6f.'%(acc_lstmcnn_5_5_pl_mean, acc_lstmcnn_5_5_pl_std))
#
# f1_lstmcnn_5_5_pl_mean = np.mean(lstmcnn_5_5_pl[:,metric_dict['f1']])
# f1_lstmcnn_5_5_pl_std = np.std(lstmcnn_5_5_pl[:,metric_dict['f1']])
# print('      Pl-type lstmcnn_5_5 f1-mean:%.6f, f1-std:%.6f.'%(f1_lstmcnn_5_5_pl_mean, f1_lstmcnn_5_5_pl_std))
#
# mcc_lstmcnn_5_5_pl_mean = np.mean(lstmcnn_5_5_pl[:,metric_dict['mcc']])
# mcc_lstmcnn_5_5_pl_std = np.std(lstmcnn_5_5_pl[:,metric_dict['mcc']])
# print('      Pl-type lstmcnn_5_5 mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_lstmcnn_5_5_pl_mean, mcc_lstmcnn_5_5_pl_std))
#
#
#
tpr_lstmcnn_3_3_pl_mean = np.mean(lstmcnn_3_3_pl[:,metric_dict['tpr']])
tpr_lstmcnn_3_3_pl_std = np.std(lstmcnn_3_3_pl[:,metric_dict['tpr']])
print('##### Pl-type lstmcnn_3_3 tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_lstmcnn_3_3_pl_mean, tpr_lstmcnn_3_3_pl_std))

# fpr_lstmcnn_3_3_pl_mean = np.mean(1 - lstmcnn_3_3_pl[:,metric_dict['tnr']])
# fpr_lstmcnn_3_3_pl_std = np.std(1 - lstmcnn_3_3_pl[:,metric_dict['tnr']])
# print('      PL-type lstmcnn_3_3 fpr-mean:%.6f, fpr-std:%.6f.'%(fpr_lstmcnn_3_3_pl_mean, fpr_lstmcnn_3_3_pl_std))
#
auc_lstmcnn_3_3_pl_mean = np.mean(lstmcnn_3_3_pl[:,metric_dict['auc']])
auc_lstmcnn_3_3_pl_std = np.std(lstmcnn_3_3_pl[:,metric_dict['auc']])
print('      PL-type lstmcnn_3_3 auc-mean:%.6f, auc-std:%.6f.'%(auc_lstmcnn_3_3_pl_mean, auc_lstmcnn_3_3_pl_std))

tnr_lstmcnn_3_3_pl_mean = np.mean(lstmcnn_3_3_pl[:,metric_dict['tnr']])
tnr_lstmcnn_3_3_pl_std = np.std(lstmcnn_3_3_pl[:,metric_dict['tnr']])
print('      Pl-type lstmcnn_3_3 tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_lstmcnn_3_3_pl_mean, tnr_lstmcnn_3_3_pl_std))

ppv_lstmcnn_3_3_pl_mean = np.mean(lstmcnn_3_3_pl[:,metric_dict['ppv']])
ppv_lstmcnn_3_3_pl_std = np.std(lstmcnn_3_3_pl[:,metric_dict['ppv']])
print('      Pl-type lstmcnn_3_3 ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_lstmcnn_3_3_pl_mean, ppv_lstmcnn_3_3_pl_std))

npv_lstmcnn_3_3_pl_mean = np.mean(lstmcnn_3_3_pl[:,metric_dict['npv']])
npv_lstmcnn_3_3_pl_std = np.std(lstmcnn_3_3_pl[:,metric_dict['npv']])
print('      Pl-type lstmcnn_3_3 npv-mean:%.6f, npv-std:%.6f.'%(npv_lstmcnn_3_3_pl_mean, npv_lstmcnn_3_3_pl_std))

acc_lstmcnn_3_3_pl_mean = np.mean(lstmcnn_3_3_pl[:,metric_dict['acc']])
acc_lstmcnn_3_3_pl_std = np.std(lstmcnn_3_3_pl[:,metric_dict['acc']])
print('      Pl-type lstmcnn_3_3 acc-mean:%.6f, acc-std:%.6f.'%(acc_lstmcnn_3_3_pl_mean, acc_lstmcnn_3_3_pl_std))

f1_lstmcnn_3_3_pl_mean = np.mean(lstmcnn_3_3_pl[:,metric_dict['f1']])
f1_lstmcnn_3_3_pl_std = np.std(lstmcnn_3_3_pl[:,metric_dict['f1']])
print('      Pl-type lstmcnn_3_3 f1-mean:%.6f, f1-std:%.6f.'%(f1_lstmcnn_3_3_pl_mean, f1_lstmcnn_3_3_pl_std))

mcc_lstmcnn_3_3_pl_mean = np.mean(lstmcnn_3_3_pl[:,metric_dict['mcc']])
mcc_lstmcnn_3_3_pl_std = np.std(lstmcnn_3_3_pl[:,metric_dict['mcc']])
print('      Pl-type lstmcnn_3_3 mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_lstmcnn_3_3_pl_mean, mcc_lstmcnn_3_3_pl_std))


#
tpr_clstmcnn_3_3_pl_mean = np.mean(clstmcnn_3_3_pl[:,metric_dict['tpr']])
tpr_clstmcnn_3_3_pl_std = np.std(clstmcnn_3_3_pl[:,metric_dict['tpr']])
print('##### Pl-type clstmcnn_3_3 tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_clstmcnn_3_3_pl_mean, tpr_clstmcnn_3_3_pl_std))

# fpr_clstmcnn_3_3_pl_mean = np.mean(1 - clstmcnn_3_3_pl[:,metric_dict['tnr']])
# fpr_clstmcnn_3_3_pl_std = np.std(1 - clstmcnn_3_3_pl[:,metric_dict['tnr']])
# print('      PL-type clstmcnn_3_3 fpr-mean:%.6f, fpr-std:%.6f.'%(fpr_clstmcnn_3_3_pl_mean, fpr_clstmcnn_3_3_pl_std))
#
auc_clstmcnn_3_3_pl_mean = np.mean(clstmcnn_3_3_pl[:,metric_dict['auc']])
auc_clstmcnn_3_3_pl_std = np.std(clstmcnn_3_3_pl[:,metric_dict['auc']])
print('      PL-type clstmcnn_3_3 auc-mean:%.6f, auc-std:%.6f.'%(auc_clstmcnn_3_3_pl_mean, auc_clstmcnn_3_3_pl_std))

tnr_clstmcnn_3_3_pl_mean = np.mean(clstmcnn_3_3_pl[:,metric_dict['tnr']])
tnr_clstmcnn_3_3_pl_std = np.std(clstmcnn_3_3_pl[:,metric_dict['tnr']])
print('      Pl-type clstmcnn_3_3 tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_clstmcnn_3_3_pl_mean, tnr_clstmcnn_3_3_pl_std))

ppv_clstmcnn_3_3_pl_mean = np.mean(clstmcnn_3_3_pl[:,metric_dict['ppv']])
ppv_clstmcnn_3_3_pl_std = np.std(clstmcnn_3_3_pl[:,metric_dict['ppv']])
print('      Pl-type clstmcnn_3_3 ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_clstmcnn_3_3_pl_mean, ppv_clstmcnn_3_3_pl_std))

npv_clstmcnn_3_3_pl_mean = np.mean(clstmcnn_3_3_pl[:,metric_dict['npv']])
npv_clstmcnn_3_3_pl_std = np.std(clstmcnn_3_3_pl[:,metric_dict['npv']])
print('      Pl-type clstmcnn_3_3 npv-mean:%.6f, npv-std:%.6f.'%(npv_clstmcnn_3_3_pl_mean, npv_clstmcnn_3_3_pl_std))

acc_clstmcnn_3_3_pl_mean = np.mean(clstmcnn_3_3_pl[:,metric_dict['acc']])
acc_clstmcnn_3_3_pl_std = np.std(clstmcnn_3_3_pl[:,metric_dict['acc']])
print('      Pl-type clstmcnn_3_3 acc-mean:%.6f, acc-std:%.6f.'%(acc_clstmcnn_3_3_pl_mean, acc_clstmcnn_3_3_pl_std))

f1_clstmcnn_3_3_pl_mean = np.mean(clstmcnn_3_3_pl[:,metric_dict['f1']])
f1_clstmcnn_3_3_pl_std = np.std(clstmcnn_3_3_pl[:,metric_dict['f1']])
print('      Pl-type clstmcnn_3_3 f1-mean:%.6f, f1-std:%.6f.'%(f1_clstmcnn_3_3_pl_mean, f1_clstmcnn_3_3_pl_std))

mcc_clstmcnn_3_3_pl_mean = np.mean(clstmcnn_3_3_pl[:,metric_dict['mcc']])
mcc_clstmcnn_3_3_pl_std = np.std(clstmcnn_3_3_pl[:,metric_dict['mcc']])
print('      Pl-type clstmcnn_3_3 mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_clstmcnn_3_3_pl_mean, mcc_clstmcnn_3_3_pl_std))


tpr_clstmcnn_3_3_pl_2_mean = np.mean(clstmcnn_3_3_pl_2[:,metric_dict['tpr']])
tpr_clstmcnn_3_3_pl_2_std = np.std(clstmcnn_3_3_pl_2[:,metric_dict['tpr']])
print('##### PL-type clstmcnn_3_3_2 tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_clstmcnn_3_3_pl_2_mean, tpr_clstmcnn_3_3_pl_2_std))

# fpr_clstmcnn_3_3_pl_2_mean = np.mean(1 - clstmcnn_3_3_pl_2[:,metric_dict['tnr']])
# fpr_clstmcnn_3_3_pl_2_std = np.std(1 - clstmcnn_3_3_pl_2[:,metric_dict['tnr']])
# print('      PL-type clstmcnn_3_3_2 fpr-mean:%.6f, fpr-std:%.6f.'%(fpr_clstmcnn_3_3_pl_2_mean, fpr_clstmcnn_3_3_pl_2_std))
#
auc_clstmcnn_3_3_pl_2_mean = np.mean(clstmcnn_3_3_pl_2[:,metric_dict['auc']])
auc_clstmcnn_3_3_pl_2_std = np.std(clstmcnn_3_3_pl_2[:,metric_dict['auc']])
print('      PL-type clstmcnn_3_3_2 auc-mean:%.6f, auc-std:%.6f.'%(auc_clstmcnn_3_3_pl_2_mean, auc_clstmcnn_3_3_pl_2_std))

tnr_clstmcnn_3_3_pl_2_mean = np.mean(clstmcnn_3_3_pl_2[:,metric_dict['tnr']])
tnr_clstmcnn_3_3_pl_2_std = np.std(clstmcnn_3_3_pl_2[:,metric_dict['tnr']])
print('      PL-type clstmcnn_3_3_2 tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_clstmcnn_3_3_pl_2_mean, tnr_clstmcnn_3_3_pl_2_std))

ppv_clstmcnn_3_3_pl_2_mean = np.mean(clstmcnn_3_3_pl_2[:,metric_dict['ppv']])
ppv_clstmcnn_3_3_pl_2_std = np.std(clstmcnn_3_3_pl_2[:,metric_dict['ppv']])
print('      PL-type clstmcnn_3_3_2 ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_clstmcnn_3_3_pl_2_mean, ppv_clstmcnn_3_3_pl_2_std))

npv_clstmcnn_3_3_pl_2_mean = np.mean(clstmcnn_3_3_pl_2[:,metric_dict['npv']])
npv_clstmcnn_3_3_pl_2_std = np.std(clstmcnn_3_3_pl_2[:,metric_dict['npv']])
print('      PL-type clstmcnn_3_3_2 npv-mean:%.6f, npv-std:%.6f.'%(npv_clstmcnn_3_3_pl_2_mean, npv_clstmcnn_3_3_pl_2_std))

acc_clstmcnn_3_3_pl_2_mean = np.mean(clstmcnn_3_3_pl_2[:,metric_dict['acc']])
acc_clstmcnn_3_3_pl_2_std = np.std(clstmcnn_3_3_pl_2[:,metric_dict['acc']])
print('      PL-type clstmcnn_3_3_2 acc-mean:%.6f, acc-std:%.6f.'%(acc_clstmcnn_3_3_pl_2_mean, acc_clstmcnn_3_3_pl_2_std))

f1_clstmcnn_3_3_pl_2_mean = np.mean(clstmcnn_3_3_pl_2[:,metric_dict['f1']])
f1_clstmcnn_3_3_pl_2_std = np.std(clstmcnn_3_3_pl_2[:,metric_dict['f1']])
print('      PL-type clstmcnn_3_3_2 f1-mean:%.6f, f1-std:%.6f.'%(f1_clstmcnn_3_3_pl_2_mean, f1_clstmcnn_3_3_pl_2_std))

mcc_clstmcnn_3_3_pl_2_mean = np.mean(clstmcnn_3_3_pl_2[:,metric_dict['mcc']])
mcc_clstmcnn_3_3_pl_2_std = np.std(clstmcnn_3_3_pl_2[:,metric_dict['mcc']])
print('      PL-type clstmcnn_3_3_2 mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_clstmcnn_3_3_pl_2_mean, mcc_clstmcnn_3_3_pl_2_std))
