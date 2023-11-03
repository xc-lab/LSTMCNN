import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plot
sub_map = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')

'''
p-type测试数据
'''
window_size_32_p = np.array([
    [0.814815, 0.782609, 0.750000, 0.818182, 0.812500, 0.866667, 0.623635],
    [0.851852, 0.818182, 0.750000, 0.900000, 0.823529, 0.933333, 0.703144],
    [0.851852, 0.818182, 0.750000, 0.900000, 0.823529, 0.933333, 0.703144],
    [0.814815, 0.782609, 0.750000, 0.818182, 0.812500, 0.866667, 0.623635],
    [0.814815, 0.782609, 0.750000, 0.818182, 0.812500, 0.866667, 0.623635]
])

window_size_64_p = np.array([
    [0.923077, 0.909091, 0.909091, 0.909091, 0.933333, 0.933333, 0.842424],
    [0.846154, 0.818182, 0.818182, 0.818182, 0.866667, 0.866667, 0.684848],
    [0.807692, 0.761905, 0.727273, 0.800000, 0.812500, 0.866667, 0.603148],
    [0.923077, 0.909091, 0.909091, 0.909091, 0.933333, 0.933333, 0.842424],
    [0.884615, 0.842105, 0.727273, 1.000000, 0.833333, 1.000000, 0.778499]
])

window_size_128_p = np.array([
    [0.961538, 0.956522, 1.000000, 0.916667, 1.000000, 0.933333, 0.924962],
    [0.961538, 0.952381, 0.909091, 1.000000, 0.937500, 1.000000, 0.923168],
    [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000],
    [0.923077, 0.909091, 0.909091, 0.909091, 0.933333, 0.933333, 0.842424],
    [0.961538, 0.952381, 0.909091, 1.000000, 0.937500, 1.000000, 0.923168],
])

window_size_256_p = np.array([
    [0.923077, 0.900000, 0.818182, 1.000000, 0.882353, 1.000000, 0.849662],
    [0.961538, 0.956522, 1.000000, 0.916667, 1.000000, 0.933333, 0.924962],
    [0.923077, 0.916667, 1.000000, 0.846154, 1.000000, 0.866667, 0.856349],
    [0.961538, 0.956522, 1.000000, 0.916667, 1.000000, 0.933333, 0.924962],
    [0.846154, 0.800000, 0.727273, 0.888889, 0.823529, 0.933333, 0.686023]
])

'''
pl-type 数据
'''
window_size_32_pl = np.array([
    [0.880000, 0.869565, 0.909091, 0.833333, 0.923077, 0.857143, 0.761306],
    [0.880000, 0.869565, 0.909091, 0.833333, 0.923077, 0.857143, 0.761306],
    [0.880000, 0.869565, 0.909091, 0.833333, 0.923077, 0.857143, 0.761306],
    [0.800000, 0.761905, 0.727273, 0.800000, 0.800000, 0.857143, 0.592157],
    [0.880000, 0.869565, 0.909091, 0.833333, 0.923077, 0.857143, 0.761306]
])

window_size_64_pl = np.array([
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114],
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114],
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114],
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132],
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132]
])

window_size_128_pl = np.array([
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132],
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132],
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132],
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132],
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114]
])

window_size_256_pl = np.array([
    [0.920000, 0.909091, 0.909091, 0.909091, 0.928571, 0.928571, 0.837662],
    [0.920000, 0.909091, 0.909091, 0.909091, 0.928571, 0.928571, 0.837662],
    [0.920000, 0.909091, 0.909091, 0.909091, 0.928571, 0.928571, 0.837662],
    [0.920000, 0.909091, 0.909091, 0.909091, 0.928571, 0.928571, 0.837662],
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132]
])

'''
取数据
'''
metric_dict = {'acc':0, 'f1':1, 'tpr':2, 'ppv':3, 'npv':4, 'tnr':5, 'mcc':6}

'''
画图
'''
#TPR
tpr_window_size_32_p_mean = np.mean(window_size_32_p[:, metric_dict['tpr']])
tpr_window_size_32_p_std = np.std(window_size_32_p[:, metric_dict['tpr']])
print('      P-type 32 tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_window_size_32_p_mean, tpr_window_size_32_p_std))

tpr_window_size_64_p_mean = np.mean(window_size_64_p[:, metric_dict['tpr']])
tpr_window_size_64_p_std = np.std(window_size_64_p[:, metric_dict['tpr']])
print('      P-type 64 tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_window_size_64_p_mean, tpr_window_size_64_p_std))

tpr_window_size_128_p_mean = np.mean(window_size_128_p[:, metric_dict['tpr']])
tpr_window_size_128_p_std = np.std(window_size_128_p[:, metric_dict['tpr']])
print('      P-type 128 tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_window_size_128_p_mean, tpr_window_size_128_p_std))

tpr_window_size_256_p_mean = np.mean(window_size_256_p[:, metric_dict['tpr']])
tpr_window_size_256_p_std = np.std(window_size_256_p[:, metric_dict['tpr']])
print('      P-type 256 tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_window_size_256_p_mean, tpr_window_size_256_p_std))


tpr_window_size_32_pl_mean = np.mean(window_size_32_pl[:, metric_dict['tpr']])
tpr_window_size_32_pl_std = np.std(window_size_32_pl[:, metric_dict['tpr']])
print('      PL-type 32 tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_window_size_32_pl_mean, tpr_window_size_32_pl_std))

tpr_window_size_64_pl_mean = np.mean(window_size_64_pl[:, metric_dict['tpr']])
tpr_window_size_64_pl_std = np.std(window_size_64_pl[:, metric_dict['tpr']])
print('      PL-type 64 tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_window_size_64_pl_mean, tpr_window_size_64_pl_std))

tpr_window_size_128_pl_mean = np.mean(window_size_128_pl[:, metric_dict['tpr']])
tpr_window_size_128_pl_std = np.std(window_size_128_pl[:, metric_dict['tpr']])
print('      PL-type 128 tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_window_size_128_pl_mean, tpr_window_size_128_pl_std))

tpr_window_size_256_pl_mean = np.mean(window_size_256_pl[:, metric_dict['tpr']])
tpr_window_size_256_pl_std = np.std(window_size_256_pl[:, metric_dict['tpr']])
print('      PL-type 256 tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_window_size_256_pl_mean, tpr_window_size_256_pl_std))




#PPV
ppv_window_size_32_p_mean = np.mean(window_size_32_p[:, metric_dict['ppv']])
ppv_window_size_32_p_std = np.std(window_size_32_p[:, metric_dict['ppv']])
print('      P-type 32 ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_window_size_32_p_mean, ppv_window_size_32_p_std))

ppv_window_size_64_p_mean = np.mean(window_size_64_p[:, metric_dict['ppv']])
ppv_window_size_64_p_std = np.std(window_size_64_p[:, metric_dict['ppv']])
print('      P-type 64 ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_window_size_64_p_mean, ppv_window_size_64_p_std))

ppv_window_size_128_p_mean = np.mean(window_size_128_p[:, metric_dict['ppv']])
ppv_window_size_128_p_std = np.std(window_size_128_p[:, metric_dict['ppv']])
print('      P-type 128 ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_window_size_128_p_mean, ppv_window_size_128_p_std))

ppv_window_size_256_p_mean = np.mean(window_size_256_p[:, metric_dict['ppv']])
ppv_window_size_256_p_std = np.std(window_size_256_p[:, metric_dict['ppv']])
print('      P-type 256 ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_window_size_256_p_mean, ppv_window_size_256_p_std))


ppv_window_size_32_pl_mean = np.mean(window_size_32_pl[:, metric_dict['ppv']])
ppv_window_size_32_pl_std = np.std(window_size_32_pl[:, metric_dict['ppv']])
print('      PL-type 32 ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_window_size_32_pl_mean, ppv_window_size_32_pl_std))

ppv_window_size_64_pl_mean = np.mean(window_size_64_pl[:, metric_dict['ppv']])
ppv_window_size_64_pl_std = np.std(window_size_64_pl[:, metric_dict['ppv']])
print('      PL-type 64 ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_window_size_64_pl_mean, ppv_window_size_64_pl_std))

ppv_window_size_128_pl_mean = np.mean(window_size_128_pl[:, metric_dict['ppv']])
ppv_window_size_128_pl_std = np.std(window_size_128_pl[:, metric_dict['ppv']])
print('      PL-type 128 ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_window_size_128_pl_mean, ppv_window_size_128_pl_std))

ppv_window_size_256_pl_mean = np.mean(window_size_256_pl[:, metric_dict['ppv']])
ppv_window_size_256_pl_std = np.std(window_size_256_pl[:, metric_dict['ppv']])
print('      PL-type 256 ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_window_size_256_pl_mean, ppv_window_size_256_pl_std))



# NPV
npv_window_size_32_p_mean = np.mean(window_size_32_p[:, metric_dict['npv']])
npv_window_size_32_p_std = np.std(window_size_32_p[:, metric_dict['npv']])
print('      P-type 32 npv-mean:%.6f, npv-std:%.6f.'%(npv_window_size_32_p_mean, npv_window_size_32_p_std))

npv_window_size_64_p_mean = np.mean(window_size_64_p[:, metric_dict['npv']])
npv_window_size_64_p_std = np.std(window_size_64_p[:, metric_dict['npv']])
print('      P-type 64 npv-mean:%.6f, npv-std:%.6f.'%(npv_window_size_64_p_mean, npv_window_size_64_p_std))

npv_window_size_128_p_mean = np.mean(window_size_128_p[:, metric_dict['npv']])
npv_window_size_128_p_std = np.std(window_size_128_p[:, metric_dict['npv']])
print('      P-type 128 npv-mean:%.6f, npv-std:%.6f.'%(npv_window_size_128_p_mean, npv_window_size_128_p_std))

npv_window_size_256_p_mean = np.mean(window_size_256_p[:, metric_dict['npv']])
npv_window_size_256_p_std = np.std(window_size_256_p[:, metric_dict['npv']])
print('      P-type 256 npv-mean:%.6f, npv-std:%.6f.'%(npv_window_size_256_p_mean, npv_window_size_256_p_std))

npv_window_size_32_pl_mean = np.mean(window_size_32_pl[:, metric_dict['npv']])
npv_window_size_32_pl_std = np.std(window_size_32_pl[:, metric_dict['npv']])
print('      PL-type 32 npv-mean:%.6f, npv-std:%.6f.'%(npv_window_size_32_pl_mean, npv_window_size_32_pl_std))

npv_window_size_64_pl_mean = np.mean(window_size_64_pl[:, metric_dict['npv']])
npv_window_size_64_pl_std = np.std(window_size_64_pl[:, metric_dict['npv']])
print('      PL-type 64 npv-mean:%.6f, npv-std:%.6f.'%(npv_window_size_64_pl_mean, npv_window_size_64_pl_std))

npv_window_size_128_pl_mean = np.mean(window_size_128_pl[:, metric_dict['npv']])
npv_window_size_128_pl_std = np.std(window_size_128_pl[:, metric_dict['npv']])
print('      PL-type 128 npv-mean:%.6f, npv-std:%.6f.'%(npv_window_size_128_pl_mean, npv_window_size_128_pl_std))

npv_window_size_256_pl_mean = np.mean(window_size_256_pl[:, metric_dict['npv']])
npv_window_size_256_pl_std = np.std(window_size_256_pl[:, metric_dict['npv']])
print('      PL-type 256 npv-mean:%.6f, npv-std:%.6f.'%(npv_window_size_256_pl_mean, npv_window_size_256_pl_std))



#TNR
tnr_window_size_32_p_mean = np.mean(window_size_32_p[:, metric_dict['tnr']])
tnr_window_size_32_p_std = np.std(window_size_32_p[:, metric_dict['tnr']])
print('      P-type 32 tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_window_size_32_p_mean, tnr_window_size_32_p_std))

tnr_window_size_64_p_mean = np.mean(window_size_64_p[:, metric_dict['tnr']])
tnr_window_size_64_p_std = np.std(window_size_64_p[:, metric_dict['tnr']])
print('      P-type 64 tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_window_size_64_p_mean, tnr_window_size_64_p_std))

tnr_window_size_128_p_mean = np.mean(window_size_128_p[:, metric_dict['tnr']])
tnr_window_size_128_p_std = np.std(window_size_128_p[:, metric_dict['tnr']])
print('      P-type 128 tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_window_size_128_p_mean, tnr_window_size_128_p_std))

tnr_window_size_256_p_mean = np.mean(window_size_256_p[:, metric_dict['tnr']])
tnr_window_size_256_p_std = np.std(window_size_256_p[:, metric_dict['tnr']])
print('      P-type 256 tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_window_size_256_p_mean, tnr_window_size_256_p_std))

tnr_window_size_32_pl_mean = np.mean(window_size_32_pl[:, metric_dict['tnr']])
tnr_window_size_32_pl_std = np.std(window_size_32_pl[:, metric_dict['tnr']])
print('      PL-type 32 tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_window_size_32_pl_mean, tnr_window_size_32_pl_std))

tnr_window_size_64_pl_mean = np.mean(window_size_64_pl[:, metric_dict['tnr']])
tnr_window_size_64_pl_std = np.std(window_size_64_pl[:, metric_dict['tnr']])
print('      PL-type 64 tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_window_size_64_pl_mean, tnr_window_size_64_pl_std))

tnr_window_size_128_pl_mean = np.mean(window_size_128_pl[:, metric_dict['tnr']])
tnr_window_size_128_pl_std = np.std(window_size_128_pl[:, metric_dict['tnr']])
print('      PL-type 128 tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_window_size_128_pl_mean, tnr_window_size_128_pl_std))

tnr_window_size_256_pl_mean = np.mean(window_size_256_pl[:, metric_dict['tnr']])
tnr_window_size_256_pl_std = np.std(window_size_256_pl[:, metric_dict['tnr']])
print('      PL-type 256 tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_window_size_256_pl_mean, tnr_window_size_256_pl_std))



#ACC
acc_window_size_32_p_mean = np.mean(window_size_32_p[:, metric_dict['acc']])
acc_window_size_32_p_std = np.std(window_size_32_p[:, metric_dict['acc']])
print('      P-type 32 acc-mean:%.6f, acc-std:%.6f.'%(acc_window_size_32_p_mean, acc_window_size_32_p_std))

acc_window_size_64_p_mean = np.mean(window_size_64_p[:, metric_dict['acc']])
acc_window_size_64_p_std = np.std(window_size_64_p[:, metric_dict['acc']])
print('      P-type 64 acc-mean:%.6f, acc-std:%.6f.'%(acc_window_size_64_p_mean, acc_window_size_64_p_std))

acc_window_size_128_p_mean = np.mean(window_size_128_p[:, metric_dict['acc']])
acc_window_size_128_p_std = np.std(window_size_128_p[:, metric_dict['acc']])
print('      P-type 128 acc-mean:%.6f, acc-std:%.6f.'%(acc_window_size_128_p_mean, acc_window_size_128_p_std))

acc_window_size_256_p_mean = np.mean(window_size_256_p[:, metric_dict['acc']])
acc_window_size_256_p_std = np.std(window_size_256_p[:, metric_dict['acc']])
print('      P-type 256 acc-mean:%.6f, acc-std:%.6f.'%(acc_window_size_256_p_mean, acc_window_size_256_p_std))


acc_window_size_32_pl_mean = np.mean(window_size_32_pl[:, metric_dict['acc']])
acc_window_size_32_pl_std = np.std(window_size_32_pl[:, metric_dict['acc']])
print('      PL-type 32 acc-mean:%.6f, acc-std:%.6f.'%(acc_window_size_32_pl_mean, acc_window_size_32_pl_std))

acc_window_size_64_pl_mean = np.mean(window_size_64_pl[:, metric_dict['acc']])
acc_window_size_64_pl_std = np.std(window_size_64_pl[:, metric_dict['acc']])
print('      PL-type 64 acc-mean:%.6f, acc-std:%.6f.'%(acc_window_size_64_pl_mean, acc_window_size_64_pl_std))

acc_window_size_128_pl_mean = np.mean(window_size_128_pl[:, metric_dict['acc']])
acc_window_size_128_pl_std = np.std(window_size_128_pl[:, metric_dict['acc']])
print('      PL-type 128 acc-mean:%.6f, acc-std:%.6f.'%(acc_window_size_128_pl_mean, acc_window_size_128_pl_std))

acc_window_size_256_pl_mean = np.mean(window_size_256_pl[:, metric_dict['acc']])
acc_window_size_256_pl_std = np.std(window_size_256_pl[:, metric_dict['acc']])
print('      PL-type 256 acc-mean:%.6f, acc-std:%.6f.'%(acc_window_size_256_pl_mean, acc_window_size_256_pl_std))


# f1
f1_window_size_32_p_mean = np.mean(window_size_32_p[:, metric_dict['f1']])
f1_window_size_32_p_std = np.std(window_size_32_p[:, metric_dict['f1']])
print('      P-type 32 f1-mean:%.6f, f1-std:%.6f.'%(f1_window_size_32_p_mean, f1_window_size_32_p_std))

f1_window_size_64_p_mean = np.mean(window_size_64_p[:, metric_dict['f1']])
f1_window_size_64_p_std = np.std(window_size_64_p[:, metric_dict['f1']])
print('      P-type 64 f1-mean:%.6f, f1-std:%.6f.'%(f1_window_size_64_p_mean, f1_window_size_64_p_std))

f1_window_size_128_p_mean = np.mean(window_size_128_p[:, metric_dict['f1']])
f1_window_size_128_p_std = np.std(window_size_128_p[:, metric_dict['f1']])
print('      P-type 128 f1-mean:%.6f, f1-std:%.6f.'%(f1_window_size_128_p_mean, f1_window_size_128_p_std))


f1_window_size_256_p_mean = np.mean(window_size_256_p[:, metric_dict['f1']])
f1_window_size_256_p_std = np.std(window_size_256_p[:, metric_dict['f1']])
print('      P-type 256 f1-mean:%.6f, f1-std:%.6f.'%(f1_window_size_256_p_mean, f1_window_size_256_p_std))

f1_window_size_32_pl_mean = np.mean(window_size_32_pl[:, metric_dict['f1']])
f1_window_size_32_pl_std = np.std(window_size_32_pl[:, metric_dict['f1']])
print('      PL-type 32 f1-mean:%.6f, f1-std:%.6f.'%(f1_window_size_32_pl_mean, f1_window_size_32_pl_std))

f1_window_size_64_pl_mean = np.mean(window_size_64_pl[:, metric_dict['f1']])
f1_window_size_64_pl_std = np.std(window_size_64_pl[:, metric_dict['f1']])
print('      PL-type 64 f1-mean:%.6f, f1-std:%.6f.'%(f1_window_size_64_pl_mean, f1_window_size_64_pl_std))


f1_window_size_128_pl_mean = np.mean(window_size_128_pl[:, metric_dict['f1']])
f1_window_size_128_pl_std = np.std(window_size_128_pl[:, metric_dict['f1']])
print('      PL-type 128 f1-mean:%.6f, f1-std:%.6f.'%(f1_window_size_128_pl_mean, f1_window_size_128_pl_std))

f1_window_size_256_pl_mean = np.mean(window_size_256_pl[:, metric_dict['f1']])
f1_window_size_256_pl_std = np.std(window_size_256_pl[:, metric_dict['f1']])
print('      PL-type 256 f1-mean:%.6f, f1-std:%.6f.'%(f1_window_size_256_pl_mean, f1_window_size_256_pl_std))


# mcc
mcc_window_size_32_p_mean = np.mean(window_size_32_p[:, metric_dict['mcc']])
mcc_window_size_32_p_std = np.std(window_size_32_p[:, metric_dict['mcc']])
print('      P-type 32 mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_window_size_32_p_mean, mcc_window_size_32_p_std))

mcc_window_size_64_p_mean = np.mean(window_size_64_p[:, metric_dict['mcc']])
mcc_window_size_64_p_std = np.std(window_size_64_p[:, metric_dict['mcc']])
print('      P-type 64 mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_window_size_64_p_mean, mcc_window_size_64_p_std))

mcc_window_size_128_p_mean = np.mean(window_size_128_p[:, metric_dict['mcc']])
mcc_window_size_128_p_std = np.std(window_size_128_p[:, metric_dict['mcc']])
print('      P-type 128 mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_window_size_128_p_mean, mcc_window_size_128_p_std))

mcc_window_size_256_p_mean = np.mean(window_size_256_p[:, metric_dict['mcc']])
mcc_window_size_256_p_std = np.std(window_size_256_p[:, metric_dict['mcc']])
print('      P-type 256 mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_window_size_256_p_mean, mcc_window_size_256_p_std))

mcc_window_size_32_pl_mean = np.mean(window_size_32_pl[:, metric_dict['mcc']])
mcc_window_size_32_pl_std = np.std(window_size_32_pl[:, metric_dict['mcc']])
print('      PL-type 32 mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_window_size_32_pl_mean, mcc_window_size_32_pl_std))

mcc_window_size_64_pl_mean = np.mean(window_size_64_pl[:, metric_dict['mcc']])
mcc_window_size_64_pl_std = np.std(window_size_64_pl[:, metric_dict['mcc']])
print('      PL-type 64 mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_window_size_64_pl_mean, mcc_window_size_64_pl_std))

mcc_window_size_128_pl_mean = np.mean(window_size_128_pl[:, metric_dict['mcc']])
mcc_window_size_128_pl_std = np.std(window_size_128_pl[:, metric_dict['mcc']])
print('      PL-type 128 mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_window_size_128_pl_mean, mcc_window_size_128_pl_std))

mcc_window_size_256_pl_mean = np.mean(window_size_256_pl[:, metric_dict['mcc']])
mcc_window_size_256_pl_std = np.std(window_size_256_pl[:, metric_dict['mcc']])
print('      PL-type 256 mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_window_size_256_pl_mean, mcc_window_size_256_pl_std))



'''
重组数据
'''
w_s_32_mean_p = [tpr_window_size_32_p_mean, ppv_window_size_32_p_mean, npv_window_size_32_p_mean, tnr_window_size_32_p_mean, acc_window_size_32_p_mean, f1_window_size_32_p_mean, mcc_window_size_32_p_mean]
w_s_32_std_p = [tpr_window_size_32_p_std, ppv_window_size_32_p_std, npv_window_size_32_p_std, tnr_window_size_32_p_std, acc_window_size_32_p_std, f1_window_size_32_p_std, mcc_window_size_32_p_std]

w_s_64_mean_p = [tpr_window_size_64_p_mean, ppv_window_size_64_p_mean, npv_window_size_64_p_mean, tnr_window_size_64_p_mean, acc_window_size_64_p_mean, f1_window_size_64_p_mean, mcc_window_size_64_p_mean]
w_s_64_std_p = [tpr_window_size_64_p_std, ppv_window_size_64_p_std, npv_window_size_64_p_std, tnr_window_size_64_p_std, acc_window_size_64_p_std, f1_window_size_64_p_std, mcc_window_size_64_p_std]

w_s_128_mean_p = [tpr_window_size_128_p_mean, ppv_window_size_128_p_mean, npv_window_size_128_p_mean, tnr_window_size_128_p_mean, acc_window_size_128_p_mean, f1_window_size_128_p_mean, mcc_window_size_128_p_mean]
w_s_128_std_p = [tpr_window_size_128_p_std, ppv_window_size_128_p_std, npv_window_size_128_p_std, tnr_window_size_128_p_std, acc_window_size_128_p_std, f1_window_size_128_p_std, mcc_window_size_128_p_std]

w_s_256_mean_p = [tpr_window_size_256_p_mean, ppv_window_size_256_p_mean, npv_window_size_256_p_mean, tnr_window_size_256_p_mean, acc_window_size_256_p_mean, f1_window_size_256_p_mean, mcc_window_size_256_p_mean]
w_s_256_std_p = [tpr_window_size_256_p_std, ppv_window_size_256_p_std, npv_window_size_256_p_std, tnr_window_size_256_p_std, acc_window_size_256_p_std, f1_window_size_256_p_std, mcc_window_size_256_p_std]

w_s_32_mean_pl = [tpr_window_size_32_pl_mean, ppv_window_size_32_pl_mean, npv_window_size_32_pl_mean, tnr_window_size_32_pl_mean, acc_window_size_32_pl_mean, f1_window_size_32_pl_mean, mcc_window_size_32_pl_mean]
w_s_32_std_pl = [tpr_window_size_32_pl_std, ppv_window_size_32_pl_std, npv_window_size_32_pl_std, tnr_window_size_32_pl_std, acc_window_size_32_pl_std, f1_window_size_32_pl_std, mcc_window_size_32_pl_std]

w_s_64_mean_pl = [tpr_window_size_64_pl_mean, ppv_window_size_64_pl_mean, npv_window_size_64_pl_mean, tnr_window_size_64_pl_mean, acc_window_size_64_pl_mean, f1_window_size_64_pl_mean, mcc_window_size_64_pl_mean]
w_s_64_std_pl = [tpr_window_size_64_pl_std, ppv_window_size_64_pl_std, npv_window_size_64_pl_std, tnr_window_size_64_pl_std, acc_window_size_64_pl_std, f1_window_size_64_pl_std, mcc_window_size_64_pl_std]

w_s_128_mean_pl = [tpr_window_size_128_pl_mean, ppv_window_size_128_pl_mean, npv_window_size_128_pl_mean, tnr_window_size_128_pl_mean, acc_window_size_128_pl_mean, f1_window_size_128_pl_mean, mcc_window_size_128_pl_mean]
w_s_128_std_pl = [tpr_window_size_128_pl_std, ppv_window_size_128_pl_std, npv_window_size_128_pl_std, tnr_window_size_128_pl_std, acc_window_size_128_pl_std, f1_window_size_128_pl_std, mcc_window_size_128_pl_std]

w_s_256_mean_pl = [tpr_window_size_256_pl_mean, ppv_window_size_256_pl_mean, npv_window_size_256_pl_mean, tnr_window_size_256_pl_mean, acc_window_size_256_pl_mean, f1_window_size_256_pl_mean, mcc_window_size_256_pl_mean]
w_s_256_std_pl = [tpr_window_size_256_pl_std, ppv_window_size_256_pl_std, npv_window_size_256_pl_std, tnr_window_size_256_pl_std, acc_window_size_256_pl_std, f1_window_size_256_pl_std, mcc_window_size_256_p_std]




import matplotlib.pyplot as plt

window_size = ['32',"64",'128','256']

acc_p_mean = [acc_window_size_32_p_mean*100, acc_window_size_64_p_mean*100, acc_window_size_128_p_mean*100, acc_window_size_256_p_mean*100]
acc_p_std = [acc_window_size_32_p_std*0, acc_window_size_64_p_std*0, acc_window_size_128_p_std*0, acc_window_size_256_p_std*0]
acc_pl_mean = [acc_window_size_32_pl_mean*100, acc_window_size_64_pl_mean*100, acc_window_size_128_pl_mean*100, acc_window_size_256_pl_mean*100]
acc_pl_std = [acc_window_size_32_pl_std*0, acc_window_size_64_pl_std*0, acc_window_size_128_pl_std*0, acc_window_size_256_pl_std*0]

f1_p_mean = [f1_window_size_32_p_mean*100, f1_window_size_64_p_mean*100, f1_window_size_128_p_mean*100, f1_window_size_256_p_mean*100]
f1_p_std = [f1_window_size_32_p_std*0, f1_window_size_64_p_std*0, f1_window_size_128_p_std*0, f1_window_size_256_p_std*0]
f1_pl_mean = [f1_window_size_32_pl_mean*100, f1_window_size_64_pl_mean*100, f1_window_size_128_pl_mean*100, f1_window_size_256_pl_mean*100]
f1_pl_std = [f1_window_size_32_pl_std*0, f1_window_size_64_pl_std*0, f1_window_size_128_pl_std*0, f1_window_size_256_pl_std*0]

#recall=tpr
tpr_p_mean = [tpr_window_size_32_p_mean*100, tpr_window_size_64_p_mean*100, tpr_window_size_128_p_mean*100, tpr_window_size_256_p_mean*100]
tpr_p_std = [tpr_window_size_32_p_std*0, tpr_window_size_64_p_std*0, tpr_window_size_128_p_std*0, tpr_window_size_256_p_std*0]
tpr_pl_mean = [tpr_window_size_32_pl_mean*100, tpr_window_size_64_pl_mean*100, tpr_window_size_128_pl_mean*100, tpr_window_size_256_pl_mean*100]
tpr_pl_std = [tpr_window_size_32_pl_std*0, tpr_window_size_64_pl_std*0, tpr_window_size_128_pl_std*0, tpr_window_size_256_pl_std*0]

#presion=ppv
ppv_p_mean = [ppv_window_size_32_p_mean*100, ppv_window_size_64_p_mean*100, ppv_window_size_128_p_mean*100, ppv_window_size_256_p_mean*100]
ppv_p_std = [ppv_window_size_32_p_std*0, ppv_window_size_64_p_std*0, ppv_window_size_128_p_std*0, ppv_window_size_256_p_std*0]
ppv_pl_mean = [ppv_window_size_32_pl_mean*100, ppv_window_size_64_pl_mean*100, ppv_window_size_128_pl_mean*100, ppv_window_size_256_pl_mean*100]
ppv_pl_std = [ppv_window_size_32_pl_std*0, ppv_window_size_64_pl_std*0, ppv_window_size_128_pl_std*0, ppv_window_size_256_pl_std*0]

#specificity=tnr
tnr_p_mean = [tnr_window_size_32_p_mean*100, tnr_window_size_64_p_mean*100, tnr_window_size_128_p_mean*100, tnr_window_size_256_p_mean*100]
tnr_p_std = [tnr_window_size_32_p_std*0, tnr_window_size_64_p_std*0, tnr_window_size_128_p_std*0, tnr_window_size_256_p_std*0]
tnr_pl_mean = [tnr_window_size_32_pl_mean*100, tnr_window_size_64_pl_mean*100, tnr_window_size_128_pl_mean*100, tnr_window_size_256_pl_mean*100]
tnr_pl_std = [tnr_window_size_32_pl_std*0, tnr_window_size_64_pl_std*0, tnr_window_size_128_pl_std*0, tnr_window_size_256_pl_std*0]

#mcc
mcc_p_mean = [mcc_window_size_32_p_mean*100, mcc_window_size_64_p_mean*100, mcc_window_size_128_p_mean*100, mcc_window_size_256_p_mean*100]
mcc_p_std = [mcc_window_size_32_p_std*0, mcc_window_size_64_p_std*0, mcc_window_size_128_p_std*0, mcc_window_size_256_p_std*0]
mcc_pl_mean = [mcc_window_size_32_pl_mean*100, mcc_window_size_64_pl_mean*100, mcc_window_size_128_pl_mean*100, mcc_window_size_256_pl_mean*100]
mcc_pl_std = [mcc_window_size_32_pl_std*0, mcc_window_size_64_pl_std*0, mcc_window_size_128_pl_std*0, mcc_window_size_256_pl_std*0]

#NPV
npv_p_mean = [npv_window_size_32_p_mean*100, npv_window_size_64_p_mean*100, npv_window_size_128_p_mean*100, npv_window_size_256_p_mean*100]
npv_p_std = [npv_window_size_32_p_std*0, npv_window_size_64_p_std*0, npv_window_size_128_p_std*0, npv_window_size_256_p_std*0]
npv_pl_mean = [npv_window_size_32_pl_mean*100, npv_window_size_64_pl_mean*100, npv_window_size_128_pl_mean*100, npv_window_size_256_pl_mean*100]
npv_pl_std = [npv_window_size_32_pl_std*0, npv_window_size_64_pl_std*0, npv_window_size_128_pl_std*0, npv_window_size_256_pl_std*0]

# fig=plt.figure(figsize=(10,8))
# my_y_ticks = np.arange(70, 100,5)
# plt.yticks(my_y_ticks)
# plt.errorbar(window_size, acc_p_mean, yerr=acc_p_std, color='#266A2E', marker='s', lw=3, ecolor='#266A2E', elinewidth=2,ms=15,capsize=6)
# plt.errorbar(window_size, acc_pl_mean, yerr=acc_pl_std, color='#f07818', marker='s', lw=3, ecolor='#f07818', elinewidth=2,ms=15,capsize=6)
# plt.tick_params(labelsize=20)
# plt.xlabel('Window Size', fontsize=25)
# plt.ylabel('Accuracy(in %)', fontsize=25)
# plt.savefig("D:/Projects/Papers/Parkinson/temporal_window_acc.png", dpi=500, bbox_inches='tight')
# plt.show()
#
# fig=plt.figure(figsize=(10,8))
# my_y_ticks = np.arange(70, 100,5)
# plt.yticks(my_y_ticks)
# plt.errorbar(window_size, f1_p_mean, yerr=f1_p_std, color='#266A2E', marker='s', lw=3, ecolor='#266A2E', elinewidth=2,ms=15,capsize=6)
# plt.errorbar(window_size, f1_pl_mean, yerr=f1_pl_std, color='#f07818', marker='s', lw=3, ecolor='#f07818', elinewidth=2,ms=15,capsize=6)
# plt.tick_params(labelsize=20)
# plt.xlabel('Window Size', fontsize=25)
# plt.ylabel('F1 score(in %)'.translate(sub_map), fontsize=25)
# plt.savefig("D:/Projects/Papers/Parkinson/temporal_window_f1.png", dpi=500, bbox_inches='tight')
# plt.show()
#
# fig=plt.figure(figsize=(10,8))
# my_y_ticks = np.arange(70, 100,5)
# plt.yticks(my_y_ticks)
# plt.errorbar(window_size, tpr_p_mean, yerr=tpr_p_std, color='#266A2E', marker='s', lw=3, ecolor='#266A2E', elinewidth=2,ms=15,capsize=6)
# plt.errorbar(window_size, tpr_pl_mean, yerr=tpr_pl_std, color='#f07818', marker='s', lw=3, ecolor='#f07818', elinewidth=2,ms=15,capsize=6)
# plt.tick_params(labelsize=20)
# plt.xlabel('Window Size', fontsize=25)
# plt.ylabel('Recall(in %)'.translate(sub_map), fontsize=25)
# plt.savefig("D:/Projects/Papers/Parkinson/temporal_window_recall.png", dpi=500, bbox_inches='tight')
# # plt.show()

# fig=plt.figure(figsize=(10,8))
# my_y_ticks = np.arange(70, 100,5)
# plt.yticks(my_y_ticks)
# plt.errorbar(window_size, ppv_p_mean, yerr=ppv_p_std, color='#266A2E', marker='s', lw=3, ecolor='#266A2E', elinewidth=2,ms=15,capsize=6)
# plt.errorbar(window_size, ppv_pl_mean, yerr=ppv_pl_std, color='#f07818', marker='s', lw=3, ecolor='#f07818', elinewidth=2,ms=15,capsize=6)
# plt.tick_params(labelsize=20)
# plt.xlabel('Window Size', fontsize=25)
# plt.ylabel('Precision(%)'.translate(sub_map), fontsize=25)
# plt.savefig("D:/Projects/Papers/Parkinson/temporal_window_presion.png", dpi=500, bbox_inches='tight')
# plt.show()

# fig=plt.figure(figsize=(10,8))
# my_y_ticks = np.arange(70, 100,5)
# plt.yticks(my_y_ticks)
# plt.errorbar(window_size, tnr_p_mean, yerr=tnr_p_std, color='#266A2E', marker='s', lw=3, ecolor='#266A2E', elinewidth=2,ms=15,capsize=6)
# plt.errorbar(window_size, tnr_pl_mean, yerr=tnr_pl_std, color='#f07818', marker='s', lw=3, ecolor='#f07818', elinewidth=2,ms=15,capsize=6)
# plt.tick_params(labelsize=20)
# plt.xlabel('Window Size', fontsize=25)
# plt.ylabel('Specificity(%)'.translate(sub_map), fontsize=25)
# plt.savefig("D:/Projects/Papers/Parkinson/temporal_window_specificity.png", dpi=500, bbox_inches='tight')
# # plt.show()
# #
fig=plt.figure(figsize=(10,8))
my_y_ticks = np.arange(70, 100,5)
plt.yticks(my_y_ticks)
plt.errorbar(window_size, mcc_p_mean, yerr=mcc_p_std, color='#266A2E', marker='s', lw=3, ecolor='#266A2E', elinewidth=2,ms=15,capsize=6)
plt.errorbar(window_size, mcc_pl_mean, yerr=mcc_pl_std, color='#f07818', marker='s', lw=3, ecolor='#f07818', elinewidth=2,ms=15,capsize=6)
plt.tick_params(labelsize=20)
plt.xlabel('Window Size', fontsize=25)
plt.ylabel('MCC(in %)'.translate(sub_map), fontsize=25)
# plt.savefig("D:/Projects/Papers/Parkinson/temporal_window_mcc.png", dpi=500, bbox_inches='tight')
plt.show()
#
# fig=plt.figure(figsize=(10,8))
# my_y_ticks = np.arange(70, 100,5)
# plt.yticks(my_y_ticks)
# plt.errorbar(window_size, npv_p_mean, yerr=npv_p_std, color='#266A2E', marker='s', lw=3, ecolor='#266A2E', elinewidth=2,ms=15,capsize=6)
# plt.errorbar(window_size, npv_pl_mean, yerr=npv_pl_std, color='#f07818', marker='s', lw=3, ecolor='#f07818', elinewidth=2,ms=15,capsize=6)
# plt.tick_params(labelsize=20)
# plt.xlabel('Window Size', fontsize=25)
# plt.ylabel('NPV(%)'.translate(sub_map), fontsize=25)
# plt.savefig("D:/Projects/Papers/Parkinson/temporal_window_npv.png", dpi=500, bbox_inches='tight')
# # plt.show()





























