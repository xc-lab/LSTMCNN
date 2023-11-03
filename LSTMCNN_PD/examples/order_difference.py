'''
一阶差分的作用
'''
import numpy as np

# P-type
none_order_p = np.array([
    [0.730769, 0.695652, 0.727273, 0.666667, 0.785714, 0.733333, 0.456475],
    [0.807692, 0.782609, 0.818182, 0.750000, 0.857143, 0.800000, 0.612637],
    [0.769231, 0.727273, 0.727273, 0.727273, 0.800000, 0.800000, 0.527273],
    [0.730769, 0.720000, 0.818182, 0.642857, 0.833333, 0.666667, 0.480500],
    [0.769231, 0.750000, 0.818182, 0.692308, 0.846154, 0.733333, 0.544949]
])

a_order_p = np.array([
    [0.884615, 0.857143, 0.818182, 0.900000, 0.875000, 0.933333, 0.763167],
    [0.846154, 0.833333, 0.909091, 0.769231, 0.923077, 0.800000, 0.700649],
    [0.846154, 0.833333, 0.909091, 0.769231, 0.923077, 0.800000, 0.700649],
    [0.923077, 0.909091, 0.909091, 0.909091, 0.933333, 0.933333, 0.842424],
    [0.923077, 0.909091, 0.909091, 0.909091, 0.933333, 0.933333, 0.842424],
])

l_order_p = np.array([
    [0.807692, 0.782609, 0.818182, 0.750000, 0.857143, 0.800000, 0.612637],
    [0.807692, 0.761905, 0.727273, 0.800000, 0.812500, 0.866667, 0.603148],
    [0.846154, 0.818182, 0.818182, 0.818182, 0.866667, 0.866667, 0.684848],
    [0.846154, 0.818182, 0.818182, 0.818182, 0.866667, 0.866667, 0.684848],
    [0.807692, 0.782609, 0.818182, 0.750000, 0.857143, 0.800000, 0.612637],
])

p_order_p = np.array([
    [0.846154, 0.833333, 0.909091, 0.769231, 0.923077, 0.800000, 0.700649],
    [0.807692, 0.782609, 0.818182, 0.750000, 0.857143, 0.800000, 0.612637],
    [0.769231, 0.750000, 0.818182, 0.692308, 0.846154, 0.733333, 0.544949],
    [0.807692, 0.782609, 0.818182, 0.750000, 0.857143, 0.800000, 0.612637],
    [0.769231, 0.750000, 0.818182, 0.692308, 0.846154, 0.733333, 0.544949],
])

xy_order_p = np.array([
    [0.961538, 0.956522, 1.000000, 0.916667, 1.000000, 0.933333, 0.924962],
    [0.961538, 0.952381, 0.909091, 1.000000, 0.937500, 1.000000, 0.923168],
    [0.961538, 0.952381, 0.909091, 1.000000, 0.937500, 1.000000, 0.923168],
    [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000],
    [0.923077, 0.909091, 0.909091, 0.909091, 0.933333, 0.933333, 0.842424],
])

# PL-type
none_order_pl = np.array([
    [0.760000, 0.769231, 0.909091, 0.666667, 0.900000, 0.642857, 0.559259],
    [0.720000, 0.740741, 0.909091, 0.625000, 0.888889, 0.571429, 0.496924],
    [0.760000, 0.785714, 1.000000, 0.647059, 1.000000, 0.571429, 0.608069],
    [0.680000, 0.714286, 0.909091, 0.588235, 0.875000, 0.500000, 0.435322],
    [0.680000, 0.714286, 0.909091, 0.588235, 0.875000, 0.500000, 0.435322],
])

a_order_pl = np.array([
    [0.760000, 0.769231, 0.909091, 0.666667, 0.900000, 0.642857, 0.559259],
    [0.760000, 0.750000, 0.818182, 0.692308, 0.833333, 0.714286, 0.529043],
    [0.800000, 0.782609, 0.818182, 0.750000, 0.846154, 0.785714, 0.600012],
    [0.800000, 0.814815, 1.000000, 0.687500, 1.000000, 0.642857, 0.664804],
    [0.840000, 0.833333, 0.909091, 0.769231, 0.916667, 0.785714, 0.690337],
])

l_order_pl = np.array([
    [0.920000, 0.916667, 1.000000, 0.846154, 1.000000, 0.857143, 0.851631],
    [0.920000, 0.916667, 1.000000, 0.846154, 1.000000, 0.857143, 0.851631],
    [0.880000, 0.880000, 1.000000, 0.785714, 1.000000, 0.785714, 0.785714],
    [0.920000, 0.916667, 1.000000, 0.846154, 1.000000, 0.857143, 0.851631],
    [0.880000, 0.880000, 1.000000, 0.785714, 1.000000, 0.785714, 0.785714],
])

p_order_pl = np.array([
    [0.800000, 0.814815, 1.000000, 0.687500, 1.000000, 0.642857, 0.664804],
    [0.760000, 0.785714, 1.000000, 0.647059, 1.000000, 0.571429, 0.608069],
    [0.760000, 0.785714, 1.000000, 0.647059, 1.000000, 0.571429, 0.608069],
    [0.840000, 0.846154, 1.000000, 0.733333, 1.000000, 0.714286, 0.723747],
    [0.760000, 0.785714, 1.000000, 0.647059, 1.000000, 0.571429, 0.608069],
])

xy_order_pl = np.array([
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132],
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132],
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132],
    [0.960000, 0.952381, 0.909091, 1.000000, 0.933333, 1.000000, 0.921132],
    [0.920000, 0.900000, 0.818182, 1.000000, 0.875000, 1.000000, 0.846114]
])


'''
取数据
'''
metric_dict = {'acc':0, 'f1':1, 'tpr':2, 'ppv':3, 'npv':4, 'tnr':5, 'mcc':6}

#P-type none order
tpr_none_p_mean = np.mean(none_order_p[:, metric_dict['tpr']])
tpr_none_p_std  = np.std(none_order_p[:, metric_dict['tpr']])
print('##### P-type none-order tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_none_p_mean, tpr_none_p_std))

ppv_none_p_mean = np.mean(none_order_p[:, metric_dict['ppv']])
ppv_none_p_std  = np.std(none_order_p[:, metric_dict['ppv']])
print('      P-type none-order ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_none_p_mean, ppv_none_p_std))

npv_none_p_mean = np.mean(none_order_p[:, metric_dict['npv']])
npv_none_p_std  = np.std(none_order_p[:, metric_dict['npv']])
print('      P-type none-order npv-mean:%.6f, npv-std:%.6f.'%(npv_none_p_mean, npv_none_p_std))

tnr_none_p_mean = np.mean(none_order_p[:, metric_dict['tnr']])
tnr_none_p_std  = np.std(none_order_p[:, metric_dict['tnr']])
print('      P-type none-order tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_none_p_mean, tnr_none_p_std))

f1_none_p_mean = np.mean(none_order_p[:, metric_dict['f1']])
f1_none_p_std  = np.std(none_order_p[:, metric_dict['f1']])
print('      P-type none-order f1-mean:%.6f, f1-std:%.6f.'%(f1_none_p_mean, f1_none_p_std))

acc_none_p_mean = np.mean(none_order_p[:, metric_dict['acc']])
acc_none_p_std  = np.std(none_order_p[:, metric_dict['acc']])
print('      P-type none-order acc-mean:%.6f, acc-std:%.6f.'%(acc_none_p_mean, acc_none_p_std))

mcc_none_p_mean = np.mean(none_order_p[:, metric_dict['mcc']])
mcc_none_p_std  = np.std(none_order_p[:, metric_dict['mcc']])
print('      P-type none-order mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_none_p_mean, mcc_none_p_std))

# P-type a order
tpr_a_p_mean = np.mean(a_order_p[:, metric_dict['tpr']])
tpr_a_p_std  = np.std(a_order_p[:, metric_dict['tpr']])
print('##### P-type a-order tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_a_p_mean, tpr_a_p_std))

ppv_a_p_mean = np.mean(a_order_p[:, metric_dict['ppv']])
ppv_a_p_std  = np.std(a_order_p[:, metric_dict['ppv']])
print('      P-type a-order ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_a_p_mean, ppv_a_p_std))

npv_a_p_mean = np.mean(a_order_p[:, metric_dict['npv']])
npv_a_p_std  = np.std(a_order_p[:, metric_dict['npv']])
print('      P-type a-order npv-mean:%.6f, npv-std:%.6f.'%(npv_a_p_mean, npv_a_p_std))

tnr_a_p_mean = np.mean(a_order_p[:, metric_dict['tnr']])
tnr_a_p_std  = np.std(a_order_p[:, metric_dict['tnr']])
print('      P-type a-order tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_a_p_mean, tnr_a_p_std))

f1_a_p_mean = np.mean(a_order_p[:, metric_dict['f1']])
f1_a_p_std  = np.std(a_order_p[:, metric_dict['f1']])
print('      P-type a-order f1-mean:%.6f, f1-std:%.6f.'%(f1_a_p_mean, f1_a_p_std))

acc_a_p_mean = np.mean(a_order_p[:, metric_dict['acc']])
acc_a_p_std  = np.std(a_order_p[:, metric_dict['acc']])
print('      P-type a-order acc-mean:%.6f, acc-std:%.6f.'%(acc_a_p_mean, acc_a_p_std))

mcc_a_p_mean = np.mean(a_order_p[:, metric_dict['mcc']])
mcc_a_p_std  = np.std(a_order_p[:, metric_dict['mcc']])
print('      P-type a-order mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_a_p_mean, mcc_a_p_std))

#P-type l order
tpr_l_p_mean = np.mean(l_order_p[:, metric_dict['tpr']])
tpr_l_p_std  = np.std(l_order_p[:, metric_dict['tpr']])
print('##### P-type l-order tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_l_p_mean, tpr_l_p_std))

ppv_l_p_mean = np.mean(l_order_p[:, metric_dict['ppv']])
ppv_l_p_std  = np.std(l_order_p[:, metric_dict['ppv']])
print('      P-type l-order ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_l_p_mean, ppv_l_p_std))

npv_l_p_mean = np.mean(l_order_p[:, metric_dict['npv']])
npv_l_p_std  = np.std(l_order_p[:, metric_dict['npv']])
print('      P-type l-order npv-mean:%.6f, npv-std:%.6f.'%(npv_l_p_mean, npv_l_p_std))

tnr_l_p_mean = np.mean(l_order_p[:, metric_dict['tnr']])
tnr_l_p_std  = np.std(l_order_p[:, metric_dict['tnr']])
print('      P-type l-order tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_l_p_mean, tnr_l_p_std))

f1_l_p_mean = np.mean(l_order_p[:, metric_dict['f1']])
f1_l_p_std  = np.std(l_order_p[:, metric_dict['f1']])
print('      P-type l-order f1-mean:%.6f, f1-std:%.6f.'%(f1_l_p_mean, f1_l_p_std))

acc_l_p_mean = np.mean(l_order_p[:, metric_dict['acc']])
acc_l_p_std  = np.std(l_order_p[:, metric_dict['acc']])
print('      P-type l-order acc-mean:%.6f, acc-std:%.6f.'%(acc_l_p_mean, acc_l_p_std))

mcc_l_p_mean = np.mean(l_order_p[:, metric_dict['mcc']])
mcc_l_p_std  = np.std(l_order_p[:, metric_dict['mcc']])
print('      P-type l-order mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_l_p_mean, mcc_l_p_std))


#P-type p order
tpr_p_p_mean = np.mean(p_order_p[:, metric_dict['tpr']])
tpr_p_p_std  = np.std(p_order_p[:, metric_dict['tpr']])
print('##### P-type p-order tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_p_p_mean, tpr_p_p_std))

ppv_p_p_mean = np.mean(p_order_p[:, metric_dict['ppv']])
ppv_p_p_std  = np.std(p_order_p[:, metric_dict['ppv']])
print('      P-type p-order ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_p_p_mean, ppv_p_p_std))

npv_p_p_mean = np.mean(p_order_p[:, metric_dict['npv']])
npv_p_p_std  = np.std(p_order_p[:, metric_dict['npv']])
print('      P-type p-order npv-mean:%.6f, npv-std:%.6f.'%(npv_p_p_mean, npv_p_p_std))

tnr_p_p_mean = np.mean(p_order_p[:, metric_dict['tnr']])
tnr_p_p_std  = np.std(p_order_p[:, metric_dict['tnr']])
print('      P-type p-order tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_p_p_mean, tnr_p_p_std))

f1_p_p_mean = np.mean(p_order_p[:, metric_dict['f1']])
f1_p_p_std  = np.std(p_order_p[:, metric_dict['f1']])
print('      P-type p-order f1-mean:%.6f, f1-std:%.6f.'%(f1_p_p_mean, f1_p_p_std))

acc_p_p_mean = np.mean(p_order_p[:, metric_dict['acc']])
acc_p_p_std  = np.std(p_order_p[:, metric_dict['acc']])
print('      P-type p-order acc-mean:%.6f, acc-std:%.6f.'%(acc_p_p_mean, acc_p_p_std))

mcc_p_p_mean = np.mean(p_order_p[:, metric_dict['mcc']])
mcc_p_p_std  = np.std(p_order_p[:, metric_dict['mcc']])
print('      P-type p-order mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_p_p_mean, mcc_p_p_std))


#P-order x y order
tpr_xy_p_mean = np.mean(xy_order_p[:, metric_dict['tpr']])
tpr_xy_p_std  = np.std(xy_order_p[:, metric_dict['tpr']])
print('##### P-type xy-order tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_xy_p_mean, tpr_xy_p_std))

ppv_xy_p_mean = np.mean(xy_order_p[:, metric_dict['ppv']])
ppv_xy_p_std  = np.std(xy_order_p[:, metric_dict['ppv']])
print('      P-type xy-order ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_xy_p_mean, ppv_xy_p_std))

npv_xy_p_mean = np.mean(xy_order_p[:, metric_dict['npv']])
npv_xy_p_std  = np.std(xy_order_p[:, metric_dict['npv']])
print('      P-type xy-order npv-mean:%.6f, npv-std:%.6f.'%(npv_xy_p_mean, npv_xy_p_std))

tnr_xy_p_mean = np.mean(xy_order_p[:, metric_dict['tnr']])
tnr_xy_p_std  = np.std(xy_order_p[:, metric_dict['tnr']])
print('      P-type xy-order tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_xy_p_mean, tnr_xy_p_std))

f1_xy_p_mean = np.mean(xy_order_p[:, metric_dict['f1']])
f1_xy_p_std  = np.std(xy_order_p[:, metric_dict['f1']])
print('      P-type xy-order f1-mean:%.6f, f1-std:%.6f.'%(f1_xy_p_mean, f1_xy_p_std))

acc_xy_p_mean = np.mean(xy_order_p[:, metric_dict['acc']])
acc_xy_p_std  = np.std(xy_order_p[:, metric_dict['acc']])
print('      P-type xy-order acc-mean:%.6f, acc-std:%.6f.'%(acc_xy_p_mean, acc_xy_p_std))

mcc_xy_p_mean = np.mean(xy_order_p[:, metric_dict['mcc']])
mcc_xy_p_std  = np.std(xy_order_p[:, metric_dict['mcc']])
print('      P-type xy-order mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_xy_p_mean, mcc_xy_p_std))


#PL-type none order
tpr_none_pl_mean = np.mean(none_order_pl[:, metric_dict['tpr']])
tpr_none_pl_std  = np.std(none_order_pl[:, metric_dict['tpr']])
print('##### PL-type none-order tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_none_pl_mean, tpr_none_pl_std))

ppv_none_pl_mean = np.mean(none_order_pl[:, metric_dict['ppv']])
ppv_none_pl_std  = np.std(none_order_pl[:, metric_dict['ppv']])
print('      PL-type none-order ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_none_pl_mean, ppv_none_pl_std))

npv_none_pl_mean = np.mean(none_order_pl[:, metric_dict['npv']])
npv_none_pl_std  = np.std(none_order_pl[:, metric_dict['npv']])
print('      PL-type none-order npv-mean:%.6f, npv-std:%.6f.'%(npv_none_pl_mean, npv_none_pl_std))

tnr_none_pl_mean = np.mean(none_order_pl[:, metric_dict['tnr']])
tnr_none_pl_std  = np.std(none_order_pl[:, metric_dict['tnr']])
print('      PL-type none-order tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_none_pl_mean, tnr_none_pl_std))

f1_none_pl_mean = np.mean(none_order_pl[:, metric_dict['f1']])
f1_none_pl_std  = np.std(none_order_pl[:, metric_dict['f1']])
print('      PL-type none-order f1-mean:%.6f, f1-std:%.6f.'%(f1_none_pl_mean, f1_none_pl_std))

acc_none_pl_mean = np.mean(none_order_pl[:, metric_dict['acc']])
acc_none_pl_std  = np.std(none_order_pl[:, metric_dict['acc']])
print('      PL-type none-order acc-mean:%.6f, acc-std:%.6f.'%(acc_none_pl_mean, acc_none_pl_std))

mcc_none_pl_mean = np.mean(none_order_pl[:, metric_dict['mcc']])
mcc_none_pl_std  = np.std(none_order_pl[:, metric_dict['mcc']])
print('      PL-type none-order mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_none_pl_mean, mcc_none_pl_std))

#PL-type a order
tpr_a_pl_mean = np.mean(a_order_pl[:, metric_dict['tpr']])
tpr_a_pl_std  = np.std(a_order_pl[:, metric_dict['tpr']])
print('##### PL-type a-order tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_a_pl_mean, tpr_a_pl_std))

ppv_a_pl_mean = np.mean(a_order_pl[:, metric_dict['ppv']])
ppv_a_pl_std  = np.std(a_order_pl[:, metric_dict['ppv']])
print('      PL-type a-order ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_a_pl_mean, ppv_a_pl_std))

npv_a_pl_mean = np.mean(a_order_pl[:, metric_dict['npv']])
npv_a_pl_std  = np.std(a_order_pl[:, metric_dict['npv']])
print('      PL-type a-order npv-mean:%.6f, npv-std:%.6f.'%(npv_a_pl_mean, npv_a_pl_std))

tnr_a_pl_mean = np.mean(a_order_pl[:, metric_dict['tnr']])
tnr_a_pl_std  = np.std(a_order_pl[:, metric_dict['tnr']])
print('      PL-type a-order tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_a_pl_mean, tnr_a_pl_std))

f1_a_pl_mean = np.mean(a_order_pl[:, metric_dict['f1']])
f1_a_pl_std  = np.std(a_order_pl[:, metric_dict['f1']])
print('      PL-type a-order f1-mean:%.6f, f1-std:%.6f.'%(f1_a_pl_mean, f1_a_pl_std))

acc_a_pl_mean = np.mean(a_order_pl[:, metric_dict['acc']])
acc_a_pl_std  = np.std(a_order_pl[:, metric_dict['acc']])
print('      PL-type a-order acc-mean:%.6f, acc-std:%.6f.'%(acc_a_pl_mean, acc_a_pl_std))

mcc_a_pl_mean = np.mean(a_order_pl[:, metric_dict['mcc']])
mcc_a_pl_std  = np.std(a_order_pl[:, metric_dict['mcc']])
print('      PL-type a-order mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_a_pl_mean, mcc_a_pl_std))


#PL-type l order
tpr_l_pl_mean = np.mean(l_order_pl[:, metric_dict['tpr']])
tpr_l_pl_std  = np.std(l_order_pl[:, metric_dict['tpr']])
print('##### PL-type l-order tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_l_pl_mean, tpr_l_pl_std))

ppv_l_pl_mean = np.mean(l_order_pl[:, metric_dict['ppv']])
ppv_l_pl_std  = np.std(l_order_pl[:, metric_dict['ppv']])
print('      PL-type l-order ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_l_pl_mean, ppv_l_pl_std))

npv_l_pl_mean = np.mean(l_order_pl[:, metric_dict['npv']])
npv_l_pl_std  = np.std(l_order_pl[:, metric_dict['npv']])
print('      PL-type l-order npv-mean:%.6f, npv-std:%.6f.'%(npv_l_pl_mean, npv_l_pl_std))

tnr_l_pl_mean = np.mean(l_order_pl[:, metric_dict['tnr']])
tnr_l_pl_std  = np.std(l_order_pl[:, metric_dict['tnr']])
print('      PL-type l-order tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_l_pl_mean, tnr_l_pl_std))

f1_l_pl_mean = np.mean(l_order_pl[:, metric_dict['f1']])
f1_l_pl_std  = np.std(l_order_pl[:, metric_dict['f1']])
print('      PL-type l-order f1-mean:%.6f, f1-std:%.6f.'%(f1_l_pl_mean, f1_l_pl_std))

acc_l_pl_mean = np.mean(l_order_pl[:, metric_dict['acc']])
acc_l_pl_std  = np.std(l_order_pl[:, metric_dict['acc']])
print('      PL-type l-order acc-mean:%.6f, acc-std:%.6f.'%(acc_l_pl_mean, acc_l_pl_std))

mcc_l_pl_mean = np.mean(l_order_pl[:, metric_dict['mcc']])
mcc_l_pl_std  = np.std(l_order_pl[:, metric_dict['mcc']])
print('      PL-type l-order mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_l_pl_mean, mcc_l_pl_std))


#PL-type p order
tpr_p_pl_mean = np.mean(p_order_pl[:, metric_dict['tpr']])
tpr_p_pl_std  = np.std(p_order_pl[:, metric_dict['tpr']])
print('##### PL-type p-order tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_p_pl_mean, tpr_p_pl_std))

ppv_p_pl_mean = np.mean(p_order_pl[:, metric_dict['ppv']])
ppv_p_pl_std  = np.std(p_order_pl[:, metric_dict['ppv']])
print('      PL-type p-order ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_p_pl_mean, ppv_p_pl_std))

npv_p_pl_mean = np.mean(p_order_pl[:, metric_dict['npv']])
npv_p_pl_std  = np.std(p_order_pl[:, metric_dict['npv']])
print('      PL-type p-order npv-mean:%.6f, npv-std:%.6f.'%(npv_p_pl_mean, npv_p_pl_std))

tnr_p_pl_mean = np.mean(p_order_pl[:, metric_dict['tnr']])
tnr_p_pl_std  = np.std(p_order_pl[:, metric_dict['tnr']])
print('      PL-type p-order tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_p_pl_mean, tnr_p_pl_std))

f1_p_pl_mean = np.mean(p_order_pl[:, metric_dict['f1']])
f1_p_pl_std  = np.std(p_order_pl[:, metric_dict['f1']])
print('      PL-type p-order f1-mean:%.6f, f1-std:%.6f.'%(f1_p_pl_mean, f1_p_pl_std))

acc_p_pl_mean = np.mean(p_order_pl[:, metric_dict['acc']])
acc_p_pl_std  = np.std(p_order_pl[:, metric_dict['acc']])
print('      PL-type p-order acc-mean:%.6f, acc-std:%.6f.'%(acc_p_pl_mean, acc_p_pl_std))

mcc_p_pl_mean = np.mean(p_order_pl[:, metric_dict['mcc']])
mcc_p_pl_std  = np.std(p_order_pl[:, metric_dict['mcc']])
print('      PL-type p-order mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_p_pl_mean, mcc_p_pl_std))


#PL-type x y order
tpr_xy_pl_mean = np.mean(xy_order_pl[:, metric_dict['tpr']])
tpr_xy_pl_std  = np.std(xy_order_pl[:, metric_dict['tpr']])
print('##### PL-type xy-order tpr-mean:%.6f, tpr-std:%.6f.'%(tpr_xy_pl_mean, tpr_xy_pl_std))

ppv_xy_pl_mean = np.mean(xy_order_pl[:, metric_dict['ppv']])
ppv_xy_pl_std  = np.std(xy_order_pl[:, metric_dict['ppv']])
print('      PL-type xy-order ppv-mean:%.6f, ppv-std:%.6f.'%(ppv_xy_pl_mean, ppv_xy_pl_std))

npv_xy_pl_mean = np.mean(xy_order_pl[:, metric_dict['npv']])
npv_xy_pl_std  = np.std(xy_order_pl[:, metric_dict['npv']])
print('      PL-type xy-order npv-mean:%.6f, npv-std:%.6f.'%(npv_xy_pl_mean, npv_xy_pl_std))

tnr_xy_pl_mean = np.mean(xy_order_pl[:, metric_dict['tnr']])
tnr_xy_pl_std  = np.std(xy_order_pl[:, metric_dict['tnr']])
print('      PL-type xy-order tnr-mean:%.6f, tnr-std:%.6f.'%(tnr_xy_pl_mean, tnr_xy_pl_std))

f1_xy_pl_mean = np.mean(xy_order_pl[:, metric_dict['f1']])
f1_xy_pl_std  = np.std(xy_order_pl[:, metric_dict['f1']])
print('      PL-type xy-order f1-mean:%.6f, f1-std:%.6f.'%(f1_xy_pl_mean, f1_xy_pl_std))

acc_xy_pl_mean = np.mean(xy_order_pl[:, metric_dict['acc']])
acc_xy_pl_std  = np.std(xy_order_pl[:, metric_dict['acc']])
print('      PL-type xy-order acc-mean:%.6f, acc-std:%.6f.'%(acc_xy_pl_mean, acc_xy_pl_std))

mcc_xy_pl_mean = np.mean(xy_order_pl[:, metric_dict['mcc']])
mcc_xy_pl_std  = np.std(xy_order_pl[:, metric_dict['mcc']])
print('      PL-type xy-order mcc-mean:%.6f, mcc-std:%.6f.'%(mcc_xy_pl_mean, mcc_xy_pl_std))


# 画ROC曲线





































