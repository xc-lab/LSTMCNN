import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.stats as st
import json
from data_utils.utils import *
from examples.plot_time_series import *


matplotlib.rcParams.update({'font.size': 12})
scale = [1, 1, 1, 1, 1, 1]
dim_id = [0, 1, 2, 3, 4, 5]
order = 1


path_kt = '../data/raw_data/KT/KT115/20190523-171459-KT115-ptrace_7e02ddc3-9be8-4570-bb28-78f28c86cb95.json'
with open(path_kt) as curr_file:
    test_data_kt = json.load(curr_file)
frame_data_kt = test_data_kt['data']
index = ['a', 'l', 'p', 'x', 'y', 't']
full_data_array_kt = get_full_data_array(frame_data_kt, index)  # 得到数组
data_scale_kt = get_column_data_scale(full_data_array_kt)  #
n_data_kt = normalization_array_data(full_data_array_kt, data_scale_kt)  # 对a,l,p三个特征进行最大最小归一化处理
n_data_kt = get_colomn_scalled_difference(n_data_kt, order, scale, dim_id)
X_kt = np.array(n_data_kt[:, 3])
X_kt = filter_extreme_3sigma(X_kt, 2, 5) #平滑处理
X_kt = X_kt[10:140]



path_pd = '../data/raw_data/PD/PD23/PD-23-20190506-071210-ptrace_dda09d7c-7ca1-48cb-b163-9f837a2c33cf.json'
with open(path_pd) as curr_file:
    test_data_pd = json.load(curr_file)
frame_data_pd = test_data_pd['data']
index = ['a', 'l', 'p', 'x', 'y', 't']
full_data_array_pd = get_full_data_array(frame_data_pd, index)  # 得到数组
data_scale_pd = get_column_data_scale(full_data_array_pd)  #
n_data_pd = normalization_array_data(full_data_array_pd, data_scale_pd)  # 对a,l,p三个特征进行最大最小归一化处理
n_data_pd = get_colomn_scalled_difference(n_data_pd, order, scale, dim_id)
X_pd = np.array(n_data_pd[:, 3])
X_pd = filter_extreme_3sigma(X_pd, 2, 5)
# X_pd = X_pd[21:151]


# plt.rcParams.update({'font.size': 8})
# # generate dataset
# data_points_kt = len(X_kt)
# Mu_kt = X_kt
# Sigma_kt = np.ones(data_points_kt) * 0.001
# data_kt = np.random.normal(loc=Mu_kt, scale=Sigma_kt, size=(100, data_points_kt))
# # predicted expect and calculate confidence interval
# low_CI_bound_kt, high_CI_bound_kt = st.t.interval(0.95, data_points_kt - 1,
#                                             loc=np.mean(data_kt, 0),
#                                             scale=st.sem(data_kt))
# # plot confidence interval
# x_kt = np.linspace(0, data_points_kt - 1, num=data_points_kt)
# plt.plot(Mu_kt, c='#006400', label='HC subject', linewidth=6.0, alpha=1)
# # plt.fill_between(x_kt, low_CI_bound_kt, high_CI_bound_kt, alpha=0.2, color='#006400',
# #                 label='HC confidence interval')


# generate dataset
data_points_pd = len(X_pd)
Mu_pd = X_pd
Sigma_pd = np.ones(data_points_pd) * 0.001
data_pd = np.random.normal(loc=Mu_pd, scale=Sigma_pd, size=(100, data_points_pd))
# predicted expect and calculate confidence interval
low_CI_bound_pd, high_CI_bound_pd = st.t.interval(0.95, data_points_pd - 1,
                                            loc=np.mean(data_pd, 0),
                                            scale=st.sem(data_pd))
# plot confidence interval
x_pd = np.linspace(0, data_points_pd - 1, num=data_points_pd)
plt.plot(Mu_pd, c='#FF8C00', label='PD patient', linewidth=6.0, alpha=0.8)
# plt.fill_between(x_pd, low_CI_bound_pd, high_CI_bound_pd, alpha=0.2, color='#FF8C00',
#                 label='PD confidence interval')

plt.xlabel('Time',fontsize=10)
plt.ylabel('Signal', fontsize=10)
# plt.legend(loc='upper left')
plt.tick_params(labelsize=10)
# plt.xlim(250,400)
plt.show()
# plt.savefig("D:/Projects/Papers/Parkinson/figure1.png", dpi=500, bbox_inches='tight')
