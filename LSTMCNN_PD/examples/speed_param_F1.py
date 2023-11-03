'''
画出模型的横坐标：推理速度，纵坐标：性能，点的大小：模型大小
'''
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
sub_map = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')

# plt.figure(figsize=(10, 6))
#
# # plt.scatter(82.08, 85.7, s=112.77, c='#d9b611')
# # plt.scatter(42.69, 88.7, s=202.37, c='#057748')
# # plt.scatter(42.93, 92.5, s=387.97, c='#8d4bbb')#fff98c
# # plt.scatter(83.89, 95.4, s=590.21, c='#dc3023')
# # plt.scatter(125.09, 92.4, s=978.05, c='#065279')
# #
# #
# # plt.scatter(82.08, 92.3, s=112.77, c='#d9b611', marker = 'o', edgecolors='k')
# # plt.scatter(42.69, 91, s=202.37, c='#057748', marker = 'o', edgecolors='k')
# # plt.scatter(42.93, 94.1, s=387.97, c='#8d4bbb', marker = 'o',  edgecolors='k')#fff98c
# # plt.scatter(83.89, 94.2, s=590.21, c='#dc3023', marker = 'o', edgecolors='k')
# # plt.scatter(125.09, 92.1, s=978.05, c='#065279', marker = 'o', edgecolors='k')
#
# plt.scatter(82.08, 79.4, s=112.77, c='#d9b611')
# plt.scatter(42.69, 83.6, s=202.37, c='#057748')
# plt.scatter(42.93, 87.7, s=387.97, c='#8d4bbb')#fff98c
# plt.scatter(83.89, 92.3, s=590.21, c='#dc3023')
# plt.scatter(125.09, 87.4, s=978.05, c='#065279')
#
#
# plt.scatter(82.08, 87.4, s=112.77, c='#d9b611', marker = 'o', edgecolors='k')
# plt.scatter(42.69, 86.1, s=202.37, c='#057748', marker = 'o', edgecolors='k')
# plt.scatter(42.93, 90.7, s=387.97, c='#8d4bbb', marker = 'o',  edgecolors='k')#fff98c
# plt.scatter(83.89, 90.6, s=590.21, c='#dc3023', marker = 'o', edgecolors='k')
# plt.scatter(125.09, 87.6, s=978.05, c='#065279', marker = 'o', edgecolors='k')
#
# # plt.xlim(8.1, 8.3)
# # plt.ylim(87, 97)
#
# my_x_ticks = np.arange(30, 155,30)
# plt.xticks(my_x_ticks)
#
# my_y_ticks = np.arange(70, 105,10)
# plt.yticks(my_y_ticks)
#
# plt.tick_params(labelsize=15)
# # plt.grid(linestyle='-.', which="both")
# plt.xlabel('Params(K)',fontsize=15)
# plt.ylabel('MCC(%)'.translate(sub_map), fontsize=15)
# plt.show()
# # plt.savefig("D:/Projects/Papers/Parkinson/speed.png", dpi=500, bbox_inches='tight')


plt.figure(figsize=(10, 6))

plt.scatter(91.5, 88.7, s=202.37, c='#d9b611')
# plt.scatter(93.8, 92.5, s=387.97, c='#057748')
# plt.scatter(92.8, 91.0, s=446.21, c='#8d4bbb')
plt.scatter(94.6, 93.1, s=542.21, c='#dc3023')
# plt.scatter(96.2, 95.4, s=590.21, c='#065279')


# plt.scatter(92.8, 91.0, s=202.37, c='#d9b611', marker = 'o', edgecolors='k')
# plt.scatter(95.2, 94.1, s=387.97, c='#057748', marker = 'o', edgecolors='k')
# plt.scatter(93.1, 90.9, s=446.21, c='#8d4bbb', marker = 'o',  edgecolors='k')#fff98c
plt.scatter(94.4, 93.1, s=542.21, c='#dc3023', marker = 'o', edgecolors='k')
# plt.scatter(95.2, 94.2, s=590.21, c='#065279', marker = 'o', edgecolors='k')

# plt.xlim(8.1, 8.3)
# plt.ylim(87, 97)


my_x_ticks = np.arange(91, 97, 1)
plt.xticks(my_x_ticks)

my_y_ticks = np.arange(88, 96, 1)
plt.yticks(my_y_ticks)

plt.tick_params(labelsize=15)
# plt.grid(linestyle='-.', which="both")
plt.xlabel('Accuracy(in %)',fontsize=15)
plt.ylabel('F1score(in %)'.translate(sub_map), fontsize=15)
plt.show()
# plt.savefig("D:/Projects/Papers/Parkinson/speed.png", dpi=500, bbox_inches='tight')

