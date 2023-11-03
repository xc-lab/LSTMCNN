import matplotlib.pyplot as plt
import numpy as np
# 定义下标数字的映射表
sub_map = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')

# 输出：H₂O

font = {
'size':30,
}
sliding_window = ['32','64','128','256']
accuracy       = [0.8519, 0.9231, 0.9615, 0.9231]
precision      = [0.9000, 0.9091, 1.0000, 1.0000]
recall         = [0.7500, 0.9091, 0.9091, 0.8182]
specificity    = [0.9333, 0.9333, 1.0000, 1.0000]
F1             = [0.8182, 0.9091, 0.9524, 0.9000]
MCC            = [0.7031, 0.8424, 0.9232, 0.8497]

import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams['font.sans-serif'] = 'SimHei'
# plt.rcParams['axes.unicode_minus'] = False
# plt.style.use('ggplot')

# '''P型'''
# values_32 = [0.814815, 0.818182, 0.7500, 0.866667, 0.782609, 0.623635, 0.8125]
# values_64 = [0.923077, 0.9091, 0.9091, 0.9333, 0.9091, 0.8424, 0.93333]
# values_128 = [0.9615, 0.916667, 0.999, 0.9333, 0.9565, 0.9249, 0.999]
# values_256 = [0.9231, 0.9999, 0.8182, 0.9999, 0.9000, 0.8497, 0.882353]
'''PL型'''
values_32 = [0.88, 0.8333, 0.909091, 0.857143, 0.869565, 0.761306, 0.923077]
values_64 = [0.92, 0.999, 0.818182, 0.999, 0.90, 0.846114, 0.875]
values_128 = [0.96, 0.999, 0.909091, 0.999, 0.952381, 0.921132, 0.9333]
values_256 = [0.9200, 0.909091, 0.909091, 0.928571, 0.909091, 0.837662, 0.928571]

# values_none = [0.8462, 0.8889, 0.7273, 0.9333, 0.8000, 0.6860]
# values_a = [0.9231, 0.8462, 0.999, 0.8667, 0.8193, 0.8563]
# values_l = [0.8077, 0.7143, 0.9091, 0.7333, 0.8000, 0.6367]
# values_p = [0.8462, 0.7692, 0.9091, 0.8000, 0.8333, 0.7006]
# values_xy = [0.9615, 0.9999, 0.9091, 0.9999, 0.9524, 0.9232]

feature = ['ACC', 'PPV', 'TPR', 'TNR', 'F1 score'.translate(sub_map), 'MCC', 'NPV']
angles = np.linspace(0, 2 * np.pi, len(values_32), endpoint=False)
values_32 = np.concatenate((values_32, [values_32[0]]))
values_64 = np.concatenate([values_64, [values_64[0]]])
values_128 = np.concatenate([values_128, [values_128[0]]])
values_256 = np.concatenate([values_256, [values_256[0]]])
# angles = np.linspace(0, 2 * np.pi, len(values_none), endpoint=False)
# values_none = np.concatenate((values_none, [values_none[0]]))
# values_a = np.concatenate([values_a, [values_a[0]]])
# values_l = np.concatenate([values_l, [values_l[0]]])
# values_p = np.concatenate([values_p, [values_p[0]]])
# values_xy = np.concatenate([values_xy, [values_xy[0]]])


angles = np.concatenate((angles, [angles[0]]))

fig = plt.figure(figsize=(18, 18))
# fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, values_32, 'o-', linewidth=2, c='#FF8C00', label='32')
ax.fill(angles, values_32, alpha=0.25)
ax.plot(angles, values_64, 'o-', linewidth=2, c='#FFD700', label='64')
ax.fill(angles, values_64, alpha=0.25)
ax.plot(angles, values_128, 'o-', linewidth=4, c='#006400', label='128')
ax.fill(angles, values_128, alpha=0.25)
ax.plot(angles, values_256, 'o-', linewidth=2, c='k', label='256')
ax.fill(angles, values_256, alpha=0.25)

# ax.plot(angles, values_none, 'o-', linewidth=3, c='#FF8C00', label='none')
# ax.fill(angles, values_none, alpha=0.25)
# ax.plot(angles, values_a, 'o-', linewidth=3, c='#FFD700', label='a')
# ax.fill(angles, values_a, alpha=0.25)
# ax.plot(angles, values_l, 'o-', linewidth=3, c='#6495ED', label='l')
# ax.fill(angles, values_l, alpha=0.25)
# ax.plot(angles, values_p, 'o-', linewidth=3, c='k', label='p')
# ax.fill(angles, values_p, alpha=0.25)
# ax.plot(angles, values_xy, 'o-', linewidth=3, c='#006400', label='x,y')
# ax.fill(angles, values_xy, alpha=0.25)

ang = angles*180/np.pi
ax.set_thetagrids(ang[:-1], feature)

plt.tick_params(labelsize=40)

ax.set_ylim(0.5, 1.0)
plt.legend(prop=font, bbox_to_anchor=(0.95, 0.99))
# ax.grid(True)
plt.show()
# plt.savefig("D:/Projects/Papers/Parkinson/figure1.png", dpi=500, bbox_inches='tight')











# plt.figure(dpi=200, figsize=(20, 2))

# plt.plot(sliding_window, accuracy, c='#FF8C00', label='Accuracy', linewidth=3.0, linestyle=':', )
# plt.plot(sliding_window, precision, c='#FFD700', label='Precision', linewidth=3.0, linestyle="--")
# plt.plot(sliding_window, recall, c='#00BFFF', label='Recall', linewidth=3.0, linestyle="-.")
# plt.plot(sliding_window, specificity, c='#7FFF00', label='Specificity', linewidth=3.0, linestyle="-")
# plt.plot(sliding_window, F1, c='#006400', label='F1 score', linewidth=3.0)
# plt.plot(sliding_window, MCC, c='k', label='MCC', linewidth=3.0)
# plt.legend()
# plt.xticks(sliding_window, ('32','64','128','256'))
# plt.tick_params(labelsize=10)
#
# plt.show()
# plt.savefig("D:/Projects/Papers/Parkinson/figure1.png", dpi=500, bbox_inches='tight')
