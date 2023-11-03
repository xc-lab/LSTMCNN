'''
新的optimizer曲线，包含阴影
'''


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


path = '../data/csv_data/optimizer/pl/val_acc'
result_dict = {}
for optimizer_type in os.listdir(path):
    data = pd.DataFrame()
    result = pd.DataFrame()
    optimizer_path = os.path.join(path, optimizer_type)
    for date in os.listdir(optimizer_path):
        optimizer_data = pd.read_csv(os.path.join(optimizer_path, date),sep=',',header='infer',usecols=[2])
        data[date] = optimizer_data
    result['min'] = data.min(axis=1)
    result['max'] = data.max(axis=1)
    result['mean'] = data.mean(axis=1)
    result_dict[optimizer_type] = result




color_dict = {'SGD':'#00BFFF', 'ASGD':'r', 'Adagrad':'#FF8C00', 'Adadelta':'k', 'Adamax':'#7B68EE', 'RMSprop':'#7FFF00', 'Adam':'#006400'}

fig=plt.figure(figsize=(10,8))

ax=fig.subplots()
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width , box.height* 0.8])
for key,value in result_dict.items():
    if key != 'RMSprop' and key != 'Adadelta':
        acc_mean = value['mean']
        acc_min = value['min']
        acc_max = value['max']
        color = color_dict[key]
        ax.plot(acc_mean, c=color, label=key, linewidth=3.0)
        x_kt = np.linspace(0, len(acc_mean) - 1, num=len(acc_mean))
        ax.fill_between(x_kt, acc_min, acc_max, alpha=0.2, color=color)

ax.set_xlabel('Epochs',fontsize=25)
# ax.set_ylabel('Accuracy', fontsize=25)
ax.set_ylabel('Accuracy', fontsize=25)

# ax.legend(loc='center', bbox_to_anchor=(0.5, 1.12),ncol=5, fontsize=20)

ax.tick_params(labelsize=25)

# ax.grid(linestyle='-.')
# plt.show()
plt.savefig("D:/Projects/Papers/Parkinson/optimizer_pl_val_acc.png", dpi=500, bbox_inches='tight')
