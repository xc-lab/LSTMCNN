'''
画出训练和测试曲线，多次组合
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

acc_max = []
acc_min = []
acc_mean = []

path = '../data/csv_data/val_acc_loss/acc.xlsx'
df_acc = pd.read_excel(path)
data_acc = np.array(df_acc)
acc_max = np.max(data_acc, axis=1)
acc_min = np.min(data_acc, axis=1)
acc_mean = np.mean(data_acc, axis=1)

path = '../data/csv_data/val_acc_loss/loss.xlsx'
df_loss = pd.read_excel(path)
data_loss = np.array(df_loss)
loss_max = np.max(data_loss, axis=1)
loss_min = np.min(data_loss, axis=1)
loss_mean = np.mean(data_loss, axis=1)

fig=plt.figure(figsize=(10,8))

ax1=fig.subplots()
ax2=ax1.twinx()    #使用twinx()，得到与ax1 对称的ax2,共用一个x轴，y轴对称（坐标不对称）
ax1.plot(acc_mean, c='#006400', label='Accuracy', linewidth=3.0)
x_kt = np.linspace(0, len(acc_mean) - 1, num=len(acc_mean))
ax1.fill_between(x_kt, acc_min, acc_max, alpha=0.2, color='#006400')
ax1.legend(['Accuracy'], loc='upper left', fontsize=20)

ax2.plot(loss_mean, c='#FF8C00', label='Loss', linewidth=3.0)
ax2.fill_between(x_kt, loss_min, loss_max, alpha=0.2, color='#FF8C00')
ax2.legend(['Loss'], loc='upper right', fontsize=20)

ax1.set_xlabel('Epochs',fontsize=20)
ax1.set_ylabel('Accuracy', fontsize=20)
ax2.set_ylabel('Loss', fontsize=20)


ax1.tick_params(labelsize=20)
ax2.tick_params(labelsize=20)

ax1.grid(linestyle='-.')
plt.show()
# plt.savefig("D:/Projects/Papers/Parkinson/figure1.png", dpi=500, bbox_inches='tight')
