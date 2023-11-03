import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

Adagrad_data = pd.read_csv('../data/csv_data/optimizer/Adagrad.csv',sep=',',header='infer',usecols=[2])
Adam_data = pd.read_csv('../data/csv_data/optimizer/Adam.csv',sep=',',header='infer',usecols=[2])
SGD_data = pd.read_csv('../data/csv_data/optimizer/SGD.csv',sep=',',header='infer',usecols=[2])
RMSprop_data = pd.read_csv('../data/csv_data/optimizer/RMSprop.csv',sep=',',header='infer',usecols=[2])
Adamax_data = pd.read_csv('../data/csv_data/optimizer/Adamax.csv',sep=',',header='infer',usecols=[2])
Adadelate_data = pd.read_csv('../data/csv_data/optimizer/Adadelate.csv',sep=',',header='infer',usecols=[2])
ASGD_data = pd.read_csv('../data/csv_data/optimizer/ASGD.csv',sep=',',header='infer',usecols=[2])


plt.figure(figsize=(7, 5.5))
plt.plot(SGD_data, c='#00BFFF', label='SGD', linewidth=2.0, )
plt.plot(ASGD_data, c='k', label='ASGD', linewidth=2.0, )
plt.plot(Adagrad_data, c='#FF8C00', label='Adagrad', linewidth=2.0)
plt.plot(RMSprop_data, c='#7FFF00', label='RMSprop', linewidth=2.0,)
plt.plot(Adam_data, c='#006400', label='Adam', linewidth=2.0, )

x_major_locator=MultipleLocator(25)
y_major_locator=MultipleLocator(0.1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

# plt.plot(T, c='k', label='t-time', linewidth=1.0)
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Loss', fontsize=15)
# plt.ylim([0, 1])
plt.grid(linestyle='-.')
plt.legend(fontsize=15)

plt.tick_params(labelsize=15)
plt.show()
# plt.savefig("D:/Projects/Papers/Parkinson/figure1.png", dpi=500, bbox_inches='tight')





















Adagrad_data_1 = pd.read_csv('../data/csv_data/optimizer/Adagrad_1.csv',sep=',',header='infer',usecols=[2])
Adam_data_1 = pd.read_csv('../data/csv_data/optimizer/Adam_1.csv',sep=',',header='infer',usecols=[2])
SGD_data_1 = pd.read_csv('../data/csv_data/optimizer/SGD_1.csv',sep=',',header='infer',usecols=[2])
RMSprop_data_1 = pd.read_csv('../data/csv_data/optimizer/RMSprop_1.csv',sep=',',header='infer',usecols=[2])
Adamax_data_1 = pd.read_csv('../data/csv_data/optimizer/Adamax_1.csv',sep=',',header='infer',usecols=[2])
Adadelate_data_1 = pd.read_csv('../data/csv_data/optimizer/Adadelate_1.csv',sep=',',header='infer',usecols=[2])
ASGD_data_1 = pd.read_csv('../data/csv_data/optimizer/ASGD_1.csv',sep=',',header='infer',usecols=[2])




# fig = plt.figure(figsize=(10, 10))
# ax1=fig.subplots()
# ax2=ax1.twinx()
# ax2.plot(Adagrad_data, c='#FF8C00', label='Adagrad', linewidth=2.0, linestyle=':')
# ax2.plot(Adam_data, c='#006400', label='Adam', linewidth=2.0, linestyle=':')
# ax2.plot(SGD_data, c='#00BFFF', label='SGD', linewidth=2.0, linestyle=':')
# ax2.plot(RMSprop_data, c='#7FFF00', label='RMSprop', linewidth=2.0, linestyle=':')
# # plt.plot(Adamax_data, c='k', label='Adamax', linewidth=2.0)
# ax2.plot(ASGD_data, c='k', label='ASGD', linewidth=2.0, linestyle=':')
# ax2.legend( fontsize=15)
#
# ax1.plot(Adagrad_data_1, c='#FF8C00', label='Adagrad', linewidth=2.0)
# ax1.plot(Adam_data_1, c='#006400', label='Adam', linewidth=2.0)
# ax1.plot(SGD_data_1, c='#00BFFF', label='SGD', linewidth=2.0)
# ax1.plot(RMSprop_data_1, c='#7FFF00', label='RMSprop', linewidth=2.0)
# # plt.plot(Adamax_data, c='k', label='Adamax', linewidth=2.0)
# ax1.plot(ASGD_data_1, c='k', label='ASGD', linewidth=2.0)
# ax1.legend( fontsize=15)
#
# ax1.set_xlabel('Epochs',fontsize=15)
# ax2.set_ylabel('Loss', fontsize=15)
# ax1.set_ylabel('Accuracy', fontsize=15)

# x_major_locator=MultipleLocator(25)
# y_major_locator=MultipleLocator(0.1)
# ax=plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)

# plt.plot(T, c='k', label='t-time', linewidth=1.0)
# plt.xlabel('Epochs', fontsize=15)
# plt.ylabel('Loss', fontsize=15)
# plt.ylim([0, 1])
# plt.grid(linestyle='-.')
# plt.legend()

# plt.tick_params(labelsize=15)
# plt.show()
# plt.savefig("D:/Projects/Papers/Parkinson/figure1.png", dpi=500, bbox_inches='tight')
