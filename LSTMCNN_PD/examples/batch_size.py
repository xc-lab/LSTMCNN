'''
变量batch size，对模型的影响
'''
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

two_data = pd.read_csv('../data/csv_data/batchsize/2022_06_14_22_36_13.csv',sep=',',header='infer',usecols=[2])
three_data = pd.read_csv('../data/csv_data/batchsize/2022_06_14_23_17_30.csv',sep=',',header='infer',usecols=[2])
four_data = pd.read_csv('../data/csv_data/batchsize/2022_06_14_23_17_41.csv',sep=',',header='infer',usecols=[2])
one_data = pd.read_csv('../data/csv_data/batchsize/2022_06_14_23_47_40.csv',sep=',',header='infer',usecols=[2])
min_data = pd.read_csv('../data/csv_data/batchsize/2022_06_15_19_48_50.csv',sep=',',header='infer',usecols=[2])
# Adadelate_data = pd.read_csv('../data/csv_data/Adadelate.csv',sep=',',header='infer',usecols=[2])
# ASGD_data = pd.read_csv('../data/csv_data/ASGD.csv',sep=',',header='infer',usecols=[2])

plt.figure(figsize=(7, 5.5))
plt.plot(min_data, c='k', label='16', linewidth=2.0)
plt.plot(one_data, c='#FF8C00', label='32', linewidth=2.0)
plt.plot(two_data, c='#006400', label='64', linewidth=2.0)
plt.plot(three_data, c='#00BFFF', label='128', linewidth=2.0)
plt.plot(four_data, c='#7FFF00', label='256', linewidth=2.0)
# plt.plot(ASGD_data, c='k', label='ASGD', linewidth=2.0)

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
