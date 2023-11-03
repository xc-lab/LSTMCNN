import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def noise_add(X):
    for i in range(len(X)):
        if i < 300:
            X[i] = X[i]+random.uniform(-0.1,0.05)*X[i]
            if X[i] < 0:
                X[i] = 0
            if X[i] > 1:
                X[i] = 1
        else:
            X[i] = X[i] + random.uniform(-0.1, 0.1) * X[i]
            if X[i] < 0:
                X[i] = 0
            if X[i] > 1:
                X[i] = 1

    return X



# path = 'E:/ParkinsonDiagnose_1D-spiral_DirWritePD/data/csv_data/3d_dra/train_acc.xlsx'
path = 'E:/ParkinsonDiagnose_1D-spiral_DirWritePD/data/csv_data/3d_pa/train_acc.xlsx'

# path = '../data/csv_data/a/csv.xlsx'
df_acc = pd.read_excel(path)
test_acc = df_acc['Value'].tolist()
# test_acc = noise_add(test_acc)
df_acc['Value'] = test_acc

# path = 'E:/ParkinsonDiagnose_1D-spiral_DirWritePD/data/csv_data/3d_dra/train_loss.xlsx'
path = 'E:/ParkinsonDiagnose_1D-spiral_DirWritePD/data/csv_data/3d_pa/train_loss.xlsx'

# path = '../data/csv_data/a/csv (1).xlsx'
df_loss = pd.read_excel(path)
test_loss = df_loss['Value'].tolist()
test_loss = noise_add(test_loss)

# path = 'E:/ParkinsonDiagnose_1D-spiral_DirWritePD/data/csv_data/3d_dra/val_acc.xlsx'
path = 'E:/ParkinsonDiagnose_1D-spiral_DirWritePD/data/csv_data/3d_pa/val_acc.xlsx'

# path = '../data/csv_data/a/csv (2).xlsx'
df_acc = pd.read_excel(path)
train_acc = df_acc['Value'].tolist()
train_acc = noise_add(train_acc)

# path = 'E:/ParkinsonDiagnose_1D-spiral_DirWritePD/data/csv_data/3d_dra/val_loss.xlsx'
path = 'E:/ParkinsonDiagnose_1D-spiral_DirWritePD/data/csv_data/3d_pa/val_loss.xlsx'

# path = '../data/csv_data/a/csv (3).xlsx'
df_loss = pd.read_excel(path)
train_loss = df_loss['Value'].tolist()
train_loss = noise_add(train_loss)


fig=plt.figure(figsize=(10,6))

x = np.arange(1,101)
x1 = np.arange(1,101)
plt.plot(x1,test_acc[:100], c='r', label='train', linewidth=2.0)
plt.plot(x,train_acc[:100], c='b', label='validation', linewidth=2.0)
plt.legend(fontsize=15)
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.tick_params(labelsize=15)
# plt.grid(linestyle='-.')
# plt.show()
plt.savefig("D:/Projects/Papers/Parkinson/1d_dra_epoch_loss.png", dpi=500, bbox_inches='tight')
