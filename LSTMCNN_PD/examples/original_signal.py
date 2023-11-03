'''
生成标准信号
'''
import numpy as np
import matplotlib.pyplot as plt

def signal(x,y):
    x0 = [x]*1000
    y0 = np.linspace(1,10,1000).tolist()

    x1 = np.linspace(x,x+5,1000).tolist()
    y1 = [10]*1000

    x2 = [x+5]*1000
    y2 = np.linspace(10,1,1000).tolist()

    x3 = np.linspace(x+5,x+10,1000).tolist()
    y3 = [1]*1000

    X = x0 + x1 + x2 + x3
    Y = y0 + y1 + y2 + y3
    return X, Y


X0, Y0 = signal(1,1)
X1, Y1 = signal(11, 1)
X2, Y2 = signal(21, 1)
X3, Y3 = signal(31, 1)
X4, Y4 = signal(41, 1)
X5, Y5 = signal(51, 1)

X = X0+X1+X2+X3+X4+X5
Y = Y0+Y1+Y2+Y3+Y4+Y5
X = np.array(X[:-1000])
Y = np.array(Y[:-1000])


# plt.figure(figsize=(18, 3))
# plt.plot(X, Y, linewidth=3.0)
# plt.axis('off')
# # plt.title(file_name)
# plt.show()



x_max = max(X)
x_min = min(X)
normalized_X = (X - x_min) / (x_max - x_min)
y_max = max(Y)
y_min = min(Y)
normalized_Y = (Y - y_min) / (y_max - y_min)


# plt.plot(normalized_Y, c='#006400', label='\'y\'', )
# plt.plot(normalized_X, c='#7FFF00', label='\'x\'', )
# # plt.plot(T, c='k', label='t-time', linewidth=8.0)
# plt.legend()
# plt.tick_params(labelsize=30)
# # plt.title('Time')
# plt.show()

diff_n_X = np.diff(normalized_X, n=1)
diff_n_Y = np.diff(normalized_Y, n=1)
plt.plot(diff_n_Y, c='#006400', label='\'y\'', )
# plt.plot(diff_n_X, c='#7FFF00', label='\'x\'', )
# plt.plot(T, c='k', label='t-time', linewidth=8.0)
plt.legend()
plt.tick_params(labelsize=30)
# plt.title('Time')
plt.show()



















