import json
from data_utils.utils import *

def filter_extreme_3sigma(data,n,times):
    # times进行times次3sigma处理
    series = data.copy()
    for i in range(times):
        mean = series.mean()
        std = series.std()
        max_range = mean + n*std
        min_range = mean - n*std
        series = np.clip(series,min_range,max_range)
    return series


scale = [1, 1, 1, 1, 1, 1]
dim_id = [0, 1, 2, 3, 4, 5]
order = 1

path_kt = '../data/raw_data/KT/KT115/20190523-171459-KT115-ptrace_7e02ddc3-9be8-4570-bb28-78f28c86cb95.json'
with open(path_kt) as curr_file:
    test_data_kt = json.load(curr_file)
frame_data_kt = test_data_kt['data']
index = ['a', 'l', 'p', 'x', 'y', 't']
data = get_full_data_array(frame_data_kt, index)  # 得到数组
data_scale_kt = get_column_data_scale(data)  #

data = get_colomn_scalled_difference(data, order, scale, dim_id)

X = data[:,4][200:800]
X = filter_extreme_3sigma(X, 2.4, 5)

x = np.arange(len(X))


plt.figure(figsize=(5,1))
plt.plot(x,X, c='#4695d6', linewidth=0.5)
plt.plot(x[100:230], X[100:230], c='r', linewidth=0.5)
plt.axis('off')
plt.show()
# plt.savefig("D:/Projects/Papers/Parkinson/figure3_p.png", dpi=500, bbox_inches='tight')


