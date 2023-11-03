# 并列柱状图
import matplotlib.pyplot as plt
import numpy as np
sub_map = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')


x=np.arange(5)#柱状图在横坐标上的位置
#列出你要显示的数据，数据的列表长度与x长度相同
# y1=[69.6, 73.1, 78.7, 85.1, 94.5]
# y2=[62.3, 71.3, 67.2, 82.2, 89.1]
# y1=[76.2, 80.0, 82.3, 88.5, 96.2]
# y2=[72.0, 79.2, 78.4, 90.4, 95.2]
# y1=[73.5, 78.0, 79.3, 86.8, 95.4]
# y2=[74.5, 79.0, 80.4, 90.2, 94.2]
y1=[52, 60, 64, 77, 92]
y2=[51, 61, 64, 83, 91]

bar_width=0.3#设置柱状图的宽度
tick_label=['original','azimuth','altitude','pressure','x,y-coord']

#绘制并列柱状图
fig=plt.figure(figsize=(10,8))

plt.bar(x, y1, bar_width, color='#266A2E', )
plt.bar(x+bar_width, y2, bar_width, color='#f07818',)

# plt.legend()#显示图例，即label
plt.xticks(x+bar_width/2,tick_label)#显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置

plt.tick_params(labelsize=20)
plt.xlabel('Feature Set', fontsize=25)
plt.ylabel('MCC(in %)'.translate(sub_map), fontsize=25)

plt.ylim((50, 95))

# my_y_ticks = np.arange(60, 100,5)
# plt.yticks(my_y_ticks)

# plt.show()
plt.savefig("D:/Projects/Papers/Parkinson/difference_mcc.png", dpi=500, bbox_inches='tight')
