
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import f1_score
import seaborn as sn  #画图模块

def roc_p():
    mnist_1_p_tpr_mean = 0.782
    mnist_1_p_tpr_std = 0.109
    mnist_1_p_fpr_mean = 0.027
    mnist_1_p_fpr_std = 0.053
    mnist_1_c = '#d9b611'
    mnist_1_X = [0, mnist_1_p_fpr_mean + mnist_1_p_fpr_std, 1, mnist_1_p_fpr_mean - mnist_1_p_fpr_std]
    mnist_1_Y = [0, mnist_1_p_tpr_mean - mnist_1_p_tpr_std, 1, mnist_1_p_tpr_mean + mnist_1_p_tpr_std]

    mnist_2_p_tpr_mean = 0.80000
    mnist_2_p_tpr_std = 0.068030
    mnist_2_p_fpr_mean = 0.0000
    mnist_2_p_fpr_std = 0.00000
    mnist_2_c = '#057748'
    mnist_2_X = [0, mnist_2_p_fpr_mean + mnist_2_p_fpr_std, 1, mnist_2_p_fpr_mean - mnist_2_p_fpr_std]
    mnist_2_Y = [0, mnist_2_p_tpr_mean - mnist_2_p_tpr_std, 1, mnist_2_p_tpr_mean + mnist_2_p_tpr_std]

    lstmcnn_3_3_p_tpr_mean = 0.909091
    lstmcnn_3_3_p_tpr_std = 0.081311
    lstmcnn_3_3_p_fpr_mean = 0.04000
    lstmcnn_3_3_p_fpr_std = 0.032660
    lstmcnn_3_3_c = '#8d4bbb'
    lstmcnn_3_3_X = [0, lstmcnn_3_3_p_fpr_mean + lstmcnn_3_3_p_fpr_std, 1,
                     lstmcnn_3_3_p_fpr_mean - lstmcnn_3_3_p_fpr_std]
    lstmcnn_3_3_Y = [0, lstmcnn_3_3_p_tpr_mean - lstmcnn_3_3_p_tpr_std, 1,
                     lstmcnn_3_3_p_tpr_mean + lstmcnn_3_3_p_tpr_std]

    clstmcnn_3_3_p_tpr_mean = 0.945455
    clstmcnn_3_3_p_tpr_std = 0.044536
    clstmcnn_3_3_p_fpr_mean = 0.026667
    clstmcnn_3_3_p_fpr_std = 0.032660
    clstmcnn_3_3_c = '#dc3023'
    clstmcnn_3_3_X = [0, clstmcnn_3_3_p_fpr_mean + clstmcnn_3_3_p_fpr_std, 1,
                      clstmcnn_3_3_p_fpr_mean - clstmcnn_3_3_p_fpr_std]
    clstmcnn_3_3_Y = [0, clstmcnn_3_3_p_tpr_mean - clstmcnn_3_3_p_tpr_std, 1,
                      clstmcnn_3_3_p_tpr_mean + clstmcnn_3_3_p_tpr_std]

    clstmcnn_3_3_p_2_tpr_mean = 0.890909
    clstmcnn_3_3_p_2_tpr_std = 0.068030
    clstmcnn_3_3_p_2_fpr_mean = 0.026667
    clstmcnn_3_3_p_2_fpr_std = 0.032660
    clstmcnn_3_3_2_c = '#065279'
    clstmcnn_3_3_2_X = [0, clstmcnn_3_3_p_2_fpr_mean + clstmcnn_3_3_p_2_fpr_std, 1,
                        clstmcnn_3_3_p_2_fpr_mean - clstmcnn_3_3_p_2_fpr_std]
    clstmcnn_3_3_2_Y = [0, clstmcnn_3_3_p_2_tpr_mean - clstmcnn_3_3_p_2_tpr_std, 1,
                        clstmcnn_3_3_p_2_tpr_mean + clstmcnn_3_3_p_2_tpr_std]

    lw = 2.0
    a = 0.2
    plt.figure(figsize=(10, 6))

    plt.plot([0, 1], [0, 1], color='silver', lw=lw, linestyle='--', label='Chance')

    plt.plot([0, mnist_1_p_fpr_mean, 1], [0, mnist_1_p_tpr_mean, 1], color=mnist_1_c, lw=lw, label='1D CNN-1 (AUC= ' + str(90.0) + '\u00B1' + str(3.4)+')')
    plt.fill(mnist_1_X, mnist_1_Y, facecolor=mnist_1_c, alpha=0.1)

    plt.plot([0, mnist_2_p_fpr_mean, 1], [0, mnist_2_p_tpr_mean, 1], color=mnist_2_c, lw=lw, label='1D CNN-2 (AUC= '+str(87.8)+'\u00B1'+str(2.8)+')')
    plt.fill(mnist_2_X, mnist_2_Y, facecolor=mnist_2_c, alpha=0.05)

    plt.plot([0, lstmcnn_3_3_p_fpr_mean, 1], [0, lstmcnn_3_3_p_tpr_mean, 1], color=lstmcnn_3_3_c, lw=lw, label='LSTM-CNN-1 (AUC= '+str(93.5)+'\u00B1'+str(3.6)+')')
    plt.fill(lstmcnn_3_3_X, lstmcnn_3_3_Y, facecolor=lstmcnn_3_3_c, alpha=0.1)

    plt.plot([0, clstmcnn_3_3_p_fpr_mean, 1], [0, clstmcnn_3_3_p_tpr_mean, 1], color=clstmcnn_3_3_c, lw=lw, label='LSTM-CNN-2 (AUC= '+str(95.9)+'\u00B1'+str(2.5)+')')
    plt.fill(clstmcnn_3_3_X, clstmcnn_3_3_Y, facecolor=clstmcnn_3_3_c, alpha=0.1)
    #
    plt.plot([0, clstmcnn_3_3_p_2_fpr_mean, 1], [0, clstmcnn_3_3_p_2_tpr_mean, 1], color=clstmcnn_3_3_2_c, lw=lw, label='LSTM-CNN-3 (AUC= '+str(93.2)+'\u00B1'+str(4.2)+')')
    plt.fill(clstmcnn_3_3_2_X, clstmcnn_3_3_2_Y, facecolor=clstmcnn_3_3_2_c, alpha=0.1)

    # plt.grid(linestyle='-.')
    plt.tick_params(labelsize=15)
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.legend(loc='lower right', fontsize=12)
    plt.savefig("D:/Projects/Papers/Parkinson/roc_p.png", dpi=500, bbox_inches='tight')
    # plt.show()

def roc_pl():
    mnist_p_tpr_mean = 0.836364
    mnist_p_tpr_std = 0.036364
    mnist_p_fpr_mean = 0.0000
    mnist_p_fpr_std = 0.00000
    mnist_c = '#d9b611'
    mnist_X = [0, mnist_p_fpr_mean + mnist_p_fpr_std, 1, mnist_p_fpr_mean - mnist_p_fpr_std]
    mnist_Y = [0, mnist_p_tpr_mean - mnist_p_tpr_std, 1, mnist_p_tpr_mean + mnist_p_tpr_std]

    mnist1_p_tpr_mean = 0.872727
    mnist1_p_tpr_std = 0.044536
    mnist1_p_fpr_mean = 0.0143
    mnist1_p_fpr_std = 0.028572
    mnist1_c = '#057748'
    mnist1_X = [0, mnist1_p_fpr_mean + mnist1_p_fpr_std, 1, mnist1_p_fpr_mean - mnist1_p_fpr_std]
    mnist1_Y = [0, mnist1_p_tpr_mean - mnist1_p_tpr_std, 1, mnist1_p_tpr_mean + mnist1_p_tpr_std]

    lstmcnn_3_3_p_tpr_mean = 0.890909
    lstmcnn_3_3_p_tpr_std = 0.068030
    lstmcnn_3_3_p_fpr_mean = 0.0
    lstmcnn_3_3_p_fpr_std = 0.0
    lstmcnn_3_3_c = '#8d4bbb'
    lstmcnn_3_3_X = [0, lstmcnn_3_3_p_fpr_mean + lstmcnn_3_3_p_fpr_std, 1,
                     lstmcnn_3_3_p_fpr_mean - lstmcnn_3_3_p_fpr_std]
    lstmcnn_3_3_Y = [0, lstmcnn_3_3_p_tpr_mean - lstmcnn_3_3_p_tpr_std, 1,
                     lstmcnn_3_3_p_tpr_mean + lstmcnn_3_3_p_tpr_std]

    clstmcnn_3_3_p_tpr_mean = 0.890909
    clstmcnn_3_3_p_tpr_std = 0.036364
    clstmcnn_3_3_p_fpr_mean = 0.0
    clstmcnn_3_3_p_fpr_std = 0.0
    clstmcnn_3_3_c = '#dc3023'
    clstmcnn_3_3_X = [0, clstmcnn_3_3_p_fpr_mean + clstmcnn_3_3_p_fpr_std, 1,
                      clstmcnn_3_3_p_fpr_mean - clstmcnn_3_3_p_fpr_std]
    clstmcnn_3_3_Y = [0, clstmcnn_3_3_p_tpr_mean - clstmcnn_3_3_p_tpr_std, 1,
                      clstmcnn_3_3_p_tpr_mean + clstmcnn_3_3_p_tpr_std]

    clstmcnn_3_3_p_2_tpr_mean = 0.854546
    clstmcnn_3_3_p_2_tpr_std = 0.044536
    clstmcnn_3_3_p_2_fpr_mean = 0.0
    clstmcnn_3_3_p_2_fpr_std = 0.0
    clstmcnn_3_3_2_c = '#065279'
    clstmcnn_3_3_2_X = [0, clstmcnn_3_3_p_2_fpr_mean + clstmcnn_3_3_p_2_fpr_std, 1,
                        clstmcnn_3_3_p_2_fpr_mean - clstmcnn_3_3_p_2_fpr_std]
    clstmcnn_3_3_2_Y = [0, clstmcnn_3_3_p_2_tpr_mean - clstmcnn_3_3_p_2_tpr_std, 1,
                        clstmcnn_3_3_p_2_tpr_mean + clstmcnn_3_3_p_2_tpr_std]

    lw = 2.0
    plt.figure(figsize=(10, 6))

    plt.plot([0, 1], [0, 1], color='silver', lw=lw, linestyle='--', label='Chance')

    plt.plot([0, mnist_p_fpr_mean, 1], [0, mnist_p_tpr_mean, 1], color=mnist_c, lw=lw, label='1D CNN-1 (AUC='+str(90.8)+'\u00B1'+str(1.8)+')')
    plt.fill(mnist_X, mnist_Y, facecolor=mnist_c, alpha=0.1)

    plt.plot([0, mnist1_p_fpr_mean, 1], [0, mnist1_p_tpr_mean, 1], color=mnist_c, lw=lw,
             label='1D CNN-2 (AUC=' + str(92.9) + '\u00B1' + str(2.1)+')')
    plt.fill(mnist1_X, mnist1_Y, facecolor=mnist_c, alpha=0.05)

    plt.plot([0, lstmcnn_3_3_p_fpr_mean, 1], [0, lstmcnn_3_3_p_tpr_mean, 1], color=lstmcnn_3_3_c, lw=lw, label='LSTM-CNN-1 (AUC='+str(94.5)+'\u00B1'+str(3.4)+")")
    plt.fill(lstmcnn_3_3_X, lstmcnn_3_3_Y, facecolor=lstmcnn_3_3_c, alpha=0.1)

    plt.plot([0, clstmcnn_3_3_p_fpr_mean, 1], [0, clstmcnn_3_3_p_tpr_mean, 1], color=clstmcnn_3_3_c, lw=lw, label='LSTM-CNN-2 (AUC='+str(94.5)+'\u00B1'+str(1.8)+')')
    plt.fill(clstmcnn_3_3_X, clstmcnn_3_3_Y, facecolor=clstmcnn_3_3_c, alpha=0.1)
    #
    plt.plot([0, clstmcnn_3_3_p_2_fpr_mean, 1], [0, clstmcnn_3_3_p_2_tpr_mean, 1], color=clstmcnn_3_3_2_c, lw=lw, label='LSTM-CNN-3 (AUC='+str(92.7)+'\u00B1'+str(2.2)+')')
    plt.fill(clstmcnn_3_3_2_X, clstmcnn_3_3_2_Y, facecolor=clstmcnn_3_3_2_c, alpha=0.1)

    # plt.grid(linestyle='-.')
    plt.tick_params(labelsize=15)
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.legend(loc='lower right', fontsize=12)
    # plt.savefig("D:/Projects/Papers/Parkinson/roc_pl.png", dpi=500, bbox_inches='tight')
    plt.show()

if __name__=='__main__':
    # roc_p()
    roc_pl()
