import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, det_curve
from sklearn.metrics import precision_recall_curve, average_precision_score


def multi_models_roc(names, colors, true_labels, pre_labels, save=True, dpin=100):
    plt.figure(figsize=(7, 5))
    # plt.rcParams.update({'font.size': 8})
    for (name, colorname, true_label, pre_label) in zip(names, colors, true_labels, pre_labels):
        fpr, tpr, thresholds = roc_curve(true_label, pre_label)
        plt.plot(fpr, tpr, lw=3, label='{}{:.4f}'.format(name, auc(fpr,tpr)), color=colorname)
        # plt.plot([0, 1], [0, 1], '--', lw=3, color='grey')
        # plt.axis('square')
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        plt.xlabel('False Positive Rate', fontsize=15)
        plt.ylabel('True Positive Rate', fontsize=15)
        # plt.title('ROC Curve', fontsize=10)
        plt.legend(loc='lower right', fontsize=15)
        plt.tick_params(labelsize=15)
    plt.grid(linestyle='-.')
    plt.show()
    # plt.savefig("D:/Projects/Papers/Parkinson/figure1.png", dpi=500, bbox_inches='tight')


def multi_models_PR(names, colors, true_labels, pre_labels, save=True, dpin=100):
    plt.figure(figsize=(7, 5))
    # plt.rcParams.update({'font.size': 8})
    for (name, colorname, true_label, pre_label) in zip(names, colors, true_labels, pre_labels):
        precision, recall, thresholds = precision_recall_curve(true_label, pre_label)
        pr_auc0 = average_precision_score(true_label, pre_label)
        plt.plot(recall, precision, lw=3, label='{}'.format(name), color=colorname)
        # plt.plot([0, 1], [1, 0], '--', lw=3, color='grey')
        # plt.axis('square')
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        plt.xlabel('Recall', fontsize=15)
        plt.ylabel('Precision', fontsize=15)
        # plt.title('ROC Curve', fontsize=10)
        plt.legend(loc='lower left', fontsize=15)
        plt.tick_params(labelsize=15)
    plt.grid(linestyle='-.')
    plt.show()
    # plt.savefig("D:/Projects/Papers/Parkinson/figure1.png", dpi=500, bbox_inches='tight')


def multi_models_det(names, colors, true_labels, pre_labels, save=True, dpin=100):
    plt.figure(figsize=(7, 5))
    # plt.rcParams.update({'font.size': 8})
    for (name, colorname, true_label, pre_label) in zip(names, colors, true_labels, pre_labels):
        fpr, fnr, thresholds = det_curve(true_label, pre_label)
        plt.plot(fpr, fnr, lw=3, label='{}'.format(name), color=colorname)
        # plt.plot([0, 1], [0, 1], '--', lw=3, color='grey')
        # plt.axis('square')
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        plt.xlabel('False Positive Rate', fontsize=15)
        plt.ylabel('False Negative Rate', fontsize=15)
        # plt.title('ROC Curve', fontsize=10)
        plt.legend(loc='upper right', fontsize=15)
        plt.tick_params(labelsize=15)
    plt.grid(linestyle='-.')
    plt.show()
    # plt.savefig("D:/Projects/Papers/Parkinson/figure1.png", dpi=500, bbox_inches='tight')

def multi_models_ks(names, colors, true_labels, pre_labels, save=True, dpin=100):
    plt.figure(figsize=(7, 5))
    # plt.rcParams.update({'font.size': 8})
    for (name, colorname, true_label, pre_label) in zip(names, colors, true_labels, pre_labels):
        fpr, tpr, thresholds = roc_curve(true_label, pre_label)
        plt.plot(abs(tpr - fpr), lw=3, label='{}'.format(name), color=colorname)
        # plt.plot([0, 1], [0, 1], '--', lw=3, color='grey')
        # plt.axis('square')
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        plt.xlabel('Threshold', fontsize=15)
        plt.ylabel('True Positive Rate - False Positive Rate', fontsize=15)
        # plt.title('ROC Curve', fontsize=10)
        plt.legend(loc='upper left', fontsize=15)
        plt.tick_params(labelsize=15)
    plt.grid(linestyle='-.')
    plt.show()
    # plt.savefig("D:/Projects/Papers/Parkinson/figure1.png", dpi=500, bbox_inches='tight')



def multi_models(names, colors, true_labels, pre_labels, save=True, dpin=100):
    fig = plt.figure()
    ax_roc = fig.add_subplot(141)
    ax_ks = fig.add_subplot(142)
    ax_det = fig.add_subplot(143)
    ax_pr = fig.add_subplot(144)

    for (name, colorname, true_label, pre_label) in zip(names, colors, true_labels, pre_labels):
        fpr, tpr, thresholds = roc_curve(true_label, pre_label)
        fpr, fnr, thresholds = det_curve(true_label, pre_label)
        precision, recall, thresholds = precision_recall_curve(true_label, pre_label)
        ax_roc.plot(fpr, tpr, lw=3, label='{}'.format(name), color=colorname)
        # ax_ks.plot(abs(tpr - fpr), lw=3, label='{}'.format(name), color=colorname)
        # ax_det.plot(fpr, fnr, lw=3, label='{}'.format(name), color=colorname)
        # ax_pr.plot(recall, precision, lw=3, label='{}'.format(name), color=colorname)

    # plt.xlabel('Recall', fontsize=10)
    # plt.ylabel('Precision', fontsize=10)
    # # plt.title('ROC Curve', fontsize=10)
    # plt.legend(loc='lower right', fontsize=5)
    # plt.tick_params(labelsize=10)
    plt.show()





if __name__=='__main__':
    names = ['None', 'a', 'l', 'p', 'x,y']
    colors = ['k', '#FF8C00', '#FFD700', '#00BFFF', '#006400']
    true_labels = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   ]

    pre_labels =  [[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,1,0,1,1,1,1,1],
                   [0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,1,1,1],
                   [0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1],
                   [0 ,0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,1 ,1 ,1 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1],
                   [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,1 ,1 ,1 ,1 ,0 ,1 ,1 ,1 ,1 ,1]
                   ]
    multi_models_roc(names, colors, true_labels, pre_labels, save=True, dpin=100)
    multi_models_PR(names, colors, true_labels, pre_labels, save=True, dpin=100)
    multi_models_det(names, colors, true_labels, pre_labels, save=True, dpin=100)
    multi_models_ks(names, colors, true_labels, pre_labels, save=True, dpin=100)
    multi_models(names, colors, true_labels, pre_labels, save=True, dpin=100)
