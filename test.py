''''
PD:1, KT:0  4.008612 展示哪一块分类正确，哪一块分类错误
'''
import numpy as np
import re
import os
import argparse
from models import *
from tqdm import tqdm
import time
from numpy import *
from torch.utils.data import DataLoader
from dataset import DatasetList
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score
from data_utils.utils import *
from estimation import  Performance
from models import Mnist, LSTMCNN, GRUCNN
from SiT import SiT
import matplotlib.pyplot as plt
from pointnet import PointNet, PointNet_loss

from collections import Counter


def image_test_result(full_data, pd_idx, hc_idx, json_file_path, target_labels_dict, pred_labels_dict, true_id):
    '''
    画出一条数据哪些部分是错误分类的
    :param full_data:
    :param wrong_idx:
    :param json_file_path:
    :param out_path:
    :param target_labels_dict:
    :param pred_labels_dict:
    :return:
    '''
    if true_id=='00':
        true_label = 'HC'
    elif true_id=='01':
        true_label = 'PD'

    if len(pred_labels_dict)==1:
        if list(pred_labels_dict.keys())==[0]:
            pred_label = 'HC'
        else:
            pred_label = 'PD'

    if len(pred_labels_dict)==2:
        if pred_labels_dict[0] > pred_labels_dict[1]:
            pred_label = 'HC'
        elif pred_labels_dict[0] <= pred_labels_dict[1]:
            pred_label = 'PD'

    if pred_label == 'HC':
        X = np.array(full_data[:, 1])
        Y = np.array(full_data[:, 0])
        plt.figure(figsize=(8, 8))
        plt.scatter(X, Y, s=75, label='Predicted HC Segment')

        x_idx = []
        y_idx = []
        for idx in pd_idx:
            x_idx.extend( full_data[idx+60:idx+68, 1] )
            y_idx.extend( full_data[idx+60:idx+68, 0] )

        plt.scatter(x_idx, y_idx, s=75, c='#dc3023', label='Predicted PD Segment')

        plt.axis('off')
        # plt.title('point number: %d; True label:%s, Predict label:%s' % (len(full_data), target_labels_dict, pred_labels_dict))
        # plt.legend()
        # plt.show()
        plt.savefig(os.path.join(r'C:\Users\xuecwang\Desktop\pictures\a', 'True_'+true_label+'_'+"Pred_"+pred_label+'_'+json_file_path+'_image.jpg'), dpi=500)
        plt.close()

    if pred_label == 'PD':
        X = np.array(full_data[:, 1])
        Y = np.array(full_data[:, 0])
        plt.figure(figsize=(8, 8))
        plt.scatter(X, Y, s=75, c='#dc3023', label='Predicted PD Segment')

        x_idx = []
        y_idx = []
        for idx in hc_idx:
            x_idx.extend(full_data[idx + 52:idx + 76, 1])
            y_idx.extend(full_data[idx + 52:idx + 76, 0])
        plt.scatter(x_idx, y_idx, s=75, label='Predicted HC Segment')

        plt.axis('off')
        # plt.title('point number: %d; True label:%s, Predict label:%s' % (len(full_data), target_labels_dict, pred_labels_dict))
        # plt.legend()
        # plt.show()
        # plt.savefig(os.path.join(r'C:\Users\xuecwang\Desktop\pictures\a',
        #                          'True_' + true_label + '_' + "Pred_" + pred_label + '_' + json_file_path + '_image.jpg'),
        #             dpi=500)
        # plt.close()


def main():

    parser = argparse.ArgumentParser(description='Testing Parkinson diagnose using deep learning network')
    parser.add_argument('-d', '--datasets', metavar='D', type=str, nargs='?', default='./data/test_new_data',
                        help='Path of testing dataset/image')

    parser.add_argument('-m', '--model-type', metavar='M', type=str, nargs='?', default='lstmcnn',
                        help='The model of deep learning network', dest='model')
    # model_Mnist_best_X128_1_1.pth.tar
    parser.add_argument('-w', '--weights', metavar='W', type=str, default='./checkpoints/lstmcnn/2024_05_06_13_25_44/model_lstmcnn_best_X128.pth.tar',
                        help='The learned training weights', dest='weights')


    parser.add_argument('-c', '--cuda', action='store_true',
                        help='Using CUDA device', dest='cuda')
    parser.add_argument('-p', '--patch_size', metavar='P', type=int, nargs='?', default=128,
                        help='The patch size of input tensor', dest='patch_size')
    parser.add_argument('-b', '--batch_size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batch_size')
    parser.add_argument('-r', '--results', metavar='R', type=str, default='./outputs',
                        help='Saving output results')

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.results):
        os.mkdir(args.results)

    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    if args.cuda:
        torch.cuda.manual_seed(123)
        gpu_list = ','.join(str(i) for i in range(args.ngpus))
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    #cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model == 'MNIST' or args.model == 'Mnist':
        model = Mnist(in_channels=1, n_classes=2)
    elif args.model == 'rnncnn' or args.model == 'RNNCNN':
        model = RNNCNN(in_channels=1, n_classes=2)
    elif args.model == 'lstmcnn' or args.model == 'LstmCnn':
        model = LSTMCNN(in_channels=1, n_classes=2)
    elif args.model == 'grucnn' or args.model == 'GRUCNN':
        model = GRUCNN(in_channels=1, n_classes=2)
    elif args.model == 'pointnet':
        model = PointNet()
    elif args.model == 'mlp':
        model = MLP()
    elif args.model == 'SiT':
        model = SiT(seq_size=128,
                    dim=6, # 序列中每一时刻dim维度（类似词嵌入的维度）
                    num_classes=2,
                    depth=2, # encoder层数
                    heads=2, # encoder中多头注意力机制中头的个数
                    mlp_dim=512, # encoder的后半部分MLP中 将dim维度数映射到mpl_dim维度数
                    dropout=0., # encoder的后半部分MLP中 对input的随机省略概率
                    emb_dropout=0.)  # 每个信号点的特征书，(x,y,p,a,l,p)
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(args.model))
    model.to(device)
    # model = torch.nn.DataParallel(model)

    checkpoint = torch.load(args.weights)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    if_remove = True
    process_on_stroke = False
    stride_size = 32
    patch_size = args.patch_size
    compute_gradient = True
    scale = [100, 100, 1, 1, 1, 1]
    dim_id = [0,1]
    data_path = args.datasets
    pattern_lists = {1}

    transform = transforms.Compose([transforms.ToTensor()])
    # time_list = []
    all_preds_dataset = []
    all_targets_dataset = []
    preds_dataset = []
    preds_dataset_float = []
    targets_dataset = []
    dataset_files = os.listdir(data_path)
    for l, label_path in enumerate(dataset_files):
        if label_path == 'HC':
            label_id = '00'
        elif label_path == 'PD':
            label_id = '01'
        else:
            print('There is no %s class.'%(label_path))
        test_files = os.listdir(os.path.join(data_path, label_path))
        for t, test_path in enumerate(test_files):
            test_id = '{0:03d}'.format(t)
            files = os.listdir(os.path.join(data_path, label_path + '/' + test_path))
            for f, file in enumerate(files):
                file_id = '{0:03d}'.format(f)
                pattern_id = int(file[7:8])
                if pattern_id in pattern_lists:
                    full_file_name = data_path + '/' + label_path + '/' + test_path + '/' + file
                    full_data, patch_idx, patches_data = get_patches_from_sequence(full_file_name, patch_size, stride_size, compute_gradient, process_on_stroke, scale, dim_id, if_remove)
                    patches_label = label_id

                    if not patches_data or len(patches_data)<args.batch_size:
                        print('Empty data:')
                        print(full_file_name)
                    else:
                        dataset = DatasetList(data=patches_data, label=patches_label, transform=transform)
                        test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

                        pred_labels = []
                        target_labels = []
                        testing_bar = tqdm(test_dataloader)

                        # start = time.clock()

                        for j, data in enumerate(testing_bar):
                            inputs, targets = data
                            if torch.cuda.is_available():
                                inputs = inputs.cuda()
                                targets = targets.cuda()

                            if args.model == 'pointnet':
                                outputs,_ = model(inputs)
                            else:
                                outputs = model(inputs)


                            _, preds = torch.max(outputs, 1)

                            pred_data = preds.data.cpu().detach().numpy().flatten()
                            target_data = targets.data.cpu().detach().numpy().flatten()
                            for k in np.arange(len(inputs)):
                                pred_labels.append(pred_data[k])
                                target_labels.append(target_data[k])

                        # end = time.clock()
                        # print (str(end - start))
                        # time_list.append(np.float(end-start))

                        target_labels = np.array(target_labels)
                        target_ar, target_num = np.unique(target_labels, return_counts=True)
                        target_labels_dict = dict(zip(target_ar,target_num))
                        all_targets_dataset.extend(target_labels)

                        pred_labels = np.array(pred_labels)
                        pred_ar, pred_num = np.unique(pred_labels, return_counts=True)
                        pred_labels_dict = dict(zip(pred_ar,pred_num))

                        # if len(pred_labels_dict)==1:
                        #     if 1 in pred_labels_dict.keys() and label_id == '01':
                        #         pro_float = 1.0
                        #     elif 0 in pred_labels_dict.keys() and label_id == '00':
                        #         pro_float = 1.0
                        #     else:
                        #         pro_float = 0.0
                        # elif label_id == '00':
                        #     pro_float = pred_labels_dict[0]/(pred_labels_dict[0]+pred_labels_dict[1])
                        # elif label_id == '01':
                        #     pro_float = pred_labels_dict[1]/(pred_labels_dict[0]+pred_labels_dict[1])

                        # 计算PD占比阈值变化曲线
                        if len(pred_labels_dict) == 1:
                            if 1 in pred_labels_dict.keys():
                                pro_float = 1.0
                            elif 0 in pred_labels_dict.keys():
                                pro_float = 0.0
                        else:
                            pro_float = pred_labels_dict[1] / (pred_labels_dict[0] + pred_labels_dict[1])

                        all_preds_dataset.extend(pred_labels)

                        targets_dataset.append(np.argmax(np.bincount(target_labels)))
                        preds_dataset.append(np.argmax(np.bincount(pred_labels)))
                        preds_dataset_float.append(pro_float)


                        if np.sum(np.argmax(np.bincount(target_labels)) - np.argmax(np.bincount(pred_labels))):
                            print('Incorrect classification: %s' % (full_file_name))
                            print('     True label:%s, Predict label:%s, %f' % (target_labels_dict, pred_labels_dict, pro_float))
                        else:
                            print('Correct classification: %s' % (full_file_name))
                            print('     True label:%s, Predict label:%s, %f' % (target_labels_dict, pred_labels_dict, pro_float))

                        # # 画图：以二维的形式展示出哪些部分是错误分类的
                        # pd_idx = patch_idx[np.where(pred_labels==1)]
                        # hc_idx = patch_idx[np.where(pred_labels==0)]
                        # image_test_result(full_data, pd_idx, hc_idx, file, target_labels_dict,
                        #                   pred_labels_dict, label_id)

    # print("s/sample time : %f"%(mean(time_list)))

    targets_dataset = np.array(targets_dataset)
    preds_dataset = np.array(preds_dataset)
    print(targets_dataset)
    print(preds_dataset)
    print(targets_dataset - preds_dataset)
    print(preds_dataset_float)
    metric = Performance(targets_dataset, preds_dataset)
    metric.roc_plot()
    metric.plot_matrix()
    acc_score = metric.accuracy()
    f1_score = metric.f1_score()
    recall_score = metric.recall()
    precision_score = metric.presision()
    specificity = metric.specificity()
    npv = metric.npv()
    mcc = metric.mcc()
    print("Method1: Accuracy(ACC) = {:f}, F1_score = {:f}, Recall(Sensitivity,TPR) = {:f}, and Precision(PPV) = {:f}, and NPV = {:f}, and Specificity(TNR) = {:f}, "
          "and Matthews correlation coefficient(MCC) = {:f}. \n".format(acc_score, f1_score, recall_score, precision_score, npv,specificity, mcc))

    all_targets_dataset = np.array(all_targets_dataset)
    all_preds_dataset = np.array(all_preds_dataset)
    all_metric = Performance(all_targets_dataset, all_preds_dataset)
    all_metric.roc_plot()
    all_metric.plot_matrix()
    all_acc_score = all_metric.accuracy()
    all_f1_score = all_metric.f1_score()
    all_recall_score = all_metric.recall()
    all_precision_score = all_metric.presision()
    all_specificity = all_metric.specificity()
    all_npv = all_metric.npv()
    all_mcc = all_metric.mcc()
    print("Method2: Accuracy(ACC) = {:f}, F1_score = {:f}, Recall(Sensitivity,TPR) = {:f}, and Precision(PPV) = {:f}, and NPV = {:f}, and Specificity(TNR) = {:f}, and Matthews correlation coefficient(MCC) = {:f}. \n".format(all_acc_score, all_f1_score, all_recall_score, all_precision_score, all_npv, all_specificity, all_mcc))

    # np.savetxt(os.path.join(args.results, 'pre_acc_spiral_5.txt'), np.array(preds_dataset_float), fmt='%0.8f')
    # np.savetxt(os.path.join(args.results, 'tru_label_spiral_1.txt'), np.array(targets_dataset), )


if __name__ == '__main__':
    main()
