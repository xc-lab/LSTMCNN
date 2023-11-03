''''
PD:1, KT:0  4.008612
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
from collections import Counter




def main():

    parser = argparse.ArgumentParser(description='Testing Parkinson diagnose using deep learning network')
    parser.add_argument('-d', '--datasets', metavar='D', type=str, nargs='?', default='./data/test_data',
                        help='Path of testing dataset/image')

    parser.add_argument('-m', '--model-type', metavar='M', type=str, nargs='?', default='rnncnn',
                        help='The model of deep learning network', dest='model')
    # model_Mnist_best_X128_1_1.pth.tar
    parser.add_argument('-w', '--weights', metavar='W', type=str, default='./checkpoints/rnncnn/2023_03_07_20_00_05/model_rnncnn_best_X128.pth.tar',
                        help='The learned training weights', dest='weights')

    parser.add_argument('-c', '--cuda', action='store_true',
                        help='Using CUDA device', dest='cuda')
    parser.add_argument('-p', '--patch_size', metavar='P', type=int, nargs='?', default=128,
                        help='The patch size of input tensor', dest='patch_size')
    parser.add_argument('-b', '--batch_size', metavar='B', type=int, nargs='?', default=8,
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
    elif args.model == 'mlp':
        model = MLP()
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(args.model))
    model.to(device)
    # model = torch.nn.DataParallel(model)

    checkpoint = torch.load(args.weights)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    stride_size = 64
    patch_size = args.patch_size
    compute_gradient = True
    process_on_stroke = False
    scale = [1, 1, 1, 100, 100]
    dim_id = [3,4]
    data_path = args.datasets
    pattern_lists = {'plcontinue', 'plcopy',  'pltrace'}

    transform = transforms.Compose([transforms.ToTensor()])
    # time_list = []
    all_preds_dataset = []
    all_targets_dataset = []
    preds_dataset = []
    preds_dataset_float = []
    targets_dataset = []
    dataset_files = os.listdir(data_path)
    for l, label_path in enumerate(dataset_files):
        if label_path == 'KT':
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
                for pattern in pattern_lists:
                    if re.search(pattern, file):
                        pattern_id = pattern
                        full_file_name = data_path + '/' + label_path + '/' + test_path + '/' + file
                        patches_data = get_patches_from_sequence(full_file_name, patch_size, stride_size, compute_gradient, process_on_stroke, scale, dim_id)
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

                            if len(pred_labels_dict)==1:
                                if 1 in pred_labels_dict.keys() and label_id == '01':
                                    pro_float = 1.0
                                elif 0 in pred_labels_dict.keys() and label_id == '00':
                                    pro_float = 1.0
                                else:
                                    pro_float = 0.0
                            elif label_id == '00':
                                pro_float = pred_labels_dict[0]/(pred_labels_dict[0]+pred_labels_dict[1])
                            elif label_id == '01':
                                pro_float = pred_labels_dict[1]/(pred_labels_dict[0]+pred_labels_dict[1])

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

    # np.savetxt(os.path.join(args.results, 'pre_acc_pl_5.txt'), np.array(preds_dataset_float), fmt='%0.8f')
    # np.savetxt(os.path.join(args.results, 'tru_label_p_1.txt'), np.array(targets_dataset), )


if __name__ == '__main__':
    main()
