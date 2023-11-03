import re
import os
import argparse
from models import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import DatasetList
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score
from data_utils.utils import *
from estimation import  Performance
from models import Mnist, LSTMCNN
import time

def get_patches_from_sequence(full_file_name, patch_size, stride_size,  compute_gradient, process_on_stroke, scale, dim_id):

    start_time_21 = time.time()
    with open(full_file_name) as curr_file:
        test_data = json.load(curr_file)
    end_time_21 = time.time()
    print("    Time-consuming data import: {:.10f}s".format(end_time_21 - start_time_21))

    start_time_22 = time.time()
    frame_data = test_data['data']
    patch_dataset = []
    index = ['a', 'l', 'p', 'x', 'y']
    full_data_array = get_full_data_array(frame_data, index)
    print('data length:%d' % (len(full_data_array[:,0])))
    data_scale = get_column_data_scale(full_data_array)

    if np.isnan(np.sum(full_data_array)):
        print('The data has Nan value (%s)!'%(full_file_name))
    else:
        if process_on_stroke:
            if (np.count_nonzero(data_scale[0,:]- data_scale[1,:])==5):
                for j, stroke_idx in enumerate(frame_data):
                    patch_data = []
                    stroke = pd.DataFrame(stroke_idx)
                    stroke_frame = stroke[index]
                    stroke_data = stroke_frame.to_numpy()
                    stroke_data = normalization_array_data(stroke_data, data_scale)
                    if compute_gradient:
                        stroke_data = get_colomn_scalled_difference(stroke_data, order=1, scale=scale, dim_id=dim_id)
                    if not np.isnan(np.sum(stroke_data)):
                        patch_data = get_random_sampling_patches(stroke_data, patch_size, stride_size)
                    else:
                        print('The data after the first difference has Nan value (%s)!' % (full_file_name))
                    patch_dataset.extend(patch_data)
            else:
                print('There are some eigenvalues in the data that do not change (%s).'%(full_file_name))
        else:
            if (np.count_nonzero(data_scale[0, :] - data_scale[1, :]) == 5):

                full_data_array = normalization_array_data(full_data_array, data_scale)

                if compute_gradient:
                    full_data_array = get_colomn_scalled_difference(full_data_array, order=1, scale=scale, dim_id=dim_id)

                if not np.isnan(np.sum(full_data_array)):
                    patch_dataset = get_random_sampling_patches(full_data_array, patch_size, stride_size)

                else:
                    print('The data has Nan value (%s)!' % (full_file_name))
            else:
                print('There are some eigenvalues in the data that do not change (%s).'%(full_file_name))
    end_time_22 = time.time()
    print("    Time-consuming data segmentation : {:.10f}s".format(end_time_22 - start_time_22))

    return patch_dataset

def main():
    parser = argparse.ArgumentParser(description='Testing Parkinson diagnose using deep learning network')

    parser.add_argument('-m', '--model-type', metavar='M', type=str, nargs='?', default='lstmcnn',
                        help='The model of deep learning network', dest='model')
    parser.add_argument('-w', '--weights', metavar='W', type=str, default='./checkpoints/lstmcnn/2022_07_20_12_16_03/model_lstmcnn_best_X128.pth.tar',
                        help='The learned training weights', dest='weights')
    parser.add_argument('-c', '--cuda', action='store_true',
                        help='Using CUDA device', dest='cuda')

    args = parser.parse_args()
    print(args)

    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    if args.cuda:
        torch.cuda.manual_seed(123)
        gpu_list = ','.join(str(i) for i in range(args.ngpus))
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    stride_size = 64
    patch_size = 128
    compute_gradient = True
    process_on_stroke = False
    scale = [1, 1, 1, 100, 100]
    dim_id = [3, 4]
    transform = transforms.Compose([transforms.ToTensor()])
    label_id = '01'
    # full_file_name = './data/raw_data/PD/PD5/PD-5_88BFC4E0_ptrace_2017-12-04_10_27_53___4e9b1599544746cf8286bf18a75c6b8a.json' # 18618
    # full_file_name = './data/raw_data/PD/PD3/PD-3_0B4B2BFF_pcopy_2017-11-24_11_48_42___56467994ac4e455c8f4600581fd15b4b.json' # 12787
    # full_file_name = './data/raw_data/PD/PD12/PD-12_1FD2AD0D_ptrace_2018-02-27_09_07_57___33e3377f7c864dce9a41398dead728e5.json' # 9471
    # full_file_name = './data/raw_data/KT/KT5/KT-5_192802B7_plcontinue_2017-11-24_10_51_50___1d46aca2a9ca485e95710a28aa26da80.json' # 7122
    # full_file_name = './data/raw_data/PD/PD13/PD-13_01A103F6_pcontinue_2018-03-02_12_58_43___76ad3a94f1f4478e8e7b5eb4dfef7cb0.json' # 3128
    # full_file_name = './data/raw_data/KT/KT1/KT-01_8D6834FB_pcontinue_2017-11-05_16_45_31___0095aa3c709f457cbe43e800ee7a299f.json' # 1175
    full_file_name = './data/raw_data/KT/KT117/20190526-161632-KT117-pcontinue_103339cd-3962-4b4e-83e3-0e71d0c29972.json' # 677





    start_time_1 = time.time()
    device = 'cpu'
    if args.model == 'MNIST' or args.model == 'Mnist':  # PSNR-oriented super resolution
        model = Mnist(in_channels=1, n_classes=2)
    elif args.model == 'lstmcnn' or args.model == 'LstmCnn':
        model = LSTMCNN(in_channels=1, n_classes=2)
    model.to(device)
    checkpoint = torch.load(args.weights)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    end_time_1 = time.time()
    print("Time-consuming model import: {:.10f}s".format(end_time_1 - start_time_1))


    start_time_2 = time.time()
    patches_data = get_patches_from_sequence(full_file_name, patch_size, stride_size, compute_gradient, process_on_stroke, scale, dim_id)
    end_time_2 = time.time()
    print("Time-consuming data preprocessing: {:.10f}s".format(end_time_2 - start_time_2))




    start_time_3 = time.time()
    dataset = DatasetList(data=patches_data, label=label_id, transform=transform)
    test_dataloader = DataLoader(dataset, batch_size=len(patches_data), shuffle=False, pin_memory=True, drop_last=False)
    pred_labels = []
    for j, data in enumerate(test_dataloader):
        inputs, _ = data
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        pred_data = preds.data.cpu().detach().numpy().flatten()
        pred_labels.extend(pred_data)
    end_time_3 = time.time()
    print("Time-consuming model prediction: {:.10f}s".format(end_time_3 - start_time_3))

    if int(np.argmax(np.bincount(pred_labels))) == 0:
        print("Model prediction results：HC")
    else:
        print("Model prediction results：PD")


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("total time spent: {:.10f}s".format(end_time - start_time))
