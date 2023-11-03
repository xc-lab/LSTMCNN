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
from models import Mnist, AlexNet


def main():
    parser = argparse.ArgumentParser(description='Testing Parkinson diagnose using deep learning network')
    parser.add_argument('-d', '--datasets', metavar='D', type=str, nargs='?', default='./data/testing1',
                        help='Path of testing dataset/image')
    parser.add_argument('-m', '--model-type', metavar='M', type=str, nargs='?', default='Mnist',
                        help='The model of deep learning network', dest='model')
    parser.add_argument('-w', '--weights', metavar='W', type=str, default='./checkpoints/best_results/model_best_X128_1.pth.tar',
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
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    #cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model == 'MNIST' or args.model == 'Mnist':  # PSNR-oriented super resolution
        model = Mnist(in_channels=1, n_classes=2)
    else:
        model = AlexNet(in_channels=1, n_classes=2)

    model.to(device)
    # model = torch.nn.DataParallel(model)
    checkpoint = torch.load(args.weights)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    stride_size = 1
    patch_size = 128
    compute_gradient = True
    process_on_stroke = True
    scale = [10, 1, 1, 50, 10]
    dim_id = [0, 3, 4]
    # scale = [10, 10, 1, 50, 10]
    # dim_id = [0, 1, 3, 4]
    transform = transforms.Compose([transforms.ToTensor()])

    label_id = '01'
    full_file_name = './data/raw_data/KT/KT118/20190526-132706-KT118-plcopy_eeeaf2a0-c363-4a20-9888-44afbdffb73a.json'
    patches_data = get_patches_from_sequence(full_file_name, patch_size, stride_size, compute_gradient, process_on_stroke, scale, dim_id)
    patches_label = label_id

    if not patches_data:
        print(full_file_name)
    else:

        dataset = DatasetList(data=patches_data, label=patches_label, transform=transform)
        test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

        pred_labels = []
        target_labels = []
        testing_bar = tqdm(test_dataloader)
        for j, data in enumerate(testing_bar):
            inputs, targets = data
            inputs = inputs.cuda()
            targets = targets.cuda()

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            pred_data = preds.data.cpu().detach().numpy().flatten()
            target_data = targets.data.cpu().detach().numpy().flatten()
            for k in np.arange(len(inputs)):
                pred_labels.append(pred_data[k])
                target_labels.append(target_data[k])

        target_labels = np.array(target_labels)
        pred_labels = np.array(pred_labels)
        print(pred_labels)
        print(target_labels)


if __name__ == '__main__':
    main()
