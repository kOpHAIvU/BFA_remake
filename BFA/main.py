from __future__ import division
from __future__ import absolute_import

import os, sys, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, clustering_loss, change_quan_bitwidth
from tensorboardX import SummaryWriter
import models
from models.quantization import quan_Conv2d, quan_Linear, quantize

from attack.BFA import *
import torch.nn.functional as F
import copy

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.nn import init
import math

class _quantize_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, step_size, half_lvls):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.step_size = step_size
        ctx.half_lvls = half_lvls
        output = F.hardtanh(input,
                            min_val=-ctx.half_lvls * ctx.step_size.item(),
                            max_val=ctx.half_lvls * ctx.step_size.item())

        output = torch.round(output / ctx.step_size)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone() / ctx.step_size

        return grad_input, None, None

quantize = _quantize_func.apply

class _bin_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, mu):
        ctx.mu = mu
        output = input.clone().zero_()
        output[input.ge(0)] = 1
        output[input.lt(0)] = -1

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone() / ctx.mu
        return grad_input, None

w_bin = _bin_func.apply

class CustomBlock(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomBlock, self).__init__()  # Bỏ tham số vào đây
        self.N_bits = 8
        self.full_lvls = 2 ** self.N_bits
        self.half_lvls = (self.full_lvls - 2) / 2

        # Initialize the step size
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)

        # Thêm khởi tạo trọng số
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))  # Khởi tạo trọng số
        self.__reset_stepsize__()  # Khởi tạo bước

        # flag để cho phép suy diễn với trọng số đã định lượng
        self.inf_with_weight = False  # disabled by default

        # create a vector to identify the weight to each bit
        self.b_w = nn.Parameter(2 ** torch.arange(start=self.N_bits - 1,
                                                  end=-1,
                                                  step=-1).unsqueeze(-1).float(),
                                requires_grad=False)

        self.b_w[0] = -self.b_w[0]  # in-place reverse

        # Khởi tạo trọng số
        self.reset_parameters()

    def reset_parameters(self):
        # Hàm khởi tạo trọng số
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, input):
        if self.inf_with_weight:
            weight_applied = self.weight * self.step_size
        else:
            self.__reset_stepsize__()
            weight_quan = quantize(self.weight, self.step_size, self.half_lvls) * self.step_size
            weight_applied = weight_quan

        # Apply softmax instead of linear transformation
        output = F.softmax(input @ weight_applied.T, dim=-1)
        return output

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = self.weight.abs().max() / self.half_lvls

    def __reset_weight__(self):
        '''
        This function will reconstruct the weight stored in self.weight.
        Replacing the original floating-point with the quantized fix-point
        weight representation.
        '''
        # replace the weight with the quantized version
        with torch.no_grad():
            self.weight.data = quantize(self.weight, self.step_size,
                                        self.half_lvls)
        # enable the flag, thus now computation does not invovle weight quantization
        self.inf_with_weight = True

class quan_Conv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(quan_Conv1d, self).__init__(in_channels,
                                          out_channels,
                                          kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=bias)

        # Số lượng bit để lượng tử hóa trọng số
        self.N_bits = 8
        self.full_lvls = 2 ** self.N_bits
        self.half_lvls = (self.full_lvls - 2) / 2

        # Bước lượng tử hóa (step size), là một tham số có thể học được
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.__reset_stepsize__()

        # Cờ để bật hoặc tắt sử dụng trọng số lượng tử hóa
        self.inf_with_weight = False  # Tắt theo mặc định

        # Tạo một vector để biểu diễn trọng số cho từng bit
        self.b_w = nn.Parameter(2 ** torch.arange(start=self.N_bits - 1,
                                                  end=-1,
                                                  step=-1).unsqueeze(-1).float(),
                                requires_grad=False)
        self.b_w[0] = -self.b_w[0]  # Biến đổi MSB thành giá trị âm để hỗ trợ bù hai

    def __reset_stepsize__(self):
        """Hàm này dùng để đặt lại giá trị `step_size`."""
        # Giá trị này có thể được tùy chỉnh tùy thuộc vào yêu cầu của mô hình
        self.step_size.data.fill_(1.0)

    def forward(self, x):
        # Kiểm tra cờ `inf_with_weight` để quyết định sử dụng trọng số đã lượng tử hóa hay không
        if self.inf_with_weight:
            quantized_weight = self.quantize_weight(self.weight)
            return nn.functional.conv1d(x, quantized_weight, self.bias, self.stride,
                                        self.padding, self.dilation, self.groups)
        else:
            return nn.functional.conv1d(x, self.weight, self.bias, self.stride,
                                        self.padding, self.dilation, self.groups)

    def quantize_weight(self, weight):
        """Lượng tử hóa trọng số theo số bit đã định."""
        # Tạo trọng số lượng tử hóa bằng cách sử dụng step_size
        quantized_weight = torch.round(weight / self.step_size) * self.step_size
        quantized_weight = torch.clamp(quantized_weight, -self.half_lvls * self.step_size,
                                       (self.half_lvls - 1) * self.step_size)
        return quantized_weight


class CustomModel(nn.Module):
    def __init__(self, input_size=69, hidden_size1=128, hidden_size2=64, hidden_size3=32, hidden_size4=16, hidden_size5=128, output_size=9):
        super(CustomModel, self).__init__()
        self.fc1 = quan_Conv1d(input_size, hidden_size1, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.stage_1 = quan_Conv1d(hidden_size1, hidden_size2, kernel_size=3, stride=2, padding=1)
        self.stage_1_1 = quan_Conv1d(hidden_size2, hidden_size2, kernel_size=3, stride=2, padding=1)

        self.stage_2 = quan_Conv1d(hidden_size2, hidden_size3, kernel_size=3, stride=2, padding=1)
        self.stage_2_1 = quan_Conv1d(hidden_size3, hidden_size3, kernel_size=3, stride=2, padding=1)

        self.stage_3 = quan_Conv1d(hidden_size3, hidden_size4, kernel_size=3, stride=2, padding=1)
        self.stage_3_1 = quan_Conv1d(hidden_size4, hidden_size4, kernel_size=3, stride=2, padding=1)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.4)  # Dropout layer to reduce overfitting

        self.classifier = CustomBlock(hidden_size4, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.pool(x), inplace=True)

        x = self.stage_1(x)
        x = F.relu(self.pool(x), inplace=True)

        x = self.stage_1_1(x)
        x = F.relu(self.pool(x), inplace=True)

        x = self.stage_2(x)
        x = F.relu(self.pool(x), inplace=True)

        x = self.stage_2_1(x)
        x = F.relu(self.pool(x), inplace=True)

        x = self.stage_3(x)
        x = F.relu(self.pool(x), inplace=True)

        x = self.stage_3_1(x)
        x = F.relu(self.pool(x), inplace=True)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        return self.classifier(x)


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

################# Options ##################################################
############################################################################
parser = argparse.ArgumentParser(
    description='Training network for image classification',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_path',
                    default='/home/elliot/data/pytorch/svhn/',
                    type=str,
                    help='Path to dataset')
parser.add_argument(
    '--dataset',
    type=str,
    choices=['inid', 'cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10', 'mnist'],
    help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch',
                    metavar='ARCH',
                    default='lbcnn',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs',
                    type=int,
                    default=200,
                    help='Number of epochs to train.')
parser.add_argument('--optimizer',
                    type=str,
                    default='SGD',
                    choices=['SGD', 'Adam', 'YF'])
parser.add_argument('--test_batch_size',
                    type=int,
                    default=256,
                    help='Batch size.')
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.001,
                    help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay',
                    type=float,
                    default=1e-4,
                    help='Weight decay (L2 penalty).')
parser.add_argument('--schedule',
                    type=int,
                    nargs='+',
                    default=[80, 120],
                    help='Decrease learning rate at these epochs.')
parser.add_argument(
    '--gammas',
    type=float,
    nargs='+',
    default=[0.1, 0.1],
    help=
    'LR is multiplied by gamma on schedule, number of gammas should be equal to schedule'
)
# Checkpoints
parser.add_argument('--print_freq',
                    default=100,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 200)')
parser.add_argument('--save_path',
                    type=str,
                    default='./save/',
                    help='Folder to save checkpoints and log.')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument(
    '--fine_tune',
    dest='fine_tune',
    action='store_true',
    help='fine tuning from the pre-trained model, force the start epoch be zero'
)
parser.add_argument('--model_only',
                    dest='model_only',
                    action='store_true',
                    help='only save the model without external utils_')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--gpu_id',
                    type=int,
                    default=0,
                    help='device range [0,ngpu-1]')
parser.add_argument('--workers',
                    type=int,
                    default=4,
                    help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
# quantization
parser.add_argument(
    '--quan_bitwidth',
    type=int,
    default=None,
    help='the bitwidth used for quantization')
parser.add_argument(
    '--reset_weight',
    dest='reset_weight',
    action='store_true',
    help='enable the weight replacement with the quantized weight')
# Bit Flip Attack
parser.add_argument('--bfa',
                    dest='enable_bfa',
                    action='store_true',
                    help='enable the bit-flip attack')
parser.add_argument('--attack_sample_size',
                    type=int,
                    default=128,
                    help='attack sample size')
parser.add_argument('--n_iter',
                    type=int,
                    default=20,
                    help='number of attack iterations')
parser.add_argument(
    '--k_top',
    type=int,
    default=None,
    help='k weight with top ranking gradient used for bit-level gradient check.'
)
parser.add_argument('--random_bfa',
                    dest='random_bfa',
                    action='store_true',
                    help='perform the bit-flips randomly on weight bits')

# Piecewise clustering
parser.add_argument('--clustering',
                    dest='clustering',
                    action='store_true',
                    help='add the piecewise clustering term.')
parser.add_argument('--lambda_coeff',
                    type=float,
                    default=1e-3,
                    help='lambda coefficient to control the clustering term')

##########################################################################

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.ngpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        args.gpu_id)  # make only device #gpu_id visible, then

args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()  # check GPU

# Give a random seed if no manual configuration
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

cudnn.benchmark = True

###############################################################################
###############################################################################


def main():
    # Init logger6
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(
        os.path.join(args.save_path,
                     'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')),
              log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()),
              log)

    # Init the tensorboard path and writer
    tb_path = os.path.join(args.save_path, 'tb_log',
                           'run_' + str(args.manualSeed))
    # logger = Logger(tb_path)
    writer = SummaryWriter(tb_path)

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif args.dataset == 'svhn':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'mnist':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif args.dataset == 'inid':
        print("Inid dataset import sucsess!")
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    if args.dataset == 'imagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])  # here is actually the validation dataset
    elif args.dataset == 'inid':
        print("Inid dataset import sucsess!")
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])

    if args.dataset == 'mnist':
        train_data = dset.MNIST(args.data_path,
                                train=True,
                                transform=train_transform,
                                download=True)
        test_data = dset.MNIST(args.data_path,
                               train=False,
                               transform=test_transform,
                               download=True)
        num_classes = 10
    elif args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path,
                                  train=True,
                                  transform=train_transform,
                                  download=True)
        test_data = dset.CIFAR10(args.data_path,
                                 train=False,
                                 transform=test_transform,
                                 download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path,
                                   train=True,
                                   transform=train_transform,
                                   download=True)
        test_data = dset.CIFAR100(args.data_path,
                                  train=False,
                                  transform=test_transform,
                                  download=True)
        num_classes = 100
    elif args.dataset == 'svhn':
        train_data = dset.SVHN(args.data_path,
                               split='train',
                               transform=train_transform,
                               download=True)
        test_data = dset.SVHN(args.data_path,
                              split='test',
                              transform=test_transform,
                              download=True)
        num_classes = 10
    elif args.dataset == 'stl10':
        train_data = dset.STL10(args.data_path,
                                split='train',
                                transform=train_transform,
                                download=True)
        test_data = dset.STL10(args.data_path,
                               split='test',
                               transform=test_transform,
                               download=True)
        num_classes = 10
    elif args.dataset == 'inid':
        print("Inid dataset import sucsess!")
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    if args.dataset == 'inid':
        data = pd.read_csv(r"D:\UIT\NCKH\Dataset\IoT_Network_Intrusion_Dataset\IoT_Network_Intrusion_Dataset.csv", skipinitialspace=True)
        data = data.drop_duplicates()
        data = data.drop(columns=['Flow_ID', 'Src_IP', 'Dst_IP', 'Timestamp'])
        data = data.drop(columns=['Fwd_PSH_Flags', 'Fwd_URG_Flags', 'Fwd_Byts/b_Avg', 'Fwd_Pkts/b_Avg',
                                  'Fwd_Blk_Rate_Avg', 'Bwd_Byts/b_Avg', 'Bwd_Pkts/b_Avg', 'Bwd_Blk_Rate_Avg',
                                  'Init_Fwd_Win_Byts', 'Fwd_Seg_Size_Min'])

        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(0, inplace=True)
        data = data.drop_duplicates()
        dataLabel = data[['Sub_Cat']]
        data = data.drop(columns=['Label', 'Cat', 'Sub_Cat'])

        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        # Tách dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(data, dataLabel, test_size=0.2, random_state=100)

        # Chuyển đổi y_train và y_test thành mã số
        y_train, _ = pd.factorize(y_train['Sub_Cat'])  # Chuyển đổi thành mã số
        y_test, _ = pd.factorize(y_test['Sub_Cat'])  # Chuyển đổi thành mã số

        # Kiểm tra kiểu dữ liệu của y_train và y_test
        print("Kiểu dữ liệu y_train:", y_train.dtype)
        print("Kiểu dữ liệu y_test:", y_test.dtype)

        # Tạo DataLoader cho tập huấn luyện
        train_loader = DataLoader(
            torch.utils.data.TensorDataset(
                torch.FloatTensor(X_train),
                torch.LongTensor(y_train)  # y_train đã được chuyển đổi thành mã số
            ),
            batch_size=512,
            num_workers=5,
            shuffle=True,
            pin_memory=True
        )

        # Tạo DataLoader cho tập kiểm tra
        test_loader = DataLoader(
            torch.utils.data.TensorDataset(
                torch.FloatTensor(X_test),
                torch.LongTensor(y_test)  # y_test đã được chuyển đổi thành mã số
            ),
            batch_size=512,
            num_workers=5,
            shuffle=True,
            pin_memory=True
        )

    else:
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.attack_sample_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=args.test_batch_size,
                                                  shuffle=True,
                                                  num_workers=args.workers,
                                                  pin_memory=True)

    print_log("=> creating model '{}'".format(args.arch), log)

    # Init model, criterion, and optimizer
    if args.dataset == 'inid':
        # Khởi tạo mô hình, criterion và optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = CustomModel().to(device)
    else:
        net = models.__dict__[args.arch](num_classes)
    print_log("=> network :\n {}".format(net), log)

    if args.use_cuda:
        if args.ngpu > 1:
            net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # separate the parameters thus param groups can be updated by different optimizer
    all_param = [
        param for name, param in net.named_parameters()
        if not 'step_size' in name
    ]

    step_param = [
        param for name, param in net.named_parameters() if 'step_size' in name 
    ]

    if args.optimizer == "SGD":
        print("using SGD as optimizer")
        optimizer = torch.optim.SGD(all_param,
                                    lr=0.01,
                                    momentum=0.9,
                                    weight_decay=0.0001,
                                    nesterov=True)

    elif args.optimizer == "Adam":
        print("using Adam as optimizer")
        optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad,
                                            all_param),
                                     lr=0.01,
                                     momentum=0.9,
                                     weight_decay=0.0001)

    elif args.optimizer == "RMSprop":
        print("using RMSprop as optimizer")
        optimizer = torch.optim.RMSprop(
            filter(lambda param: param.requires_grad, net.parameters()),
            lr=0.01,
            alpha=0.99,
            eps=1e-08,
            momentum=0.9,
            weight_decay=0.0001)

    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    recorder = RecorderMeter(args.epochs)  # count number of epoches

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            if not (args.fine_tune):
                args.start_epoch = checkpoint['epoch']
                recorder = checkpoint['recorder']
                optimizer.load_state_dict(checkpoint['optimizer'])

            state_tmp = net.state_dict()
            if 'state_dict' in checkpoint.keys():
                state_tmp.update(checkpoint['state_dict'])
            else:
                state_tmp.update(checkpoint)

            net.load_state_dict(state_tmp, strict=False)

            print_log(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, args.start_epoch), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume),
                      log)
    else:
        print_log(
            "=> do not use any checkpoint for {} model".format(args.arch), log)
        
    # Configure the quantization bit-width
    if args.quan_bitwidth is not None:
        change_quan_bitwidth(net, args.quan_bitwidth)

    # update the step_size once the model is loaded. This is used for quantization.
    for m in net.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear) or isinstance(m, CustomBlock) or m.__class__.__name__ == "CustomBlock" or m.__class__.__name__ == "quan_Conv1d":
            # simple step size update based on the pretrained model or weight init
            m.__reset_stepsize__()

    # block for weight reset
    if args.reset_weight:
        for m in net.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear) or isinstance(m, CustomBlock):
                m.__reset_weight__()
                # print(m.weight)

    attacker = BFA(criterion, net, args.k_top)
    net_clean = copy.deepcopy(net)
    # weight_conversion(net)

    if args.enable_bfa:
        perform_attack(attacker, net, net_clean, train_loader, test_loader,
                       args.n_iter, log, writer, csv_save_path=args.save_path,
                       random_attack=args.random_bfa)
        return

    if args.evaluate:
        print("Evaluate mode")
        _,_,_, output_summary = validate(test_loader, net, criterion, log, summary_output=True)
        pd.DataFrame(output_summary).to_csv(os.path.join(args.save_path, 'output_summary_{}.csv'.format(args.arch)),
                                            header=['top-1 output'], index=False)
        return

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()

    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate, current_momentum = adjust_learning_rate(
            optimizer, epoch, args.gammas, args.schedule)
        # Display simulation time
        need_hour, need_mins, need_secs = convert_secs2time(
            epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(
            need_hour, need_mins, need_secs)

        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f}][M={:1.2f}]'.format(time_string(), epoch, args.epochs,
                                                                                   need_time, current_learning_rate,
                                                                                   current_momentum) \
            + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                               100 - recorder.max_accuracy(False)), log)

        # train for one epoch
        train_acc, train_los = train(train_loader, net, criterion, optimizer,
                                     epoch, log)

        # evaluate on validation set
        val_acc, _, val_los = validate(test_loader, net, criterion, log)
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        is_best = val_acc >= recorder.max_accuracy(False)

        if args.model_only:
            checkpoint_state = {'state_dict': net.state_dict}
        else:
            checkpoint_state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'recorder': recorder,
                'optimizer': optimizer.state_dict(),
            }

        save_checkpoint(checkpoint_state, is_best, args.save_path,
                        'checkpoint.pth.tar', log)

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))

        # save addition accuracy log for plotting
        accuracy_logger(base_dir=args.save_path,
                        epoch=epoch,
                        train_accuracy=train_acc,
                        test_accuracy=val_acc)

        # ============ TensorBoard logging ============#

        ## Log the graidents distribution
        for name, param in net.named_parameters():
            name = name.replace('.', '/')
            try:
                writer.add_histogram(name + '/grad',
                                    param.grad.clone().cpu().data.numpy(),
                                    epoch + 1,
                                    bins='tensorflow')
            except:
                pass

            try:
                writer.add_histogram(name, param.clone().cpu().data.numpy(),
                                      epoch + 1, bins='tensorflow')
            except:
                pass

        total_weight_change = 0

        for name, module in net.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                try:
                    writer.add_histogram(name+'/bin_weight', module.bin_weight.clone().cpu().data.numpy(), epoch + 1,
                                        bins='tensorflow')
                    writer.add_scalar(name + '/bin_weight_change', module.bin_weight_change, epoch+1)
                    total_weight_change += module.bin_weight_change
                    writer.add_scalar(name + '/bin_weight_change_ratio', module.bin_weight_change_ratio, epoch+1)
                except:
                    pass

        writer.add_scalar('total_weight_change', total_weight_change, epoch + 1)
        print('total weight changes:', total_weight_change)

        writer.add_scalar('loss/train_loss', train_los, epoch + 1)
        writer.add_scalar('loss/test_loss', val_los, epoch + 1)
        writer.add_scalar('accuracy/train_accuracy', train_acc, epoch + 1)
        writer.add_scalar('accuracy/test_accuracy', val_acc, epoch + 1)
        # ============ TensorBoard logging ============#

    log.close()

def perform_attack(attacker, model, model_clean, train_loader, test_loader,
                   N_iter, log, writer, csv_save_path=None, random_attack=True):
    # Note that, attack has to be done in evaluation model due to batch-norm.
    # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/6946
    model.eval()
    losses = AverageMeter()
    iter_time = AverageMeter()
    attack_time = AverageMeter()

    # attempt to use the training data to conduct BFA
    for _, (data, target) in enumerate(train_loader):
        if args.use_cuda:
            target = target.cuda()
            data = data.cuda()

        data = data.unsqueeze(-1)  # data giờ có kích thước [512, 69, 1]
        # Override the target to prevent label leaking
        _, target = model(data).data.max(1)
        break

    # evaluate the test accuracy of clean model
    val_acc_top1, val_acc_top5, val_loss, output_summary = validate(test_loader, model,
                                                    attacker.criterion, log, summary_output=True)
    tmp_df = pd.DataFrame(output_summary, columns=['top-1 output'])
    tmp_df['BFA iteration'] = 0
    tmp_df.to_csv(os.path.join(args.save_path, 'output_summary_{}_BFA_0.csv'.format(args.arch)),
                                        index=False)

    writer.add_scalar('attack/val_top1_acc', val_acc_top1, 0)
    writer.add_scalar('attack/val_top5_acc', val_acc_top5, 0)
    writer.add_scalar('attack/val_loss', val_loss, 0)

    print_log('k_top is set to {}'.format(args.k_top), log)
    print_log('Attack sample size is {}'.format(data.size()[0]), log)
    end = time.time()
    
    df = pd.DataFrame() #init a empty dataframe for logging
    last_val_acc_top1 = val_acc_top1
    
    for i_iter in range(N_iter):
        print_log('**********************************', log)
        if not random_attack:
            attack_log = attacker.progressive_bit_search(model, data, target)
        else:
            attack_log = attacker.random_flip_one_bit(model)

        # measure data loading time
        attack_time.update(time.time() - end)
        end = time.time()

        h_dist = hamming_distance(model, model_clean)

        # record the loss
        if hasattr(attacker, "loss_max"):
            losses.update(attacker.loss_max, data.size(0))

        print_log(
            'Iteration: [{:03d}/{:03d}]   '
            'Attack Time {attack_time.val:.3f} ({attack_time.avg:.3f})  '.
            format((i_iter + 1),
                   N_iter,
                   attack_time=attack_time,
                   iter_time=iter_time) + time_string(), log)
        try:
            print_log('loss before attack: {:.4f}'.format(attacker.loss.item()),
                    log)
            print_log('loss after attack: {:.4f}'.format(attacker.loss_max), log)
        except:
            pass
        
        print_log('bit flips: {:.0f}'.format(attacker.bit_counter), log)
        print_log('hamming_dist: {:.0f}'.format(h_dist), log)

        writer.add_scalar('attack/bit_flip', attacker.bit_counter, i_iter + 1)
        writer.add_scalar('attack/h_dist', h_dist, i_iter + 1)
        writer.add_scalar('attack/sample_loss', losses.avg, i_iter + 1)

        # exam the BFA on entire val dataset
        val_acc_top1, val_acc_top5, val_loss, output_summary = validate(
            test_loader, model, attacker.criterion, log, summary_output=True)
        tmp_df = pd.DataFrame(output_summary, columns=['top-1 output'])
        tmp_df['BFA iteration'] = i_iter + 1
        tmp_df.to_csv(os.path.join(args.save_path, 'output_summary_{}_BFA_{}.csv'.format(args.arch, i_iter + 1)),
                                    index=False)
    
        
        # add additional info for logging
        acc_drop = last_val_acc_top1 - val_acc_top1
        last_val_acc_top1 = val_acc_top1
        
        # print(attack_log)
        for i in range(attack_log.__len__()):
            attack_log[i].append(val_acc_top1)
            attack_log[i].append(acc_drop)
        # Giả sử attack_log là danh sách các danh sách
        df = pd.concat([df, pd.DataFrame(attack_log)], ignore_index=True)

        writer.add_scalar('attack/val_top1_acc', val_acc_top1, i_iter + 1)
        writer.add_scalar('attack/val_top5_acc', val_acc_top5, i_iter + 1)
        writer.add_scalar('attack/val_loss', val_loss, i_iter + 1)

        # measure elapsed time
        iter_time.update(time.time() - end)
        print_log(
            'iteration Time {iter_time.val:.3f} ({iter_time.avg:.3f})'.format(
                iter_time=iter_time), log)
        end = time.time()
        
        # Stop the attack if the accuracy is below the configured break_acc.
        if args.dataset == 'cifar10':
            break_acc = 11.0
        elif args.dataset == 'imagenet':
            break_acc = 0.2
        # if val_acc_top1 <= break_acc:
        #     break
        
    # attack profile
    column_list = ['module idx', 'bit-flip idx', 'module name', 'weight idx',
                  'weight before attack', 'weight after attack', 'validation accuracy',
                  'accuracy drop']
    df.columns = column_list

    df['trial seed'] = args.manualSeed
    if csv_save_path is not None:
        csv_file_name = 'attack_profile_{}.csv'.format(args.manualSeed)
        export_csv = df.to_csv(os.path.join(csv_save_path, csv_file_name), index=None)

    return

# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            target = target.cuda()
            # the copy will be asynchronous with respect to the host.
            input = input.cuda()

        input = input.view(input.size(0), 69, -1)  # Chuyển đổi kích thước

        # compute output
        output = model(input)
        loss = criterion(output, target)
        if args.clustering:
            loss += clustering_loss(model, args.lambda_coeff)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log(
                '  Epoch: [{:03d}][{:03d}/{:03d}]   '
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5) + time_string(), log)
    print_log(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
        .format(top1=top1, top5=top5, error1=100 - top1.avg), log)
    return top1.avg, losses.avg

def validate(val_loader, model, criterion, log, summary_output=False):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    output_summary = []  # init a list for output summary

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # Check if input has the correct number of channels

            if input.size(1) == 69 and input.dim() == 2:  # kiểm tra nếu thiếu chiều thứ 3
                input = input.unsqueeze(-1)

            if torch.cuda.is_available() and args.use_cuda:
                target = target.cuda(non_blocking=True)
                input = input.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # summary the output
            if summary_output:
                tmp_list = output.max(1, keepdim=True)[1].flatten().cpu().numpy()
                output_summary.append(tmp_list if isinstance(tmp_list, np.ndarray) else np.array(tmp_list))

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

        print_log(
            '  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
            .format(top1=top1, top5=top5, error1=100 - top1.avg), log)

    if summary_output:
        output_summary = np.concatenate(output_summary, axis=0).flatten()
        return top1.avg, top5.avg, losses.avg, output_summary
    else:
        return top1.avg, top5.avg, losses.avg


def print_log(print_string, log):
    try:
        print("{}".format(print_string))
        log.write('{}\n'.format(print_string))
        log.flush()  # Đảm bảo rằng nội dung được ghi ngay lập tức
    except ValueError as e:
        print(f"Error writing to log: {e}")


def save_checkpoint(state, is_best, save_path, filename, log):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:  # copy the checkpoint to the best model if it is the best_accuracy
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)
        print_log("=> Obtain best accuracy, and update the best model", log)


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    mu = args.momentum

    if args.optimizer != "YF":
        assert len(gammas) == len(
            schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif args.optimizer == "YF":
        lr = optimizer._lr
        mu = optimizer._mu

    return lr, mu

def accuracy2(outputs_label, outputs_cat, outputs_sub_cat, y_label_batch, y_cat_batch, y_sub_cat_batch, topk=(1, )):
    """Tính độ chính xác cho từng đầu ra của mô hình (label, category, sub-category)"""
    with torch.no_grad():
        maxk = max(topk)

        # Đảm bảo target có ít nhất một chiều
        if y_label_batch.dim() == 0:
            y_label_batch = y_label_batch.unsqueeze(0)
        if y_cat_batch.dim() == 0:
            y_cat_batch = y_cat_batch.unsqueeze(0)
        if y_sub_cat_batch.dim() == 0:
            y_sub_cat_batch = y_sub_cat_batch.unsqueeze(0)

        batch_size = y_label_batch.size(0)

        if outputs_label.dim() != 2 or outputs_cat.dim() != 2 or outputs_sub_cat.dim() != 2:
            raise ValueError("Outputs must have dimension [batch_size, num_classes].")

        # Tính độ chính xác top-k cho từng loại đầu ra
        res_label = []
        res_cat = []
        res_sub_cat = []

        # Top-k cho đầu ra nhãn (label)
        _, pred_label = outputs_label.topk(maxk, 1, True, True)
        pred_label = pred_label.t()
        correct_label = pred_label.eq(y_label_batch.view(1, -1).expand_as(pred_label))
        for k in topk:
            correct_k = correct_label[:k].reshape(-1).float().sum(0)
            res_label.append(correct_k.mul_(100.0 / batch_size))

        # Top-k cho đầu ra danh mục (category)
        _, pred_cat = outputs_cat.topk(maxk, 1, True, True)
        pred_cat = pred_cat.t()
        correct_cat = pred_cat.eq(y_cat_batch.view(1, -1).expand_as(pred_cat))
        for k in topk:
            correct_k = correct_cat[:k].reshape(-1).float().sum(0)
            res_cat.append(correct_k.mul_(100.0 / batch_size))

        # Top-k cho đầu ra danh mục con (sub-category)
        _, pred_sub_cat = outputs_sub_cat.topk(maxk, 1, True, True)
        pred_sub_cat = pred_sub_cat.t()
        correct_sub_cat = pred_sub_cat.eq(y_sub_cat_batch.view(1, -1).expand_as(pred_sub_cat))
        for k in topk:
            correct_k = correct_sub_cat[:k].reshape(-1).float().sum(0)
            res_sub_cat.append(correct_k.mul_(100.0 / batch_size))

        return res_label, res_cat, res_sub_cat

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        # Ensure target has at least one dimension
        if target.dim() == 0:
            target = target.unsqueeze(0)

        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_logger(base_dir, epoch, train_accuracy, test_accuracy):
    file_name = 'accuracy.txt'
    file_path = "%s/%s" % (base_dir, file_name)
    # create and format the log file if it does not exists
    if not os.path.exists(file_path):
        create_log = open(file_path, 'w')
        create_log.write('epochs train test\n')
        create_log.close()

    recorder = {}
    recorder['epoch'] = epoch
    recorder['train'] = train_accuracy
    recorder['test'] = test_accuracy
    # append the epoch index, train accuracy and test accuracy:
    with open(file_path, 'a') as accuracy_log:
        accuracy_log.write(
            '{epoch}       {train}    {test}\n'.format(**recorder))


if __name__ == '__main__':
    main()