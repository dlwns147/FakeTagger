# author:oldpan
# data:2018-4-16
# Just for study and research

from __future__ import print_function
import argparse
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import cv2
import numpy as np
import torch

import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import torch.backends.cudnn as cudnn

from models import Autoencoder, toTensor, var_to_np
from util import get_image_paths, load_images, stack_images
from training_data import get_training_data

parser = argparse.ArgumentParser(description='DeepFake-Pytorch')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda is True:
    print('===> Using GPU to train')
else:
    print('===> Using CPU to train')

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print('===> Loaing datasets')
images_A = get_image_paths("data/trump")
images_B = get_image_paths("data/cage")
images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0
print(images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2)))
exit()
# images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))
# print(f'len(A) = {len(images_A)}')
# print(f'len(B) = {len(images_B)}')
# exit()
model = Autoencoder()

print('===> Try resume from checkpoint')
if os.path.isdir('checkpoint'):
    try:
        checkpoint = torch.load('./checkpoint/autoencoder.t7')
        model.load_state_dict(checkpoint['state'])
        start_epoch = checkpoint['epoch']
        print('===> Load last checkpoint data')
    except FileNotFoundError:
        print('Can\'t found autoencoder.t7')
else:
    start_epoch = 0
    print('===> Start from scratch')

if args.cuda:
    model.cuda()
    cudnn.benchmark = True

# print all the parameters im model
# s = sum([np.prod(list(p.size())) for p in model.parameters()])
# print('Number of params: %d' % s)

if __name__ == "__main__":

    print('Start training, press \'q\' to stop')
    

    # batch_size = args.batch_size        
    batch_size = args.batch_size
    # print(f'barch_size : {batch_size}')
    with torch.no_grad() :
        
        target_A = get_training_data(images_A, batch_size)
        target_B = get_training_data(images_B, batch_size)

        target_A = toTensor(target_A)
        target_B = toTensor(target_B)

        if args.cuda:
            target_A = target_A.cuda()
            target_B = target_B.cuda()

        # target_A, target_B =  Variable(target_A.float()), Variable(target_B.float())
        target_A, target_B =  target_A.float(), target_B.float()
        
        # print(f'target_A : {target_A.shape}')
        # print(f'target_B : {target_B.shape}')

        # test_A_ = target_A[0:14]
        # test_B_ = target_B[0:14]
        # test_A = var_to_np(target_A[0:14])
        # test_B = var_to_np(target_B[0:14])
        
        
        test_A_ = target_A[0].unsqueeze(0)
        # print(f'test_A_ : {test_A_.shape}')
        test_B_ = target_B[0].unsqueeze(0)
        test_A = var_to_np(target_A[0].unsqueeze(0))
        test_B = var_to_np(target_B[0].unsqueeze(0))
        
        # print(f"res : {var_to_np(model(test_A_, 'B')).shape}")
        cv2.imwrite("src.png", var_to_np(test_A_.permute(0, 2, 3, 1)[0]) * 255)
        cv2.imwrite("res.png", var_to_np(model(test_A_, 'B').permute(0, 2, 3, 1)[0]) * 255)
        
        # figure_A = np.stack([
        #     test_A,
        #     var_to_np(model(test_A_, 'A')),
        #     var_to_np(model(test_A_, 'B')),
        # ], axis=1)
        # figure_B = np.stack([
        #     test_B,
        #     var_to_np(model(test_B_, 'B')),
        #     var_to_np(model(test_B_, 'A')),
        # ], axis=1)

        # figure = np.concatenate([figure_A, figure_B], axis=0)
        # figure = figure.transpose((0, 1, 3, 4, 2))
        # figure = figure.reshape((4, 7) + figure.shape[1:])
        # figure = stack_images(figure)

        # figure = np.clip(figure * 255, 0, 255).astype('uint8')

        # cv2.imshow("", figure)
        # key = cv2.waitKey(1)
