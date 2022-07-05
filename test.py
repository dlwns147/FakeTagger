from re import I
import torch
import os
import numpy as np
import random
import argparse
from tqdm import tqdm
import cv2
import math

from torchvision import transforms
from dataset import CustomDataset, split_dataset
from faketagger import FaceTagger
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch import optim


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
set_seed(0)

parser = argparse.ArgumentParser(description='FakeTagger')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--message_size', default=15, type=int, help='message size')
parser.add_argument('--name', default='w_df_lambda_1_alpha_05', type=str, help='name to save')
parser.add_argument('--gpus', default='1', type=str, help='id of gpus to use')
# parser.add_argument('--num_gpus', default=1, type=int, help='numbers of gpus to use')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpus  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRUMP_PATH = '/media/data1/sangjun/faketagger/resized_data/trump'
CAGE_PATH = '/media/data1/sangjun/faketagger/resized_data/cage'
SAVE_PATH = '/media/data1/sangjun/faketagger/result/test'
LOAD_PATH = os.path.join('/media/data1/sangjun/faketagger/result/save', args.name, 'best.pt') 

transform_test = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

trump_train_dataset, trump_test_dataset, trump_test_dataset = split_dataset(TRUMP_PATH, train_transform = None, test_transform = transform_test)
trump_test_loader = DataLoader(trump_test_dataset, batch_size = args.batch_size, shuffle = False)

cage_train_dataset, cage_test_dataset, cage_test_dataset = split_dataset(CAGE_PATH, train_transform = None, test_transform = transform_test)
cage_test_loader = DataLoader(cage_test_dataset, batch_size = args.batch_size, shuffle = False)

model = FaceTagger(args.message_size, device = device)
model.encoder.load_state_dict(torch.load(LOAD_PATH)['encoder'])
model.decoder.load_state_dict(torch.load(LOAD_PATH)['decoder'])

model.encoder.eval()
model.decoder.eval()

test_message_correct = 0
test_df_message_correct = 0
test_size = 0

trump_psnr_sum = 0
cage_psnr_sum = 0
trump_ssim_sum = 0
cage_ssim_sum = 0


with torch.no_grad() :
    for (trump_test_x, cage_test_x) in zip(trump_test_loader, cage_test_loader) :
        trump_test_x = trump_test_x.to(device)
        cage_test_x = cage_test_x.to(device)
    
        for k in range(len(trump_test_x)) :
            cv2.imwrite(os.path.join(SAVE_PATH, 'trump/trump_real_' + str(k) + '.png'), (trump_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
            
        for k in range(len(cage_test_x)) :
            cv2.imwrite(os.path.join(SAVE_PATH, 'cage/cage_real_' + str(k) + '.png'), (cage_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
            
        trump_message = torch.randint(0, 2, (trump_test_x.shape[0], args.message_size), dtype = torch.float).to(device).detach()
        cage_message = torch.randint(0, 2, (cage_test_x.shape[0], args.message_size), dtype = torch.float).to(device).detach()

        encoded_trump = model.encode(trump_test_x, trump_message)
        encoded_cage = model.encode(cage_test_x, cage_message)
        
        
        for k in range(len(trump_test_x)) :
            cv2.imwrite(os.path.join(SAVE_PATH, 'trump/trump_encoded_' + str(k) + '.png'), (encoded_trump[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
            
        for k in range(len(cage_test_x)) :
            cv2.imwrite(os.path.join(SAVE_PATH, 'cage/cage_encoded_' + str(k) + '.png'), (encoded_cage[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
            

        for k in range(len(trump_test_x)) :
            trump_psnr_sum += calculate_psnr((trump_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy(), (encoded_trump[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
            trump_ssim_sum += calculate_ssim((trump_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy(), (encoded_trump[k] * 255).permute(1, 2, 0).detach().cpu().numpy())

        for k in range(len(cage_test_x)) :
            cage_psnr_sum += calculate_psnr((cage_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy(), (encoded_cage[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
            cage_ssim_sum += calculate_ssim((cage_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy(), (encoded_cage[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
        encoded_trump_df = model.deepfake(encoded_trump, 'B') 
        encoded_cage_df = model.deepfake(encoded_cage, 'A')
        
        for k in range(len(trump_test_x)) :
            cv2.imwrite(os.path.join(SAVE_PATH, 'trump/trump_encoded_fake_' + str(k) + '.png'), (encoded_trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
            
        for k in range(len(cage_test_x)) :
            cv2.imwrite(os.path.join(SAVE_PATH, 'cage/cage_encoded_fake_' + str(k) + '.png'), (encoded_cage_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
            
        encoded_trump_df_message = model.decode(encoded_trump_df)
        encoded_cage_df_message = model.decode(encoded_cage_df)
        
        encoded_trump_message = model.decode(encoded_trump)
        encoded_cage_message = model.decode(encoded_cage)
        
        test_df_message_correct += ((encoded_trump_df_message > 0.5) == trump_message).sum().item() + ((encoded_cage_df_message > 0.5) == cage_message).sum().item() 
        test_message_correct += ((encoded_trump_message > 0.5) == trump_message).sum().item() + ((encoded_cage_message > 0.5) == cage_message).sum().item() 
            
        test_size += trump_test_x.shape[0] + cage_test_x.shape[0]
        
    test_message_acc = test_message_correct / (test_size * args.message_size)
    test_df_message_acc = test_df_message_correct / (test_size * args.message_size)
    trump_psnr_avg = trump_psnr_sum / len(trump_test_loader.dataset)
    cage_psnr_avg = cage_psnr_sum / len(cage_test_loader.dataset)
    trump_ssim_avg = trump_ssim_sum / len(trump_test_loader.dataset)
    cage_ssim_avg = cage_ssim_sum / len(cage_test_loader.dataset)
        
    print(f"Trump PSNR : {trump_psnr_avg}, Cage PSNR : {cage_psnr_avg}")
    print(f"Trump SSIM : {trump_ssim_avg}, Cage SSIM : {cage_ssim_avg}")
    print(f"Test encoded message accuracy : {test_message_acc}, Test DF message accuracy : {test_df_message_acc}")
    
    