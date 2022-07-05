from re import I
import torch
import os
import numpy as np
import random
import argparse
from tqdm import tqdm
import cv2

from torchvision import transforms
from dataset import CustomDataset, split_dataset
from faketagger import FaceTagger
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch import optim


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
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epoch', default=1000, type=int, help='epochs')
parser.add_argument('--start_decode', default=30, type=int, help='epoch to start training decoder')
parser.add_argument('--clip', default=15, type=int, help='clip')
parser.add_argument('--message_size', default=15, type=int, help='message size')
parser.add_argument('--lambda_val', default=1, type=float, help='weight of message loss')
parser.add_argument('--alpha_val', default=0.5, type=float, help='weight of image loss')
parser.add_argument('--T_max', default=50, type=int, help='cosine annealing LR scheduler t_max')
parser.add_argument('--name', default='w_df_lambda_1_alpha_05', type=str, help='name to save')
parser.add_argument('--gpus', default='1', type=str, help='id of gpus to use')
# parser.add_argument('--num_gpus', default=1, type=int, help='numbers of gpus to use')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpus  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRUMP_PATH = '/media/data1/sangjun/faketagger/resized_data/trump'
CAGE_PATH = '/media/data1/sangjun/faketagger/resized_data/cage'
SAVE_PATH = '/media/data1/sangjun/faketagger/result/save'


transform_train = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Resize((68, 68)),
                # transforms.RandomCrop((64, 64)),
                transforms.RandomHorizontalFlip(0.5),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

transform_test = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

trump_train_dataset, trump_val_dataset, trump_test_dataset = split_dataset(TRUMP_PATH, train_transform = transform_train, test_transform = transform_test)
trump_train_loader = DataLoader(trump_train_dataset, batch_size = args.batch_size, shuffle = True)
trump_val_loader = DataLoader(trump_val_dataset, batch_size = args.batch_size, shuffle = False)
trump_test_loader = DataLoader(trump_test_dataset, batch_size = args.batch_size, shuffle = False)

cage_train_dataset, cage_val_dataset, cage_test_dataset = split_dataset(CAGE_PATH, train_transform = transform_train, test_transform = transform_test)
cage_train_loader = DataLoader(cage_train_dataset, batch_size = args.batch_size, shuffle = True)
cage_val_loader = DataLoader(cage_val_dataset, batch_size = args.batch_size, shuffle = False)
cage_test_loader = DataLoader(cage_test_dataset, batch_size = args.batch_size, shuffle = False)

model = FaceTagger(args.message_size, device = device)

opt = optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr = args.lr, weight_decay = 1e-5)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max = args.T_max, eta_min = 1e-8)

criterion = torch.nn.MSELoss(reduction = 'sum')
message_criterion = torch.nn.BCELoss(reduction = 'sum')

min_val_loss = float('inf')
for i in tqdm(range(args.epoch)):
    
    for param_group in opt.param_groups:
        lr = param_group["lr"]
    
    print(f"Epoch {i}, lr : {lr}")    
    
    train_epoch_image_loss = 0
    train_epoch_message_loss = 0
    train_message_correct = 0
    
    model.encoder.train()
    model.decoder.train()
    
    train_size = 0
    for trump_train_x, cage_train_x in zip(trump_train_loader, cage_train_loader) :
        
        trump_train_x = trump_train_x.to(device)
        cage_train_x = cage_train_x.to(device)
        
        opt.zero_grad()
        trump_message = torch.randint(0, 2, (trump_train_x.shape[0], args.message_size), dtype = torch.float).to(device).detach()
        cage_message = torch.randint(0, 2, (cage_train_x.shape[0], args.message_size), dtype = torch.float).to(device).detach()
        # print(f'trump_message : {trump_message.shape}, cage_message : {cage_message.shape}')

        encoded_trump = model.encode(trump_train_x, trump_message)
        encoded_cage = model.encode(cage_train_x, cage_message)
        # print(f'encoded_trump : {encoded_trump.shape}, encoded_cage : {encoded_cage.shape}')
        
        image_loss = (criterion(encoded_trump, trump_train_x) + criterion(encoded_cage, cage_train_x))
        image_loss *= args.alpha_val
        loss = image_loss
        
        train_epoch_image_loss += image_loss.item()
        
        if i >= args.start_decode :
            encoded_trump_message = model.decode(encoded_trump)
            encoded_cage_message = model.decode(encoded_cage)
        
            encoded_trump_df = model.deepfake(encoded_trump, 'B') 
            encoded_cage_df = model.deepfake(encoded_cage, 'A')
            
            encoded_trump_df_message = model.decode(encoded_trump_df)
            encoded_cage_df_message = model.decode(encoded_cage_df)
            
            # print(f'trump_decoded : {trump_decoded.shape}, cage_decoded : {cage_decoded.shape}')
            message_loss = message_criterion(encoded_trump_df_message, trump_message) + message_criterion(encoded_cage_df_message, cage_message) + \
                message_criterion(encoded_trump_message, trump_message) + message_criterion(encoded_cage_message, cage_message)
            # message_loss = message_criterion(encoded_trump_message, trump_message) + message_criterion(encoded_cage_message, cage_message)
            message_loss *= args.lambda_val
            loss += message_loss
            train_epoch_message_loss += message_loss.item()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        loss.backward()
        opt.step()
        
        train_size += trump_train_x.shape[0] + cage_train_x.shape[0]
        # train_message_correct += ((trump_decoded > 0.5) == trump_message).sum().item() + ((cage_decoded == cage_message).sum().item()
        
    
    # train_size = len(trump_train_loader.dataset) + len(cage_train_loader.dataset)
    train_epoch_image_loss /= train_size
    train_epoch_message_loss /= train_size
    # train_message_acc = train_message_correct / (train_size * args.message_size)
    print(f"Train image loss : {train_epoch_image_loss}, Train message loss : {train_epoch_message_loss}")
    
    lr_scheduler.step()
    
    model.encoder.eval()
    model.decoder.eval()
    
    val_image_loss = 0
    val_message_loss = 0
    val_message_correct = 0
    val_df_message_correct = 0
    val_size = 0
    
    with torch.no_grad() :
        for (trump_val_x, cage_val_x) in zip(trump_val_loader, cage_val_loader) :
            
            trump_val_x = trump_val_x.to(device)
            cage_val_x = cage_val_x.to(device)
            
            for k in range(0, 10, 1) :
                cv2.imwrite('/media/data1/sangjun/faketagger/result/trump/' + str(k) + '/trump_real_' + str(i) + '.png', (trump_val_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite('/media/data1/sangjun/faketagger/result/cage/' + str(k) + '/cage_real_' + str(i) + '.png', (cage_val_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                
            trump_message = torch.randint(0, 2, (trump_val_x.shape[0], args.message_size), dtype = torch.float).to(device).detach()
            cage_message = torch.randint(0, 2, (cage_val_x.shape[0], args.message_size), dtype = torch.float).to(device).detach()

            encoded_trump = model.encode(trump_val_x, trump_message)
            encoded_cage = model.encode(cage_val_x, cage_message)
            
            for k in range(0, 10, 1) :
                cv2.imwrite('/media/data1/sangjun/faketagger/result/trump/' + str(k) + '/trump_encoded_' + str(i) + '.png', (encoded_trump[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite('/media/data1/sangjun/faketagger/result/cage/' + str(k) + '/cage_encoded_' + str(i) + '.png', (encoded_cage[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
            
            
            image_loss = criterion(encoded_trump, trump_val_x) + criterion(encoded_cage, cage_val_x)
            
            image_loss *= args.alpha_val
            
            if i >= args.start_decode :
                encoded_trump_df = model.deepfake(encoded_trump, 'B') 
                encoded_cage_df = model.deepfake(encoded_cage, 'A')
                
                for k in range(0, 10, 1) :
                    cv2.imwrite('/media/data1/sangjun/faketagger/result/trump/' + str(k) + '/trump_encoded_fake_' + str(i) + '.png', (encoded_trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                    cv2.imwrite('/media/data1/sangjun/faketagger/result/cage/' + str(k) + '/cage_encoded_fake_' + str(i) + '.png', (encoded_cage_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                    
                encoded_trump_df_message = model.decode(encoded_trump_df)
                encoded_cage_df_message = model.decode(encoded_cage_df)
                
                encoded_trump_message = model.decode(encoded_trump)
                encoded_cage_message = model.decode(encoded_cage)
                message_loss = message_criterion(encoded_trump_df_message, trump_message) + message_criterion(encoded_cage_df_message, cage_message) + \
                    message_criterion(encoded_trump_message, trump_message) + message_criterion(encoded_cage_message, cage_message)
                val_df_message_correct += ((encoded_trump_df_message > 0.5) == trump_message).sum().item() + ((encoded_cage_df_message > 0.5) == cage_message).sum().item() 
                val_message_correct += ((encoded_trump_message > 0.5) == trump_message).sum().item() + ((encoded_cage_message > 0.5) == cage_message).sum().item() 
                message_loss *= args.lambda_val
                val_message_loss += message_loss
                
            
            val_image_loss += image_loss
            val_size += trump_val_x.shape[0] + cage_val_x.shape[0]
            
        val_image_loss /= val_size
        val_message_loss /= val_size
        val_message_acc = val_message_correct / (val_size * args.message_size)
        val_loss = val_image_loss + val_message_loss
        val_df_message_acc = val_df_message_correct / (val_size * args.message_size)
        
        if min_val_loss > val_loss and i > args.start_decode :
            print(f'model saved at epoch {i}')
            path = os.path.join(SAVE_PATH, args.name)
            min_val_loss = val_loss
            if not os.path.isdir(path) :
                os.makedirs(path)
            torch.save({
                "encoder" : model.encoder.state_dict(),
                "decoder" : model.decoder.state_dict()
            }, os.path.join(path, "best.pt"))
            
        print(f"Val image loss : {val_image_loss}, Val message loss : {val_message_loss}, Val encoded message accuracy : {val_message_acc}, Val DF message accuracy : {val_df_message_acc}")
        