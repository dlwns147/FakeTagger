import torch
import torch.nn as nn
from encoder import ResNetUNet
from decoder import Decoder
from faceswap_pytorch.models import Autoencoder

class FaceTagger(nn.Module) :
    def __init__(self, message_size, device) :
        super().__init__()
        self.encoder = ResNetUNet(message_size)
        self.df_model = Autoencoder()
        self.decoder = Decoder(message_size)
        
        checkpoint = torch.load('./faceswap_pytorch/checkpoint/autoencoder.t7')
        self.df_model.load_state_dict(checkpoint['state'])

        if device :
            self.encoder = self.encoder.to(device)
            self.df_model = self.df_model.to(device)
            self.decoder = self.decoder.to(device)
        # for param in self.df_model.features.parameter() :
        #     param.requires_grad = False
        # self.df_model.eval()
        
        nets = self.df_model
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = False
            
    def encode(self, x, message) :
        return self.encoder(x, message)
        
    def deepfake(self, x, type) : 
        return self.df_model(x, type)
        
    def decode(self, x) :
        # print(x.shape)
        return self.decoder(x)