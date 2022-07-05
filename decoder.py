import torch
import torch.nn as nn

class Conv2dBatchNorm2dReLU(nn.Module) :
    def __init__(self, dim_in, dim_out) :
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU()
        )
    def forward(self, x) :
        return self.layer(x)

class Decoder(nn.Module) :
    def __init__(self, message_size) :
        super().__init__()
        self.layer1 = Conv2dBatchNorm2dReLU(3, 32)
        self.layer2 = Conv2dBatchNorm2dReLU(32, 64)
        self.layer3 = Conv2dBatchNorm2dReLU(64, 128)
        self.layer4 = Conv2dBatchNorm2dReLU(128, 256)
        self.layer5 = Conv2dBatchNorm2dReLU(256, 512)
        self.layer6 = Conv2dBatchNorm2dReLU(512, 1024)
        self.layer7 = Conv2dBatchNorm2dReLU(1024, 1024)
        self.max_pool_layer = nn.MaxPool2d(2)
        self.linear = nn.Linear(1024 * 8 * 8, message_size)
        self.sigmoid = nn.Sigmoid()
        
        
        for layer in self.modules() :
            if isinstance(layer, (nn.Conv2d)) :
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(layer, (nn.Linear)) :
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        
    def forward(self, x) :
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.max_pool_layer(x) # (b, 64, 32, 32)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.max_pool_layer(x) # (b, 256, 16, 16)
        x = self.layer5(x)
        x = self.layer6(x) # (b, 1024, 32, 32)
        x = self.max_pool_layer(x) # (b, 1024, 8, 8)
        x = self.layer7(x) # (b, 1024, 8, 8)
        x = x.flatten(start_dim = 1)
        x = self.linear(x) # (b, message_size)
        return self.sigmoid(x)
        