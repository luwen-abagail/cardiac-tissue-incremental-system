import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Sequential(nn.Conv2d(in_channels//2,2*in_channels,kernel_size=1,bias=False),
                                nn.PixelShuffle(2))
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, layer_num):
        super(OutConv, self).__init__()
        temp = []
        for i in range(len(layer_num)):
            if i == 0:
                temp.append(nn.Conv2d(in_channels, layer_num[i], kernel_size=1))
            else:
                temp.append(nn.Conv2d(in_channels, layer_num[i]-1, kernel_size=1))
        self.convs = nn.ModuleList(temp)

    def forward(self, x):
        fds = []
        for i in range(len(self.convs)):
            fds.append(self.convs[i](x))

        for i in range(len(self.convs)):
            if i == 0:
                P = fds[i]
            else:
                P = torch.cat([P,fds[i]],dim=1)
        return P.softmax(dim=1)
class FeatureExtractor(nn.Module):  
    def __init__(self, in_channels, out_channels):  
        super(FeatureExtractor, self).__init__()  
        self.conv = nn.Sequential(  
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True)  
        )  
  
    def forward(self, x):  
        return self.conv(x)
class UNet(nn.Module):
    def __init__(self, n_channels, layer_num):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.layer_num = layer_num

        self.inc = DoubleConv(n_channels, 64)
        
        self.feature_extractor1 = FeatureExtractor(64, 1) 
        self.down1 = Down(64, 128)
        
        self.feature_extractor2 = FeatureExtractor(128, 1) 
        self.down2 = Down(128, 256)
        
        self.feature_extractor3 = FeatureExtractor(256, 1) 
        self.down3 = Down(256, 256)
        
        self.feature_extractor4 = FeatureExtractor(256, 1) 
        
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        
        self.out = nn.Conv2d(64, layer_num, 1)

    def forward(self, x):
        features =[]
        x1 = self.inc(x)
        
        fea1 = self.feature_extractor1(x1)
        features.append(fea1)
        x2 = self.down1(x1)
        fea2 = self.feature_extractor2(x2)
        features.append(fea2)
        x3 = self.down2(x2)
        fea3 = self.feature_extractor3(x3)
        features.append(fea3)
        x4 = self.down3(x3)
        fea4 = self.feature_extractor4(x4)
        features.append(fea4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out(x)
        return features,logits

