import torch.nn as nn
import torch

import torch.nn.functional as F
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class FeatureExtractor(nn.Module):  
    def __init__(self, in_channels, out_channels):  
        super(FeatureExtractor, self).__init__()  
        self.conv = nn.Sequential(  
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True)  
        )  
  
    def forward(self, x):  
        return self.conv(x)
class Inception(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Inception, self).__init__()
        hide_ch = out_ch // 2
        self.inception = nn.Sequential(
            nn.Conv2d(in_ch, hide_ch, 1),
            nn.BatchNorm2d(hide_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hide_ch, hide_ch, 3, padding=1, groups=hide_ch),
            nn.BatchNorm2d(hide_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hide_ch, out_ch, 1)
        )

    def forward(self, x):
        return self.inception(x)


class DoubleConv(nn.Module):
    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=False):  
  
        super(DoubleConv, self).__init__()  
  
        self.conv1 = nn.Sequential(  
                        nn.Conv2d(inp_feat, out_feat, kernel_size=kernel,  
                                    stride=stride, padding=padding, bias=True),  
                        nn.BatchNorm2d(out_feat),  
                        nn.ReLU())  
  
        self.conv2 = nn.Sequential(  
                        nn.Conv2d(out_feat, out_feat, kernel_size=kernel,  
                                    stride=stride, padding=padding, bias=True),  
                        nn.BatchNorm2d(out_feat),  
                        nn.ReLU())  
  
        self.residual = residual  
  
        if self.residual:  
            self.residual_upsampler = nn.Conv2d(inp_feat, out_feat, kernel_size=1, bias=False)  
    def forward(self, x):

        res = x

        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNet, self).__init__()
        # down
        self.conv1 = DoubleConv(in_ch, 64)
        self.feature_extractor1 = FeatureExtractor(64, 3)  
        self.pool1 = nn.Conv2d(64, 64, 2, 2, groups=64)
        self.conv2 = DoubleConv(64, 128)
        self.feature_extractor2 = FeatureExtractor(128, 3) 
        self.pool2 = nn.Conv2d(128, 128, 2, 2, groups=128)
        self.bottom = DoubleConv(128, 256)
        self.feature_extractor3 = FeatureExtractor(256, 3)  
        # attention
        self.se1 = SEBlock(64)
        self.se2 = SEBlock(128)
        self.se3 = SEBlock(256)

        # up
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv3 = DoubleConv(128 * 2, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv4 = DoubleConv(64 * 2, 64)
        self.out = nn.Conv2d(64, out_ch, 1)
        
        self.classifier = nn.Conv2d(out_ch, out_ch, kernel_size=1)  
        self.sigmoid = nn.Sigmoid()  # 激活函数
        

    def forward(self, x):
        # down
        features=[]
        conv1 = self.conv1(x)
        conv1 = self.se1(conv1)
        fea1 = self.feature_extractor1(conv1)
        features.append(fea1)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1)
        conv2 = self.se2(conv2)
        fea2 = self.feature_extractor2(conv2)
        features.append(fea2)
        pool2 = self.pool2(conv2)
        
        bottom = self.bottom(pool2)
        bottom = self.se3(bottom)
        features.append(self.feature_extractor3(bottom))
        # up
        features .append(self.feature_extractor3(bottom))
        up3 = self.up3(bottom)
        merge3 = torch.cat([up3, conv2], dim=1)
        conv3 = self.conv3(merge3)
        up4 = self.up4(conv3)
        merge4 = torch.cat([up4, conv1], dim=1)
        conv4 = self.conv4(merge4)
        out = self.out(conv4) 
        # 返回特征和输出  
        return features, out
    