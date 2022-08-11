import torchvision
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from skimage import exposure
from collections import OrderedDict
from torch.nn import BatchNorm2d as bn

class AGCnet(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(AGCnet, self).__init__()
        out_channels = in_channels
        self.global_avgpool = nn.AdaptiveAvgPool2d((2,2))   #平均池化为2*2的块,输出channels*2*2
        self.conv1 = nn.Conv2d(in_channels,out_channels,3,stride=1,padding=1,bias=False)#3*3卷积块，输出channels*2*2
        self.conv2 = nn.Conv2d(in_channels,out_channels,1,stride=2,padding=1,bias=False)#1*1卷积块，输出channels*2*2
        #self.gamma_layers = nn.Sequential(self.global_avgpool,self.conv1,self.conv2)

    def forward(self, x):
        """
        print(x.size())
        p = torch.mean(x)
        p = p/3
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x0 = self.global_avgpool(x)
        x0 = self.conv1(x0)
        x1 = self.conv2(x0)
        
        #规范输出到0-2之间
        x1min = torch.min(x1).cpu().detach().numpy()
        x1max = torch.max(x1).cpu().detach().numpy()
        x1 = (torch.from_numpy((x1.cpu().detach().numpy() - x1min)/(x1max - x1min) * 2)).to(device)
        #x1 = torch.sigmoid(x1)
        #x1 = torch.add(x1,0.5)  #规范输出范围0.5-1.5

        x2 = torch.min(x).cpu().detach().numpy()
        x3 = torch.max(x).cpu().detach().numpy()
        x4 = (torch.from_numpy((x.cpu().detach().numpy()-x2)/(x3-x2))).to(device)    #归一化

        #tensor分块
        hd1 = torch.split(x4,x4.size()[2] // 2,2)[0]
        d11 = torch.split(hd1,x4.size()[3] // 2,3)[0].cpu().detach().numpy()
        d12 = torch.split(hd1,x4.size()[3] // 2,3)[1].cpu().detach().numpy()
        hd2 = torch.split(x4,x4.size()[2] // 2,2)[1]
        d21 = torch.split(hd2,x4.size()[3] // 2,3)[0].cpu().detach().numpy()
        d22 = torch.split(hd2,x4.size()[3] // 2,3)[1].cpu().detach().numpy()

        h1 = torch.split(x1,1,2)[0]
        ga11 = torch.split(h1,1,3)[0].cpu().detach().numpy()
        ga12 = torch.split(h1,1,3)[1].cpu().detach().numpy()
        h2 = torch.split(x1,1,2)[1]
        ga21 = torch.split(h2,1,3)[0].cpu().detach().numpy()
        ga22 = torch.split(h2,1,3)[1].cpu().detach().numpy()
       
        #分块进行gamma校正和log校正
        
        g11 = torch.zeros([d11.shape[0], d11.shape[1], d11.shape[2], d11.shape[3]]).cpu().detach().numpy()
        g12 = torch.zeros([d12.shape[0], d12.shape[1], d12.shape[2], d12.shape[3]]).cpu().detach().numpy()
        g21 = torch.zeros([d21.shape[0], d21.shape[1], d21.shape[2], d21.shape[3]]).cpu().detach().numpy()
        g22 = torch.zeros([d22.shape[0], d22.shape[1], d22.shape[2], d22.shape[3]]).cpu().detach().numpy()
        
        for i in range(ga11.shape[0]):
            for j in range(ga11.shape[1]):         
                if  ga11[i][j][0][0]<1:
                    g11[i][j][:][:] = exposure.adjust_log(d11[i][j][:][:], ga11[i][j][0][0])
                else: 
                    g11[i][j][:][:]= exposure.adjust_gamma(d11[i][j][:][:], ga11[i][j][0][0])
                if  ga12[i][j][0][0]<1:
                    g12[i][j][:][:]= exposure.adjust_log(d12[i][j][:][:], ga12[i][j][0][0])
                else:
                    g12[i][j][:][:] = exposure.adjust_gamma(d12[i][j][:][:], ga12[i][j][0][0])
                if  ga21[i][j][0][0]<1:
                    g21[i][j][:][:] = exposure.adjust_log(d21[i][j][:][:], ga21[i][j][0][0])
                else:
                    g21[i][j][:][:] = exposure.adjust_gamma(d21[i][j][:][:], ga21[i][j][0][0])
                
                if  ga22[i][j][0][0]<1:
                    g22[i][j][:][:] = exposure.adjust_log(d22[i][j][:][:], ga22[i][j][0][0])
                else:
                    g22[i][j][:][:] = exposure.adjust_gamma(d22[i][j][:][:], ga22[i][j][0][0])


        g11 = torch.from_numpy(g11)
        g12 = torch.from_numpy(g12)
        g21 = torch.from_numpy(g21)
        g22 = torch.from_numpy(g22)

        #print(hd1.shape)
        #print(d11.shape)
        #print(g11.shape)


        #tensor拼接
        g1 = torch.cat((g11,g12),3)
        g2 = torch.cat((g21,g22),3)
        #device =torch.device("cuda")
        x44 = torch.cat((g1,g2),2).to(device)#.to(device='cuda:2')

        #x4 = torch.from_numpy(exposure.adjust_gamma(x4.cpu().detach().numpy(), 1.5))

        x5 = (((x4 *(x3-x2)).cpu()).detach().cpu().numpy() + x2)    #反归一化
        x5 = torch.tensor(x5).to(device)#.to(device='cuda:2')
        y = x+x5  #注意力机制
        return y


