#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 20:06:47 2022

@author: dse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import torchvision
import torchvision.transforms as transform
import numpy as np

class MSFCN(nn.Module):  # sub-classing nn.module class
    def __init__(self):
        super(MSFCN,self).__init__()
        
        self.conv1=nn.Conv2d(4,64,5,padding = 2,padding_mode= 'replicate')  # first convolution layer with 64 filters and kernel size 5x5 assuming input image is an RGB image(3 channels)
        self.conv2=nn.Conv2d(64,32,3,padding = 1,padding_mode = 'replicate')  #convolution layer with 32 filters  kernel size 3x3 i
        self.pool=nn.MaxPool2d(2,2)
        #self.batchnorm = nn.BatchNorm2d(32)
        self.conv3=nn.Conv2d(32,32,3,padding = 1,padding_mode = 'replicate')
        self.conv4=nn.Conv2d(32,32,1)
        self.upsample=nn.Upsample(scale_factor=2)
        
#self.conv6=nn.conv2D(,1,1)    # 1x1 convolution layer
        self.conv5=nn.Conv2d(32,16,1)
        self.conv6 = nn.Conv2d(48,1,1)
        self.fl = nn.Flatten()
        self.ll = nn.Linear(1120,1)
    def forward(self,inp):
        inp=F.leaky_relu(self.conv1(inp))
        inp1=F.leaky_relu(self.conv2(inp))
        #inp1 = self.batchnorm(inp1)
        inp2=self.pool(inp1)
        inp2=F.leaky_relu(self.conv3(inp2))
        inp2=F.leaky_relu(self.conv4(inp2))
        inp2=self.upsample(inp2)

        
        
        inp3=F.leaky_relu(self.conv3(inp1))
        inp3=F.leaky_relu(self.conv5(inp3))
        #print(inp2.size())
        #print(inp3.size())
        
        
        #CONCATENATE inp2 and inp3

        out=torch.cat((inp2,inp3),dim = 1)
        #print(out.shape[1])
        output1 = self.conv6(out)
        #out = F.tanh(output1)
        
        return output1
    
