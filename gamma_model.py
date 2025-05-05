import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#import pytorch_colors as colors
import numpy as np

class enhance_net_nopool(nn.Module):

    def __init__(self):
        super(enhance_net_nopool, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        number_f = 32
        self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv5 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv6 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv7 = nn.Conv2d(number_f,6,3,1,1,bias=True)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):

        x1 = self.relu(self.e_conv1(x))
        # p1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
        # p2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
        # p3 = self.maxpool(x3)
        x4 = self.relu(self.e_conv4(x3))
        
        x5 = self.relu(self.e_conv5(x4))
        x6 = self.relu(self.e_conv6(x5))
        x7 = self.sigmoid(self.e_conv7(x6))
        x_gamma,x_beta = torch.split(x7, 3, dim=1)

        '''
        [0,0.5] -> [0.3,1]
        [0.5,1] -> [1,2.3]
        '''
        gamma = 0.4 + x_gamma 
        beta = x_beta
        # x_r = 0.5+ 2*F.sigmoid(self.e_conv7(x6))
        # r1,r2,r3,r4 = torch.split(x_r, 3, dim=1)
        # beta=0.5

        x = torch.clamp(x, min=1e-8)
        x = torch.pow(x,gamma) + beta * (x - torch.pow(x,gamma))
        # x = torch.pow(x,r2) + beta * (x - torch.pow(x,r2))
        # x = torch.pow(x,r3) + beta * (x - torch.pow(x,r3))
        # x = torch.pow(x,r4) + beta * (x - torch.pow(x,r4))
        # r = torch.cat([r1,r2,r3,r4],1)

        return x,gamma,beta
    
    # def forward(self, x):

    #     x1 = self.relu(self.e_conv1(x))
    #     # p1 = self.maxpool(x1)
    #     x2 = self.relu(self.e_conv2(x1))
    #     # p2 = self.maxpool(x2)
    #     x3 = self.relu(self.e_conv3(x2))
    #     # p3 = self.maxpool(x3)
    #     x4 = self.relu(self.e_conv4(x3))
        
    #     x5 = self.relu(self.e_conv5(x4))
    #     x6 = self.relu(self.e_conv6(x5))
    #     x_r = 0.5+ 2*F.sigmoid(self.e_conv7(x6))

    #     # x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
    #     # # x5 = self.upsample(x5)
    #     # x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))

    #     # x_r = 0.5+ 2*F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
    #     r1,r2,r3,r4 = torch.split(x_r, 3, dim=1)
    #     beta=0.5

    #     x = torch.clamp(x, min=1e-8)
    #     x = torch.pow(x,r1) + beta * (x - torch.pow(x,r1))
    #     x = torch.pow(x,r2) + beta * (x - torch.pow(x,r2))
    #     x = torch.pow(x,r3) + beta * (x - torch.pow(x,r3))
    #     x = torch.pow(x,r4) + beta * (x - torch.pow(x,r4))
    #     r = torch.cat([r1,r2,r3,r4],1)

    #     return x,r
    
    # def forward(self, x):

    # 	x1 = self.relu(self.e_conv1(x))
    # 	# p1 = self.maxpool(x1)
    # 	x2 = self.relu(self.e_conv2(x1))
    # 	# p2 = self.maxpool(x2)
    # 	x3 = self.relu(self.e_conv3(x2))
    # 	# p3 = self.maxpool(x3)
    # 	x4 = self.relu(self.e_conv4(x3))

    # 	x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
    # 	# x5 = self.upsample(x5)
    # 	x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))

    # 	x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
    # 	r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)


    # 	x = x + r1*(torch.pow(x,2)-x)
    # 	x = x + r2*(torch.pow(x,2)-x)
    # 	x = x + r3*(torch.pow(x,2)-x)
    # 	enhance_image_1 = x + r4*(torch.pow(x,2)-x)		
    # 	x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)		
    # 	x = x + r6*(torch.pow(x,2)-x)	
    # 	x = x + r7*(torch.pow(x,2)-x)
    # 	enhance_image = x + r8*(torch.pow(x,2)-x)
    # 	r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)
    # 	return enhance_image_1,enhance_image,r