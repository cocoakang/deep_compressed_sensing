import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
import numpy as np
from scipy.stats import ortho_group
import math

class SENSING_NET(nn.Module):
    def __init__(self,args):
        super(SENSING_NET,self).__init__()

        self.pattern_num = args["pattern_num"]
        self.src_num = args["src_num"]
        self.det_num = args["det_num"]

        tmp_kernel = torch.empty(self.pattern_num,self.src_num)
        nn.init.xavier_normal_(tmp_kernel)
     
        self.kernel = torch.nn.Parameter(
            data = tmp_kernel,
            requires_grad=True
        )#(m_len,lumi_len,3)
    
    def get_lighting_pattern(self,device,as_img=False):
        tmp_kernel = self.kernel.to(device)
        # tmp_kernel = torch.sigmoid(tmp_kernel)*2.0-1.0
        tmp_kernel = torch.nn.functional.normalize(tmp_kernel,dim=0)
        if as_img:
            tmp_kernel = tmp_kernel.reshape((self.pattern_num,28,28))
            return tmp_kernel
        return tmp_kernel

    def forward(self,sinogram_3d_img):
        '''
        sinogram_3d = (batch_size,src_num(layer_num,grp_num,grp_id),det_num(layer_num,grp_num,grp_id),channel_num)
        '''
        batch_size = sinogram_3d_img.shape[0]
        channel_num = sinogram_3d_img.shape[1]
        device = sinogram_3d_img.device

        sinogram_3d_img = torch.reshape(sinogram_3d_img,(batch_size,28,28))

        lighting_pattern = self.get_lighting_pattern(device)#(pattern_num,src_num)
        lighting_pattern = lighting_pattern.reshape((self.pattern_num,28,28))

        sinogram_3d_img = sinogram_3d_img[:,None,:,:]#(batch_size,1,height,width)
        lighting_pattern = lighting_pattern[None,:,:]#(batch_size,pattern_num,height,width)
        sinogram_3d_img_multiplexed =  sinogram_3d_img* lighting_pattern#(batch_size,pattern_num,src_num,det_num)
        sinogram_3d_img_multiplexed = sinogram_3d_img_multiplexed.reshape((batch_size,self.pattern_num,28*28))
        sinogram_3d_img_multiplexed = torch.sum(sinogram_3d_img_multiplexed,dim=2)#(batch_size,channel_num,pattern_num)

        return sinogram_3d_img_multiplexed