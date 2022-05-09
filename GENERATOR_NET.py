import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
import numpy as np
from scipy.stats import ortho_group
import math
from collections import OrderedDict


class GENERATOR_NET(nn.Module):
    def __init__(self,args):
        super(GENERATOR_NET,self).__init__()

        self.latent_code_len = args["latent_code_len"]
        self.src_num = args["src_num"]
        self.det_num = args["det_num"]

        self.decoder_stack = self.get_decoder_stack(self.latent_code_len)

    def get_decoder_stack(self,input_size,name_prefix="Decoder_"):
        layer_stack = OrderedDict()
        
        layer_count = 0

        output_size=500
        layer_stack[name_prefix+"FC_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=500
        layer_stack[name_prefix+"FC_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU()
        layer_count+=1
        input_size = output_size

        output_size=self.src_num
        layer_stack[name_prefix+"FC_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.Tanh()
        layer_count+=1
        input_size = output_size

        layer_stack = nn.Sequential(layer_stack)

        return layer_stack
    
    def forward(self,latent_codes):
        '''
        latent_codes = (batch_size,latent_code_len)
        return = (batch_size,layer_num,src_num)
        '''
        batch_size = latent_codes.shape[0]
        # assert batch_size == self.src_grp_num*self.num_per_grp,"{} {}".format(batch_size,self.src_grp_num*self.num_per_grp)
        decoded_local_sinogram = self.decoder_stack(latent_codes)
        decoded_local_sinogram = decoded_local_sinogram*0.5+0.5
        decoded_local_sinogram = decoded_local_sinogram.reshape((batch_size,1,28,28))

        return decoded_local_sinogram
