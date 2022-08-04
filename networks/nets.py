from __future__ import absolute_import, division, print_function
#from msilib.schema import Class

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *

from .resnet_encoder import ResnetEncoder
from .hr_decoder import DepthDecoder
from .pose_decoder import PoseDecoder
from .mpvit import *


class DeepNet(nn.Module):
    def __init__(self,type,weights_init= "pretrained",num_layers=18,num_pose_frames=2,scales=range(4)):
        super(DeepNet, self).__init__()
        self.type = type
        self.num_layers=num_layers
        self.weights_init=weights_init
        self.num_pose_frames=num_pose_frames
        self.scales = scales

        if self.type =='depthnet':
            self.encoder = ResnetEncoder(
            self.num_layers, self.weights_init == "pretrained")
            self.decoder = DepthDecoder(
            self.encoder.num_ch_enc, self.scales)

        elif self.type =='mpvitnet':
            self.encoder = mpvit_small()
            #self.decoder = MPDecoder()
            self.decoder = DepthDecoder()
 
        elif self.type == 'posenet':
            self.encoder = ResnetEncoder(
                self.num_layers,
                self.weights_init == "pretrained",
                num_input_images=self.num_pose_frames)
            self.decoder = PoseDecoder(
                self.encoder.num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)
        else:
            print("wrong type of the networks, only depthnet and posenet")
    

    def forward(self, inputs):
                #self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
        if self.type =='depthnet': 
            self.outputs = self.decoder(self.encoder(inputs))
        elif self.type =='mpvitnet': 
            self.outputs = self.decoder(self.encoder(inputs))
        elif self.type =='posenet': 
            self.outputs = self.decoder([self.encoder(inputs)])
        return self.outputs