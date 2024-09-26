# from models.MVCNN import MVCNN
from __future__ import division, absolute_import
from models.fused_layer import Fusedformer
from models.resnet import resnet18
from tools.utils import calculate_accuracy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import argparse
import torch.optim as optim
import time
import torch.utils.checkpoint as cp
# from .Model import Model


class Img_encoder(nn.Module):

    def __init__(self, pre_trained = None):
        super(Img_encoder, self).__init__()

        if pre_trained:
            self.img_net = torch.load(pre_trained)
        else:
            print('---------Loading ImageNet pretrained weights --------- ')
            resnet18 = models.resnet18(pretrained=True)
            resnet18 = list(resnet18.children())[:-1]
            self.img_net = nn.Sequential(*resnet18)
            self.linear1 = nn.Linear(512, 256, bias=False)
            self.bn6 = nn.BatchNorm1d(256)
            

    def forward(self, img, img_v):
        
        # img_feat = cp.checkpoint(self.img_net, img)
        # img_feat_v = cp.checkpoint(self.img_net, img_v)
        img_feat = self.img_net(img)
        img_feat_v = self.img_net(img_v)
        img_feat = img_feat.squeeze(3)
        img_feat = img_feat.squeeze(2)
        img_feat_v = img_feat_v.squeeze(3)
        img_feat_v = img_feat_v.squeeze(2)

        img_feat = F.relu(self.bn6(self.linear1(img_feat)))
        img_feat_v = F.relu(self.bn6(self.linear1(img_feat_v)))
        

        final_feat = 0.5*(img_feat + img_feat_v)

        return final_feat
    

class HeadNet_dual_fused(nn.Module):

    def __init__(self, num_classes, num_modals=2, fused_type=None, mix_alpha=5.0):
        super(HeadNet_dual_fused, self).__init__()
        self.num_classes=num_classes
        self.num_modals = num_modals
        self.mix_alpha=mix_alpha
        self.u_head = nn.Sequential(*[nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, self.num_classes)])
        
        self.m_head = nn.Sequential(*[nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, self.num_classes)])
        
        self.fused_type = fused_type
        
        if fused_type=="attention":
            self.fused_layer= Fusedformer(dim=256,depth=2,heads=4,dim_head=64,mlp_dim=256,dropout=0.0)
        
        if fused_type=="concat":
            self.m_head = nn.Sequential(*[nn.Linear(256*num_modals, 128*num_modals), nn.ReLU(), nn.Linear(128*num_modals, self.num_classes)])
        
        

    def forward(self, img_feat, pt_feat, jointed=False):

        img_pred = self.u_head(img_feat)
        pt_pred = self.u_head(pt_feat)
        if self.fused_type=="attention":
            fuesed_feature = self.fused_layer(torch.cat((img_feat[:,None,:], pt_feat[:,None,:]), dim=1))
            fuesed_feature = fuesed_feature[:,0,:]
            joint_pred = self.m_head(fuesed_feature)
        elif self.fused_type=="add":
            joint_pred = self.m_head(img_feat+pt_feat)
        elif self.fused_type=="concat":
            joint_pred = self.m_head(torch.cat((img_feat, pt_feat), dim=-1))
        elif self.fused_type=="mixup":
            l = np.random.beta(self.mix_alpha, self.mix_alpha)        
            l = max(l, 1-l)
            joint_pred = self.m_head(l*img_feat+(1-l)*pt_feat)
            
        
        # mesh_pred = self.head(mesh_feat)
        if jointed:
            return img_pred, pt_pred, joint_pred
            
        return img_pred, pt_pred
