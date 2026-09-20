import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .MLP_para_class import *

class MLP_conf_bl01(nn.Module):

    def __init__( self, input_dim):
        super().__init__()
    
        self.MLP_para=MLP_para()
        self.input_layer_val_dim = input_dim

        #cac tham so trong 4 layer
        self.fc1=nn.Linear(input_dim,self.MLP_para.l01_unit)
        self.fc2=nn.Linear(self.MLP_para.l01_unit,self.MLP_para.l02_unit)
        self.fc3=nn.Linear(self.MLP_para.l02_unit,self.MLP_para.l03_unit)
        self.fc4=nn.Linear(self.MLP_para.l03_unit,self.MLP_para.l04_unit)

        #dropout cua 4 layer
        self.drop1=nn.Dropout(p=self.MLP_para.l01_drop_prob)
        self.drop2=nn.Dropout(p=self.MLP_para.l02_drop_prob)
        self.drop3=nn.Dropout(p=self.MLP_para.l03_drop_prob)
        self.drop4=nn.Dropout(p=self.MLP_para.l04_drop_prob)


    def _apply_activation(self, x, act_name):
        if act_name == 'RELU':
            return F.relu(x)
        elif act_name == 'GELU':
            return F.gelu(x)
        elif act_name == 'SIGMOID':
            return torch.sigmoid(x)
        return x

    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        if self.MLP_para.l01_is_act: 
            x = self._apply_activation(x, self.MLP_para.l01_act_fun)
        if self.MLP_para.l01_is_drop: 
            x = self.drop1(x)

        # Layer 2
        x = self.fc2(x)
        if self.MLP_para.l02_is_act: 
            x = self._apply_activation(x, self.MLP_para.l02_act_fun)
        if self.MLP_para.l02_is_drop: 
            x = self.drop2(x)

        # Layer 3
        x = self.fc3(x)
        if self.MLP_para.l03_is_act: 
            x = self._apply_activation(x, self.MLP_para.l03_act_fun)
        if self.MLP_para.l03_is_drop: 
            x = self.drop3(x)

        # Layer 4
        x = self.fc4(x)
        if self.MLP_para.l04_is_act: 
            x = self._apply_activation(x, self.MLP_para.l04_act_fun)
        if self.MLP_para.l04_is_drop: 
            x = self.drop4(x)

        return x