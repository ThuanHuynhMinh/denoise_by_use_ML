import torch
import torch.nn as nn
from .MLP_para_denoise import *

class MLP_conf_bl01(nn.Module):

    def __init__( self, input_dim):
        super().__init__()
    
        self.MLP_para=MLP_para()
        self.input_layer_val_dim = input_dim

        #cac tham so trong 5 layer
        self.fc1=nn.Linear(input_dim,self.MLP_para.l01_unit)
        self.fc2=nn.Linear(self.MLP_para.l01_unit,self.MLP_para.l02_unit)
        self.fc3=nn.Linear(self.MLP_para.l02_unit,self.MLP_para.l03_unit)
        self.fc4=nn.Linear(self.MLP_para.l03_unit,self.MLP_para.l04_unit)
        self.fc5=nn.Linear(self.MLP_para.l04_unit,self.MLP_para.l05_unit)



        #dropout cua 5 layer
        self.drop1=nn.Dropout(p=self.MLP_para.l01_drop_prob)
        self.drop2=nn.Dropout(p=self.MLP_para.l02_drop_prob)
        self.drop3=nn.Dropout(p=self.MLP_para.l03_drop_prob)
        self.drop4=nn.Dropout(p=self.MLP_para.l04_drop_prob)
        self.drop5=nn.Dropout(p=self.MLP_para.l05_drop_prob)

        #acti cua 4 layer
        self.act01=_apply_activation(act_name=self.MLP_para.l01_act_fun,is_act=self.MLP_para.l01_is_act)
        self.act02=_apply_activation(act_name=self.MLP_para.l02_act_fun,is_act=self.MLP_para.l02_is_act)
        self.act03=_apply_activation(act_name=self.MLP_para.l03_act_fun,is_act=self.MLP_para.l03_is_act)
        self.act04=_apply_activation(act_name=self.MLP_para.l04_act_fun,is_act=self.MLP_para.l04_is_act)
        self.act05=_apply_activation(act_name=self.MLP_para.l05_act_fun,is_act=self.MLP_para.l05_is_act)

    def forward(self, x):
        # Layer 1
        X = self.fc1(x)
        X=self.act01(X)
        if self.MLP_para.l01_is_drop: 
            X = self.drop1(X)

        # Layer 2
        X = self.fc2(X)
        X=self.act02(X)
        if self.MLP_para.l02_is_drop: 
            X = self.drop2(X)

        # Layer 3
        X = self.fc3(X)
        X=self.act03(X)
        if self.MLP_para.l03_is_drop: 
            X = self.drop3(X)

        # Layer 4
        X = self.fc4(X)
        X=self.act04(X)
        if self.MLP_para.l04_is_drop: 
            X = self.drop4(X)

        # Layer 5
        X = self.fc5(X)
        X=self.act05(X)
        if self.MLP_para.l05_is_drop: 
            X = self.drop5(X)

        return X

def _apply_activation(act_name,is_act):
    if is_act is True:
        if act_name == 'RELU':
            return nn.ReLU()
        if act_name == 'SIGMOID':
            return nn.Sigmoid()
        if act_name == 'TANH':
            return nn.Tanh()
        else:
            return nn.Identity()
    else:
        return nn.Identity()