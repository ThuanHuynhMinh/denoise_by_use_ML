import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


from .MLP_class import *
from .model_para_class import *



class model_conf(nn.Module):

    def __init__(self):
        super().__init__()

        self.model_para=model_para()

        #Khoi tao MLP
        self.MLP=MLP_conf_bl01(self.model_para.n_dim)

        #Dinh nghia Loss Function

        self.Loss= nn.CrossEntropyLoss()


    def forward( self, input_layer_val, expected_classes=None):

        #Du lieu chay qua cac layer tinh logits dau ra
        logits=self.MLP(input_layer_val)

        if expected_classes is None:
            return logits

        #Chuyen nhan one-hit sang index

        if expected_classes.ndim > 1 and expected_classes.size(1) > 1:
            targets = torch.argmax(expected_classes, dim=1)
        else:
            targets = expected_classes.long()

        #Tinh loss voi logits

        loss=self.Loss(logits,targets)


        #Tinh accury
        preds=torch.argmax(logits,dim=1)
        accuracy=torch.mean((preds == targets).float())

        return logits,loss,accuracy






