import torch
import torch.nn as nn
from .MLP_denoise import *
from .model_para_denoise import *

class model_conf(nn.Module):

    def __init__(self):
        super().__init__()
        self.model_para=model_para()

        #Khoi tao MLP
        self.MLP=MLP_conf_bl01(self.model_para.n_dim)

        #Dinh nghia Loss Function

        self.Loss= nn.BCEWithLogitsLoss()


    def forward( self, input_layer_val, expected_classes=None):

        #Du lieu chay qua cac layer tinh logits dau ra
        logits=self.MLP(input_layer_val)

        if expected_classes is None:
            return logits


        targets = expected_classes.float()
        
        # Dam bao targets co cung Shape [batch_size, 1] nhu logits
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)

        # Tinh loss voi logits
        loss = self.Loss(logits, targets)

        # Logits > 0 tuong duong voi Sigmoid(logits) > 0.5
        preds = (logits > 0).float()
        accuracy = torch.mean((preds == targets).float())

        return logits,loss,accuracy


