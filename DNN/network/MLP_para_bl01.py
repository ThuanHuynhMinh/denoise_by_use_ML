import numpy as np
import os

class MLP_para(object):
    """
    Dinh nghia class de chua tham so cua tung lop
    """

    def __init__(self):

        #=====Layer 01:full connection
        self.l01_unit           =4096
        self.l01_is_act         =True
        self.l01_act_fun        ='RELU'
        self.l01_is_drop        =True
        self.l01_drop_prob       =0.2


        #=====Layer 02:full connection
        self.l02_unit           =4096
        self.l02_is_act         =True
        self.l02_act_fun        ='RELU'
        self.l02_is_drop        =True
        self.l02_drop_prob       =0.2 

        #=====Layer 03:full connection
        self.l03_unit           =2048
        self.l03_is_act         =True
        self.l03_act_fun        ='RELU'
        self.l03_is_drop        =True
        self.l03_drop_prob       =0.2

        #=====Layer 04:full connection
        self.l04_unit           =1024
        self.l04_is_act         =True
        self.l04_act_fun        ='RELU'
        self.l04_is_drop        =True
        self.l04_drop_prob       =0.2

        #=====Layer 05:final layer
        self.l05_unit           = 784
        self.l05_is_act         =False
        self.l05_act_fun        ='NONE'
        self.l05_is_drop        =False
        self.l05_drop_prob       =0.0         