from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def parseCfg(cfg):
    file = open(cfg,'r')
    lines = file.read().split('\n')          #store lines
    lines = [x for x in lines if len(x)>0]   #get rid of empty lines
    lines = [x for x in lines if x[0]!='#']  #get rid of comment lines

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':
            if(len(block)!=0):
                blocks.append(block)
                block = {}
            block['type']=lines[1:-1]
        else:
            key,value = line.split('=')
            block[key.rstrip()]=value.lstrip()
    blocks.append(block)

    return blocks

def createModule(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    index = 0   # 为模块计数，模块中的层也可通过index取得
    prev_filters = 3
    output_filters = []

    for x in blocks:
        module = nn.Sequential()
        #卷积模块
        if(x["type"]=="convolutional"):
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size-1)//2
            else:
                pad = 0
            #添加卷积层
            conv =nn.Conv2d(prev_filters,filters,kernel_size,pad,bias=
                            module.add_module("conv_{0}".format(index),conv))
            #添加BN层
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index),bn)
            #添加激活层
            if(activation=="leaky"):
                act = nn.LeakyReLU(0.1,inplace=True)
                module.add_module("leaky_{0}".format((index)),act)
        #上采样模块
        elif(x["type"]=="upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2,mode="nearest")
            module.add_module("upsample_{0}".format(index),upsample)
