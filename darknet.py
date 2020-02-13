from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

#定义Empty模块
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer,self).__init__()
#定义锚点检测点
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


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
    filters = 0
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

        #route模块
        elif(x["type"]=="route"):
            x["layers"] = x["layers"].split(',')
            start = int(x["layers"][0])
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            if start > 0:
                start = start-index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index),route)
            if end < 0:
                filters = output_filters[index+start] + output_filters[index+end]
            else:
                filters = output_filters[index+start]

        #shortcut模块
        elif(x["type"]=="shortcut"):
            from_ = int(x["from"])
            shortcut = EmptyLayer()
            module.add_module("shortcut_{0}".format(index),shortcut)

        #YOLO模块
        elif(x["type"]=="yolo"):
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i],anchors[i+1]) for i in range(0,len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index),detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
if __name__ == '__main__':
    blocks = parseCfg("./yolov3.cfg")
    print(createModule(blocks))