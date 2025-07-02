import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class CrossEntropy(nn.Module):
    def __init__(self, is_weight = False, weight = []):
        super(CrossEntropy, self).__init__()
        self.is_weight = is_weight
        self.weight = weight

    def forward(self, input, target , batchsize = 2):
        # target = target.squeeze(1).long()
        target = torch.argmax(target, dim=1)
        # print(input.shape)
        # print(target.shape)
        if self.is_weight == True:
            loss = F.cross_entropy(input, target, torch.tensor(self.weight).float().cuda())
        else:
            loss = F.cross_entropy(input, target)
        return loss