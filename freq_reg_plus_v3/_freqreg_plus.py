import sys
# sys.path.append("/home/cqzhao/projects/matrix/")

import copy

import os
import torch
import torch.nn as nn

import math
import numpy as np


import torch.nn.functional as F

np.set_printoptions(precision=4, suppress=True)



from torch.fft import irfftn, rfftn

# from common_py.utils import setupSeed


from torch.fft import irfftn, rfftn

from torch.autograd import Variable


# import freq_reg._dct as fr
# import conv_dct


from ._dct import dct_2d, idct_2d, idct1_2d, dct, idct, idct_4d, dct_4d, dct_3d, idct_3d
from ._dctbase import dct2Base, idct2Base

from time import time

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=9999999999999999999)


np.set_printoptions(precision=2, suppress=True)
torch.set_printoptions(precision=2, sci_mode=False)


PI = np.pi

def getZIdx_2d(data):
     
    zmat = torch.empty(data.shape)


    maxval1 = data.shape[0] + 1.0
    maxval2 = data.shape[1] + 1.0


    matdim1 = torch.tensor(range(data.shape[0])) + torch.tensor(range(data.shape[0]))/maxval1
    matdim1 = matdim1.expand(data.shape[1], data.shape[0])

    matdim2 = torch.tensor(range(data.shape[1])) + torch.tensor(range(data.shape[1]))/(maxval1*maxval2)
    matdim2 = matdim2.expand(data.shape[0], data.shape[1])

    matdim = matdim1.permute(1, 0) + matdim2.permute(0, 1)

    
    return matdim   



def getZIdx_4d(data):
     
    zmat = torch.empty(data.shape)


    maxval1 = data.shape[0] + 1.0
    maxval2 = data.shape[1] + 1.0
    maxval3 = data.shape[2] + 1.0
    maxval4 = data.shape[3] + 1.0


    matdim1 = torch.tensor(range(data.shape[0])) + torch.tensor(range(data.shape[0]))/maxval1
    matdim1 = matdim1.expand(data.shape[1], data.shape[2], data.shape[3], data.shape[0])

    matdim2 = torch.tensor(range(data.shape[1])) + torch.tensor(range(data.shape[1]))/(maxval1*maxval2)
    matdim2 = matdim2.expand(data.shape[0], data.shape[2], data.shape[3], data.shape[1])

    matdim3 = torch.tensor(range(data.shape[2])) + torch.tensor(range(data.shape[2]))/(maxval1*maxval2*maxval3)
    matdim3 = matdim3.expand(data.shape[0], data.shape[1], data.shape[3], data.shape[2])

    matdim4 = torch.tensor(range(data.shape[3])) + torch.tensor(range(data.shape[3]))/(maxval1*maxval2*maxval3*maxval4)
    matdim4 = matdim4.expand(data.shape[0], data.shape[1], data.shape[2], data.shape[3])


    matdim = matdim1.permute(3, 0, 1, 2) + matdim2.permute(0, 3, 1, 2) + matdim3.permute(0, 1, 3, 2) + matdim4


    return matdim



class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 128, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 196),
            nn.ReLU(inplace=True),
            nn.Linear(196, num_classes),
        )

        self.frconv2d = Conv2d_fr_2d(20, 1, 5, stride=2)


    def forward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)





def fftBase(N):

    n = torch.tensor(np.arange(N))
    k = n.reshape((N, 1))

    temp = k * n

    real_part = torch.cos(-2 * np.pi * temp / N)
    imag_part = torch.sin(-2 * np.pi * temp / N)


    M = real_part + 1j * imag_part

    return M


def ifftBase(N):
    n = torch.tensor(np.arange(N))
    k = n.reshape((N, 1))

    temp = k * n

    real_part = torch.cos(2 * np.pi * temp / N)
    imag_part = torch.sin(2 * np.pi * temp / N)

    M = (real_part + 1j * imag_part) / N

    return M


def zrSrch(mat, th=1e-6):

    data = mat.sum(-1)
        
    idx = data.abs() < th 

    return ~idx


def block_diag_stride(matrices, stride):

    total_height = max((m.shape[0] + stride * i for i, m in enumerate(matrices)), default=0)
    total_width = sum(m.shape[1] for m in matrices)
    
    output = torch.zeros(total_height, total_width, dtype=matrices[0].dtype, device=matrices[0].device)
    
    current_col = 0
    for i, matrix in enumerate(matrices):
        height, width = matrix.shape
        vertical_offset = stride * i
        output[vertical_offset:vertical_offset+height, current_col:current_col+width] = matrix
        current_col += width
    
    return output





def printAcc(out3, labs):


    outlab = out3.real.cpu().squeeze()

    outlab = F.log_softmax(outlab, dim=1)


    labs_prd = outlab.argmax(dim=1).detach().cpu()
    labs_tru = labs.detach().cpu()

    eqidx = labs_prd == labs_tru
    accuracy = torch.sum(eqidx)/labs.shape[0]
    
    print("fr accuracy:", accuracy.item())



def evenIdx(idx):
    
    if (idx.numel() < 2):
        return idx

    idxf = torch.flip(idx[1:], [0])
    idxf = torch.cat((idxf[:2], idxf), dim=0)[1:]
    
    re_idx = idx | idxf
    return re_idx




def toComplex(data):

    data_cp = torch.complex(data, torch.zeros_like(data))

    return data_cp






def getRsMat(mat, num=4):
    
    mat_ex = insZeros(mat, num-1)

#    print(block_diag_stride([mat_ex[:, 0].squeeze().unsqueeze(-1) for _ in range(num)], stride=1).shape)
    
    mat_ex = torch.cat([block_diag_stride([mat_ex[:, j].squeeze().unsqueeze(-1) for _ in range(num)], stride=1) for j in range(mat_ex.shape[1])], dim=1)
   
    return mat_ex






def insZeros(tensor, stride=1):
    M, N = tensor.shape
    new_rows = M + (M - 1) * stride
    new_tensor = torch.zeros((new_rows, N), dtype=tensor.dtype)
    
    for i in range(M):
        new_tensor[i * (1 + stride)] = tensor[i]
    
    return new_tensor




def dropMat4D(mat):

    DROPMAT = torch.zeros_like(mat)

    pos0 = torch.tensor(range(mat.shape[0])).repeat(mat.shape[1], mat.shape[2], mat.shape[3], 1).permute(3, 0, 1, 2)
    pos1 = torch.tensor(range(mat.shape[1])).repeat(mat.shape[0], mat.shape[2], mat.shape[3], 1).permute(0, 3, 1, 2)
    pos2 = torch.tensor(range(mat.shape[2])).repeat(mat.shape[0], mat.shape[1], mat.shape[3], 1).permute(0, 1, 3, 2)
    pos3 = torch.tensor(range(mat.shape[3])).repeat(mat.shape[0], mat.shape[1], mat.shape[2], 1).permute(0, 1, 2, 3)

    posmat = torch.cat((pos0.unsqueeze(-1),
                        pos1.unsqueeze(-1), 
                        pos2.unsqueeze(-1), 
                        pos3.unsqueeze(-1)), dim=-1).to(mat.device)
    
    
    maxdis = 0
   
    while maxdis < max(mat.shape):
        idx = (posmat - maxdis).max(dim=-1)[0] < 1
        idx_t = DROPMAT < 0.1
        idx = idx & idx_t
        DROPMAT[idx] += (torch.tensor(range(idx.sum())).to(mat.device) + DROPMAT.max()) + 1

        maxdis += 1
    
    return DROPMAT



def conSyExtend(data):

    data_sy = data[:, 1:]
    data_sy = torch.flip(data_sy, [1])
    data_sy.imag *= -1

    re_data = torch.cat((data, data_sy), dim=1)

    return re_data



def conSyExtend_even(data):

    data_sy = data[:, 1:-1]
    data_sy = torch.flip(data_sy, [1])
    data_sy.imag *= -1

    re_data = torch.cat((data, data_sy), dim=1)
    
    return re_data


def reluFr_ri(data):

    N = data.shape[1]

    data_fr = torch.fft.rfftn(data, dim=1)
    
    data_fr.real = F.relu(data_fr.real)
    data_fr.imag = F.relu(data_fr.imag)

    data_ = torch.fft.irfftn(data_fr, dim=1, s=N)
    
    return data_ 


def ZIdx2Int(inputmat):
    mat = inputmat.view(-1)
    idxpos = torch.tensor(range(mat.numel() ))

    packmat = torch.cat((mat.unsqueeze(-1), idxpos.unsqueeze(-1)), dim=-1)
    packmat = packmat[packmat[:, 0].sort()[1]]
    
    x = torch.cat((packmat, idxpos.unsqueeze(-1)), dim=-1)[:, 1:]
    intZidx = (x[x[:, 0].sort()[1]])[:, 1].squeeze()

    intZidx = intZidx.reshape(inputmat.shape)

    return intZidx


class Conv2d_fr_4d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, minrate=0.01, droprate=0.01, dynamicdrop=False, groups=1, dilation=1):
        super().__init__()

        
        self.weight = nn.Parameter(torch.fft.fftn(torch.empty(out_channels, in_channels//groups, kernel_size, kernel_size).normal_(0, 0.1)))

        self.conv_size = torch.empty(out_channels, in_channels//groups, kernel_size, kernel_size).normal_(0, 0.1).shape

        self.stride     = stride
        self.padding    = padding
        self.groups     = groups
        self.dilation   = dilation

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels).normal_(0, 0.1))
        else:
            self.bias = None


        self.register_buffer("ZMAT",    ZIdx2Int(dropMat4D(torch.empty(out_channels, in_channels//groups, kernel_size, kernel_size))))
        self.register_buffer("IDROP",  torch.zeros(out_channels, in_channels//groups, kernel_size, kernel_size) + 1.0)

        self.minrate = minrate
        self.minnum = max(round(self.IDROP.numel()*minrate), 127)
        
        self.keeprate = 1.0
        self.droprate = droprate 
        self.dynamicdrop = dynamicdrop

        
        self.threval = self.IDROP.numel()

        self.weightrate = 0
        self.weightnum = -1

    def setDroprate(self, droprate):
        self.droprate = droprate

    def setminnum(self, minnum, protectnum=127):
        self.minnum = max(minnum, protectnum)
        self.minrate = self.minnum/(self.IDROP.numel()*1.0)
        self.IDROP.fill_(1.0)
        self.keeprate = 1.0

    def setminrate(self, minrate, protectnum=127):
        self.minrate = minrate
        self.minnum = max(round(self.IDROP.numel()*minrate), 127)
        self.IDROP.fill_(1.0)
        self.keeprate = 1.0

    def resetparams(self):
        self.minnum = max(round(self.weight.numel()*self.minrate//self.groups), 127)
        self.IDROP.fill_(1.0)
        self.keeprate = 1.0
        self.threval = self.IDROP.numel()
    
    def frPrunning(self):
        print("Hell oworld")

        self.weight_infer = self.weight.clone()

    def forward(self, data):

        if self.threval <= self.minnum + 1:
            if self.dynamicdrop:
                self.IDROP.fill_(1.0)
                self.keeprate = 1.0
        else:
            self.threval = max(self.keeprate*self.IDROP.numel(), self.minnum)
            self.IDROP[self.ZMAT > self.threval] = 0

            self.keeprate = self.keeprate - self.droprate*(self.keeprate - self.minrate)


        # tracking the drop rate and drop num
        self.weightnum = self.IDROP.sum().item()
        self.weightrate = (self.weightnum/self.IDROP.numel())

        dropweight = self.weight*self.IDROP 
        weight = torch.fft.irfftn(dropweight, s=self.conv_size)
        
        output = F.conv2d(data, weight, self.bias, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)

        return output


def prePruConv(conv, IDROP, stride=2):

    conv = conv*IDROP
    

    conv = torch.fft.irfftn(conv, s=conv.shape)

    ifbase = ifftBase(conv.shape[1]).to(conv.device)
    
    conv_fr = torch.fft.fftn(conv, dim=1)
    conv_fr.imag *= -1

    conv_fr = conv_fr * ifbase[0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    

    conv_fr = torch.fft.fftn(conv_fr, dim=0)
    

    return conv_fr


def pruConv(conv, IDROP, thval=1e-6):
    
    conv = prePruConv(conv, IDROP)

    idxmat = IDROP.sum(-1).sum(-1).sum(-1)

    idx = idxmat > thval
    idx = evenIdx(idx)
    
    # conv = conv[idx]
    idxmat_ex = IDROP.sum(-1).sum(-1).sum(0)
    idx_ex = idxmat_ex > thval
    idx_ex = evenIdx(idx_ex)

    return conv, idx, idx_ex


def prePruFC(mat_cl, IDROP):
    mat_cl = mat_cl*IDROP
    mat_cl = torch.fft.ifftn(mat_cl, dim=1)
    
    # 这里之前乘了IDROP 所有要对称共厄一下
    mat_fr = torch.fft.irfftn(mat_cl, dim=0, s=mat_cl.shape[0])
    mat_cl = torch.fft.fftn(mat_fr, dim=0)
    
    ifbase = ifftBase(mat_cl.shape[0]).to(mat_cl.device)

    mat_cl.imag *= -1
    
    ifvec = ifbase[0]
    finmat = mat_cl * ifvec.unsqueeze(-1)
    

    fbase_out = fftBase(mat_cl.shape[1]).to(mat_cl.device)
    finmat = finmat @ fbase_out

    return finmat


def pruFC(fc, IDROP, thval=1e-6):

    fc = prePruFC(fc, IDROP)

    idxmat = IDROP.sum(-1)

    idx = idxmat > thval
    idx = evenIdx(idx)

    idx_ex = IDROP.sum(0) > thval
    idx_ex = evenIdx(idx_ex)
    
    # fc = fc[idx]

    return fc, idx, idx_ex


class Conv2d_fr_2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, minrate=0.01, droprate=0.01, dynamicdrop=False, groups=1, dilation=1):
        super().__init__()

        
        self.weight = nn.Parameter(torch.fft.fftn(torch.empty(out_channels, in_channels//groups, kernel_size, kernel_size).normal_(0, 0.1)))

        self.conv_size = torch.empty(out_channels, in_channels//groups, kernel_size, kernel_size).normal_(0, 0.1).shape

        self.stride     = stride
        self.padding    = padding
        self.groups     = groups
        self.dilation   = dilation

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels).normal_(0, 0.1))
        else:
            self.bias = None


        self.register_buffer("ZMAT",    ZIdx2Int(dropMat4D(torch.empty(out_channels, in_channels//groups, 1, 1))))
        self.register_buffer("IDROP",  torch.zeros_like(self.ZMAT) + 1.0)

        self.minrate = minrate
        self.minnum = max(round(self.IDROP.numel()*minrate), 1)
        
        self.keeprate = 1.0
        self.droprate = droprate 
        self.dynamicdrop = dynamicdrop

        
        self.threval = self.IDROP.numel()

        self.weightrate = 0
        self.weightnum = -1


        self.infer_fr = False
        self.infer_pru = False

    def setDroprate(self, droprate):
        self.droprate = droprate

    def setminnum(self, minnum, protectnum=127):
        self.minnum = max(minnum, protectnum)
        self.minrate = self.minnum/(self.IDROP.numel()*1.0)
        self.IDROP.fill_(1.0)
        self.keeprate = 1.0

    def setminrate(self, minrate, protectnum=127):
        self.minrate = minrate
        self.minnum = max(round(self.IDROP.numel()*minrate), 127)
        self.IDROP.fill_(1.0)
        self.keeprate = 1.0

    def resetparams(self):
        self.minnum = max(round(self.weight.numel()*self.minrate//self.groups), 127)
        self.IDROP.fill_(1.0)
        self.keeprate = 1.0
        self.threval = self.IDROP.numel()

    def frPrunning(self):
        weight_fr, idx_outch, idx_inpch = pruConv(self.weight, self.IDROP)
        
        self.infer_fr       = True

        self.idx_inpch = idx_inpch
        self.idx_outch = idx_outch
        self.weight_fr = weight_fr


    def forward(self, data, prev_mod=None):
        
        if self.infer_fr:
            if prev_mod == None:
                return F.conv2d(data, self.weight_fr, self.bias, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)[:, self.idx_outch]
            else:
                
                idx_fin = self.idx_inpch[prev_mod.idx_outch]
                idx_fin_ = self.idx_inpch & prev_mod.idx_outch


                return F.conv2d(data[:, idx_fin], self.weight_fr[:, idx_fin_], self.bias, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)[:, self.idx_outch]


        if self.threval <= self.minnum + 1:
            if self.dynamicdrop:
                self.IDROP.fill_(1.0)
                self.keeprate = 1.0
        else:
            self.threval = max(self.keeprate*self.IDROP.numel(), self.minnum)
            self.IDROP[self.ZMAT > self.threval] = 0

            self.keeprate = self.keeprate - self.droprate*(self.keeprate - self.minrate)


        # tracking the drop rate and drop num
        self.weightnum = self.IDROP.sum().item()
        self.weightrate = (self.weightnum/self.IDROP.numel())

        dropweight = self.weight*self.IDROP
        weight = torch.fft.irfftn(dropweight, s=self.conv_size)
        
        output = F.conv2d(data, weight, self.bias, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)

        return output



class Linear_fft_2d(nn.Module):
    def __init__(self, in_features, out_features, bias=True, minrate=0.01, droprate=0.01, dynamicdrop=False):
        super().__init__()

        self.weight = nn.Parameter(torch.fft.fftn(torch.empty(out_features, in_features).normal_(0, 0.1)))

        self.weight_size = self.weight.shape

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features).normal_(0, 0.1))
        else:
            self.bias = None


        self.register_buffer("ZMAT",    ZIdx2Int(dropMat4D(torch.empty(out_features, in_features, 1, 1)).squeeze(-1).squeeze(-1)))
        self.register_buffer("IDROP",   torch.zeros(out_features, in_features).squeeze() + 1.0)

        self.minrate = minrate
        self.minnum = max(round(self.IDROP.numel()*minrate), 255)
        

        self.keeprate = 1.0
        self.droprate = droprate 
        self.dynamicdrop = dynamicdrop

        
        self.threval = self.IDROP.numel()

        self.weightrate = 0
        self.weightnum = -1     

        self.infer_fr = False


    def setDroprate(self, droprate):
        self.droprate = droprate

    def setminnum(self, minnum, protectnum=255):
        self.minnum = max(minnum, protectnum)
        self.minrate = self.minnum/(self.IDROP.numel()*1.0)
        self.IDROP.fill_(1.0)
        self.keeprate = 1.0
        self.threval = self.IDROP.numel()

    def setminrate(self, minrate, protectnum=255):
        self.minrate = minrate
        self.minnum = max(round(self.IDROP.numel()*minrate), protectnum)
        self.IDROP.fill_(1.0)
        self.keeprate = 1.0
        self.threval = self.IDROP.numel()

    def resetparams(self):
        self.minnum = max(round(self.weight.numel()*self.minrate), 255)
        self.IDROP.fill_(1.0)
        self.keeprate = 1.0
        self.threval = self.IDROP.numel()


    def frPrunning(self):

        weight_fr, idx_inpch, idx_outch = pruFC(self.weight.t(), self.IDROP.t())
        
        self.weight_fr = weight_fr.t()
        self.idx_inpch = idx_inpch
        self.idx_outch = idx_outch
        
        self.infer_fr = True


    def forward(self, data, prev_mod=None):
        
        if self.infer_fr:
            if prev_mod == None:

                return F.linear(data, self.weight_fr[:, self.idx_inpch], self.bias)[:, self.idx_outch]
            else:
                idx_fin = self.idx_inpch[prev_mod.idx_outch]
                idx_fin_ = self.idx_inpch & prev_mod.idx_outch

                return F.linear(data[:, idx_fin], self.weight_fr[:, idx_fin_], self.bias)[:, self.idx_outch]


        if self.threval <= self.minnum + 1:
            if self.dynamicdrop:
                self.IDROP.fill_(1.0)
                self.keeprate = 1.0
        else:
            self.threval = max(self.keeprate*self.IDROP.numel(), self.minnum)
            self.IDROP[self.ZMAT > self.threval] = 0

            self.keeprate = self.keeprate - self.droprate*(self.keeprate - self.minrate)


        # tracking the drop rate and drop num
        self.weightnum = self.IDROP.sum().item()
        self.weightrate = self.weightnum/self.IDROP.numel()

        dropweight = self.weight*self.IDROP
        weight = torch.fft.irfftn(dropweight, s=self.weight_size)

        output = F.linear(data, weight, self.bias)


        return output







class BatchNorm2d_fr(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        super().__init__()

        self.norm = nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
    
    def forward(self, data):

        data_fr = torch.fft.fftn(data, dim=1)

        data_fr.real = self.norm(data_fr.real)
        data_fr.imag = self.norm(data_fr.imag)


        return torch.fft.ifftn(data_fr, dim=1).real





# dct-i 型的 linear
class Linear_dct1(nn.Module):
    def __init__(self, in_features, out_features, bias=True, minrate=0.01, droprate=0.01, dynamicdrop=False):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.1))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features).normal_(0, 0.1))
        else:
            self.bias = None

        # self.register_buffer("ZMAT",    ZIdx2Int(getZIdx_2d(torch.empty(out_features, in_features)).squeeze()))
        self.register_buffer("ZMAT",    ZIdx2Int(dropMat4D(torch.empty(out_features, in_features, 1, 1)).squeeze(-1).squeeze(-1)))

        self.register_buffer("IDROP",   torch.zeros(out_features, in_features).squeeze() + 1.0)

        self.minrate = minrate
        self.minnum = max(round(self.IDROP.numel()*minrate), 255)
        

        self.keeprate = 1.0
        self.droprate = droprate 
        self.dynamicdrop = dynamicdrop

        
        self.threval = self.IDROP.numel()

        self.weightrate = 0
        self.weightnum = -1     


    def setDroprate(self, droprate):
        self.droprate = droprate

    def setminnum(self, minnum, protectnum=255):
        self.minnum = max(minnum, protectnum)
        self.minrate = self.minnum/(self.IDROP.numel()*1.0)
        self.IDROP.fill_(1.0)
        self.keeprate = 1.0
        self.threval = self.IDROP.numel()

    def setminrate(self, minrate, protectnum=255):
        self.minrate = minrate
        self.minnum = max(round(self.IDROP.numel()*minrate), protectnum)
        self.IDROP.fill_(1.0)
        self.keeprate = 1.0
        self.threval = self.IDROP.numel()

    def resetparams(self):
        self.minnum = max(round(self.weight.numel()*self.minrate), 255)
        self.IDROP.fill_(1.0)
        self.keeprate = 1.0
        self.threval = self.IDROP.numel()

    def forward(self, data):

        if self.threval <= self.minnum + 1:
            if self.dynamicdrop:
                self.IDROP.fill_(1.0)
                self.keeprate = 1.0
        else:
            self.threval = max(self.keeprate*self.IDROP.numel(), self.minnum)
            self.IDROP[self.ZMAT > self.threval] = 0

            self.keeprate = self.keeprate - self.droprate*(self.keeprate - self.minrate)


        # tracking the drop rate and drop num
        self.weightnum = self.IDROP.sum().item()
        self.weightrate = self.weightnum/self.IDROP.numel()

        dropweight = self.weight*self.IDROP
        weight = idct1_2d(dropweight)
        
        data = data*2
        data[:,  0] = data[:, 0] *0.5
        data[:, -1] = data[:, -1]*0.5
        output = F.linear(data, weight, self.bias)

        return output









## ======================================================================================= ##
## ======================================================================================= ##
##                                                                                 ======= ##
## borderline：上面的代码包含fft的conv和linear，dct-i的conv和linear        
## 下面的代码代码就是实际使用的，dct-ii 的conv和linear                     
##                                                                                 ======= ##
## ======================================================================================= ##
## ======================================================================================= ##

def avg_pool2d(data, kernel_size=4, stride=1):

    if type(data).__name__ == 'PruTracker':
        data.m_data = F.avg_pool2d(data.m_data, kernel_size, stride=stride)
    else:
        data = F.avg_pool2d(data, kernel_size, stride=stride)

    return data 


def add(data1, data2):

    if type(data1).__name__ == 'PruTracker':
        re_data = data1 + data2
    else:
        num = max(data1.shape[1], data2.shape[1])
        re_data = chPadding(data1, num) + chPadding(data2, num)

    return re_data


# #torch.nn.functional.pad(input, pad, mode='constant', value=None)
# def pad(inpdata, pad, mode='constant', value=None):
# 
#     if type(inpdata).__name__ == 'PruTracker':
# 
#         m_data = inpdata.m_data
#         outdata = F.pad(m_data, pad, mode, value)
# 
#         return PruTracker(outdata, inpdata.m_datatype)
#     else:
# 
#         outdata = F.pad(inpdata, pad, mode, value)
# 
#         return outdata


def chPadding(x, N):

    if x.shape[1] < N:

        cutshape = x.shape[1]

        if len(x.shape) > 2:
            zmat = torch.zeros(x.shape[0], N - x.shape[1], x.shape[2], x.shape[3]).to(x.device)
        else:
            zmat = torch.zeros(x.shape[0], N - x.shape[1]).to(x.device)
        
        x = torch.cat((x,zmat), dim=1)
        return x
    else:
        return x


def idctCh(x):

    shape = x.shape
    if len(x.shape) == 2:
        x = x.unsqueeze(-1).unsqueeze(-1)

    x = idct(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    x = x.reshape(shape)

    return x



class PruTracker():
    def __init__(self, data, datatype='sp'):

        self.m_data = data

        self.m_datatype = datatype
   
    # prunning会导致 skip connection的结果不一致。想解决就只能追踪整个计算图
    # 太复杂，最简单的方式就是输入tracker而不是tensor，这样重载会调用这个加法函数
    def __add__(self, tracker):

        if isinstance(tracker, PruTracker):
            re_datatype = self.m_datatype
            num = max(self.m_data.shape[1], tracker.m_data.shape[1])
            re_data = chPadding(self.m_data, num) + chPadding(tracker.m_data, num)
        else:
            print("PruData Error: data type must be the same")
        
        return PruTracker(re_data, re_datatype)


    def __iadd__(self, tracker):

        num = max(self.m_data.shape[1], tracker.m_data.shape[1])
        re_data = chPadding(self.m_data, num) + chPadding(tracker.m_data, num)
        
        return PruTracker(re_data, self.m_datatype)
    
    
    def __getitem__(self, idx):
        return PruTracker(self.m_data[idx], self.m_datatype)


    def size(self, num=None):

        if num == None:
            return self.m_data.shape
        else:
            return self.m_data.shape[num]


    def squeeze(self, pos):
        
        self.m_data = self.m_data.squeeze(pos)

        return self

        
    m_datatype = 'sp'
    m_data = None




class  Conv2d_dct2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, minrate=0.01, droprate=0.01, dynamicdrop=False, groups=1, dilation=1):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels//groups, kernel_size, kernel_size).normal_(0, 0.1))

        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels).normal_(0, 0.1))
        else:
            self.bias = None


        # self.register_buffer("ZMAT",    ZIdx2Int(getZIdx_4d(torch.empty(out_channels, in_channels//groups, kernel_size, kernel_size))))
        self.register_buffer("ZMAT",    ZIdx2Int(dropMat4D(torch.empty(out_channels, in_channels//groups, kernel_size, kernel_size))))


        self.register_buffer("IDROP",  torch.zeros(out_channels, in_channels//groups, kernel_size, kernel_size) + 1.0)

        self.register_buffer("ibase_inp", idct2Base(in_channels))
        self.register_buffer("ibase_out", idct2Base(out_channels))
        self.register_buffer("ibase_ker", idct2Base(kernel_size))

        
        self.weight_cut = None
        self.inpnum = None
        self.prunning = None

        self.minrate = minrate
        self.minnum = max(round(self.IDROP.numel()*minrate), 2)
        
        self.keeprate = 1.0
        self.droprate = droprate 
        self.dynamicdrop = dynamicdrop

        
        self.threval = self.IDROP.numel()

        self.weightrate = 0
        self.weightnum = -1

        self.out_type = 'fr'


    def setDroprate(self, droprate):
        self.droprate = droprate

    def setminnum(self, minnum, protectnum=127):
        self.minnum = max(minnum, protectnum)
        self.minrate = self.minnum/(self.IDROP.numel()*1.0)
        self.IDROP.fill_(1.0)
        self.keeprate = 1.0

    def setminrate(self, minrate, protectnum=127):
        self.minrate = minrate
        self.minnum = max(round(self.IDROP.numel()*minrate), 127)
        self.IDROP.fill_(1.0)
        self.keeprate = 1.0

    def resetparams(self):
        self.minnum = max(round(self.weight.numel()*self.minrate//self.groups), 127)
        self.IDROP.fill_(1.0)
        self.keeprate = 1.0
        self.threval = self.IDROP.numel()

    def forward(self, inpdata):

        if self.prunning == 'done':
            
            if type(inpdata).__name__ == 'PruTracker':
                data_cut = inpdata.m_data[:, :self.inpnum]
            else:
                data_cut = inpdata[:, :self.inpnum]

            output = F.conv2d(data_cut, self.weight_cut, self.bias, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)
            
            return output
            # return PruTracker(output, self.out_type)


        if self.threval <= self.minnum + 1:
            if self.dynamicdrop:
                self.IDROP.fill_(1.0)
                self.keeprate = 1.0
        else:
            self.threval = max(self.keeprate*self.IDROP.numel(), self.minnum)
            self.IDROP[self.ZMAT > self.threval] = 0

            self.keeprate = self.keeprate - self.droprate*(self.keeprate - self.minrate)


        # tracking the drop rate and drop num
        self.weightnum = self.IDROP.sum().item()
        self.weightrate = (self.weightnum/self.IDROP.numel())

        dropweight = self.weight*self.IDROP.detach()


        if type(inpdata).__name__ == 'PruTracker':


            data = inpdata.m_data
            inptype = inpdata.m_datatype
            
            ifbase = ifftBase(dropweight.shape[1]*2)
            weight_fr = idct_2d(dropweight)
            
            if inptype == 'sp':
                data = dct(data.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            inpnum = (self.IDROP.sum(-1).sum(-1).sum(0) > 1e-6).sum()
            outnum = (self.IDROP.sum(-1).sum(-1).sum(-1) > 1e-6).sum()
            
            inpnum = min(inpnum, data.shape[1])
            
            data_cut = data[:, :inpnum]




            if self.weight_cut == None:

                weight_cut = (weight_fr[:outnum, :inpnum]*ifbase[0,0].real).detach()
                weight_cut[:, 0] *= 0.5
                self.weight_cut = nn.Parameter(weight_cut)

                self.inpnum = inpnum
                self.prunning = 'done'
                
                print("conv2d prunning information: ============================================")
                print("original weight:", self.weight.shape)
                print("prunned  weight:", self.weight_cut.shape)
                print("original data:", data.shape)
                print("prunned  data:", data_cut.shape)
                print("=========================================================================")
                print("")

                # 删除无用训练变量
                # 使用赋值None 更安全，del在删除不存在的变量会报错
                self.weight = None
                self.IDROP = None
                self.ZMAT = None
                self.ibase_inp = None
                self.ibase_out = None
                self.ibase_ker = None

            
            output = F.conv2d(data_cut, self.weight_cut, self.bias, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)#*ifbase[0,0].real

            return PruTracker(output, self.out_type)
            
        else:
            
            data = inpdata
            
            weight = dropweight

            weight = torch.tensordot(weight, self.ibase_ker, dims=([3], [0]))
            weight = torch.tensordot(weight, self.ibase_ker, dims=([2], [0])).permute(0, 1, 3, 2)
            weight = torch.tensordot(weight, self.ibase_inp, dims=([1], [0])).permute(0, 3, 1, 2)
            weight = torch.tensordot(weight, self.ibase_out, dims=([0], [0])).permute(3, 0, 1, 2)
            
            output = F.conv2d(data, weight, self.bias, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)
            
            return output
        



class Linear_dct2(nn.Module):
    def __init__(self, in_features, out_features, bias=True, minrate=0.01, droprate=0.01, dynamicdrop=False):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.1))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features).normal_(0, 0.1))
        else:
            self.bias = None


        self.register_buffer("ZMAT",    ZIdx2Int(dropMat4D(torch.empty(out_features, in_features, 1, 1)).squeeze(-1).squeeze(-1)))
        self.register_buffer("IDROP",   torch.zeros(out_features, in_features).squeeze() + 1.0)

        self.register_buffer("ibase_inp", idct2Base(in_features))
        self.register_buffer("ibase_out", idct2Base(out_features))


        self.weight_cut = None
        self.inpnum = None
        self.prunning = None

        self.minrate = minrate
        self.minnum = max(round(self.IDROP.numel()*minrate), 128)
        

        self.keeprate = 1.0
        self.droprate = droprate 
        self.dynamicdrop = dynamicdrop

        
        self.threval = self.IDROP.numel()

        self.weightrate = 0
        self.weightnum = -1     

        self.out_type = 'fr'


    def setDroprate(self, droprate):
        self.droprate = droprate

    def setminnum(self, minnum, protectnum=255):
        self.minnum = max(minnum, protectnum)
        self.minrate = self.minnum/(self.IDROP.numel()*1.0)
        self.IDROP.fill_(1.0)
        self.keeprate = 1.0
        self.threval = self.IDROP.numel()

    def setminrate(self, minrate, protectnum=255):
        self.minrate = minrate
        self.minnum = max(round(self.IDROP.numel()*minrate), protectnum)
        self.IDROP.fill_(1.0)
        self.keeprate = 1.0
        self.threval = self.IDROP.numel()

    def resetparams(self):
        self.minnum = max(round(self.weight.numel()*self.minrate), 255)
        self.IDROP.fill_(1.0)
        self.keeprate = 1.0
        self.threval = self.IDROP.numel()

    def forward(self, inpdata):
        
        # 如果已经被prunning 执行这段代码
        if self.prunning == 'done':
            
            if type(inpdata).__name__ == 'PruTracker':
                data_cut = inpdata.m_data[:, :self.inpnum]
            else:
                data_cut = inpdata[:, :self.inpnum]

            output = F.linear(data_cut, self.weight_cut)
           
            if self.bias != None:
                if self.bias.numel() != output.shape[1]:
                    output = chPadding(output, self.bias.numel())

                output += dct(self.bias)

            # return PruTracker(output, self.out_type)
            return output





        if self.threval <= self.minnum + 1:
            if self.dynamicdrop:
                self.IDROP.fill_(1.0)
                self.keeprate = 1.0
        else:
            self.threval = max(self.keeprate*self.IDROP.numel(), self.minnum)
            self.IDROP[self.ZMAT > self.threval] = 0

            self.keeprate = self.keeprate - self.droprate*(self.keeprate - self.minrate)


        # tracking the drop rate and drop num
        self.weightnum = self.IDROP.sum().item()
        self.weightrate = self.weightnum/self.IDROP.numel()

        dropweight = self.weight*self.IDROP.detach()



        if type(inpdata).__name__ == 'PruTracker':
            data = inpdata.m_data
            inptype = inpdata.m_datatype

            if inptype == 'sp':
                data = dct(data)

            ifbase = ifftBase(dropweight.shape[1]*2)

            inpnum = (self.IDROP.sum(0) > 1e-6).sum()
            outnum = (self.IDROP.sum(-1) > 1e-6).sum()
            
            inpnum = min(inpnum, data.shape[1])
            
            data_cut = data[:, :inpnum]

            if self.weight_cut == None:
                weight_cut = dropweight[:outnum, :inpnum]*ifbase[0,0].real
                weight_cut[:, 0] *= 0.5
                self.weight_cut = nn.Parameter(weight_cut)

                self.inpnum = inpnum
                self.prunning = 'done'


                print("linear prunning information: ============================================")
                print("original weight:", self.weight.shape)
                print("prunned  weight:", self.weight_cut.shape)
                print("original data:", data.shape)
                print("prunned  data:", data_cut.shape)
                print("=========================================================================")
                print("")


                self.weight  = None
                self.IDROP = None
                self.ZMAT = None
                self.ibase_inp = None
                self.ibase_out = None



            output = F.linear(data_cut, self.weight_cut) 

           
            if self.bias != None:
                if self.bias.numel() != output.shape[1]:
                    output = chPadding(output, self.bias.numel())

                output += dct(self.bias)
            

            return PruTracker(output, self.out_type)

        else:
            data = inpdata

            weight = ((dropweight @ self.ibase_inp).t() @ self.ibase_out).t()

            output = F.linear(data, weight, self.bias)

            return output







# 目前只接受数据格式为 N * ch * W * H 或者 N * ch 
# dct 不用大小的基结果不一样，所以要指定channel大小。如果不指定，第一次遇到数据的默认为N大小
class ReLU_dct2(nn.Module):
    def __init__(self, N=0, inplace=False):
        super().__init__()
        
        self.inplace = inplace
        self.out_type = 'fr'

        self.prunning = None

        self.N = N
        
        if N > 0:
            self.register_buffer("base",   dct2Base(N))
            self.register_buffer("ibase",   idct2Base(N))


    def forward(self, inpdata):

        if self.prunning == 'done':


            if type(inpdata).__name__ == 'PruTracker':
                data_cut = inpdata.m_data #[:, :self.inpnum]
            else:
                data_cut = inpdata#[:, :self.inpnum]
           
            output = F.relu(data_cut)

            # return PruTracker(output, self.out_type)
            return output




        if type(inpdata).__name__ == 'PruTracker':
            data = inpdata.m_data
            inptype = inpdata.m_datatype


            shape = data.shape
            if len(data.shape) == 2:
                data = data.unsqueeze(-1).unsqueeze(-1)


            if inptype == 'sp':
                data = dct(data.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            
            output = F.relu(data, inplace=self.inplace)

            if self.out_type == 'sp':
                output = idct(output.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            self.prunning = 'done'

            print("relu prunning information: ==============================================")
            print("prunned  weight:", data.shape)
            print("=========================================================================")
            print("")


            self.base = None
            self.ifbase = None


            return PruTracker(output.reshape(shape), self.out_type)

        else:
            data = inpdata

            shape = data.shape
            if len(data.shape) == 2:
                data = data.unsqueeze(-1).unsqueeze(-1)

            if self.N < 1:
                N = data.shape[1]
                self.N = N

                self.base = dct2Base(N).to(data.device)
                self.ibase = idct2Base(N).to(data.device)


            data = torch.tensordot(data, self.base.detach(), dims=([1], [0])).permute(0, 3, 1, 2)
            output = F.relu(data, inplace=self.inplace)

            output = torch.tensordot(output, self.ibase.detach(), dims=([1], [0])).permute(0, 3, 1, 2)
            
            return output.reshape(shape)



class BatchNorm2d_dct2(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        super().__init__()

        self.norm = nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
    
        self.inp_type = 'fr'
        self.out_type = 'fr'

        self.prunning = None

        self.N = num_features
        
        self.register_buffer("base",    dct2Base(self.N))
        self.register_buffer("ibase",   idct2Base(self.N))



    def forward(self, inpdata):
        
        if self.prunning == 'done':
 
            if type(inpdata).__name__ == 'PruTracker':
                data_cut = inpdata.m_data
            else:
                data_cut = inpdata

            cutshape = data_cut.shape[1]
            data_cut = chPadding(data_cut, self.N)

            output = self.norm(data_cut)
            # output = output[:, :cutshape]

            # return PruTracker(output, self.out_type)
            return output


        if type(inpdata).__name__ == 'PruTracker':


            data = inpdata.m_data
            inptype = inpdata.m_datatype
            

            if data.shape[1] != self.N:
                cutshape = data.shape[1]

                data = chPadding(data, self.N)

            self.prunning = 'done'
            output = self.norm(data)

            # 可能是计算误差？0的batchnorm不是0？
            # output = output[:, :cutshape]

            print("batchnorm2d prunning information: =======================================")
            print("original data:", data.shape)
            print("prunned  data:", output.shape)
            print("=========================================================================")
            print("")


            self.base = None
            self.ibase = None


            return PruTracker(output, self.out_type)

        else:
            data = inpdata
        
            data = torch.tensordot(data, self.base.detach(), dims=([1], [0])).permute(0, 3, 1, 2)
            output = self.norm(data)

            output = torch.tensordot(output, self.ibase.detach(), dims=([1], [0])).permute(0, 3, 1, 2)

            return output


class Reshape_dct2(nn.Module):
    def __init__(self):
        super().__init__()

        self.inpsize = -1
        self.outsize = -1

        self.out_type = 'fr'

        self.prunning = None

        self.ibase = None
        self.base = None

    def forward(self, x, *shape):
        
        # 这是正确的做法。但是由于神经网络的自适应性，直接reshape 让他自己学也没问题
        if self.prunning == 'done':

            data = x

            if len(data.shape) == 2:
                exflag = 1
                data = data.unsqueeze(-1).unsqueeze(-1)
            else:
                exflag = 0


            data = chPadding(data, self.inpsize)
          
            data = torch.tensordot(data, self.ibase, dims=([1], [0])).permute(0, 3, 1, 2)
            
            data = data.reshape(data.shape[0], -1)
            
            data = data @ self.base

            if exflag == 1:
                data = data.squeeze(-1).squeeze(-1)
           
            return data

        '''
        # 这是错误的做法。但是由于神经网络的自适应性，直接reshape 让他自己学也没问题
        # 这是错误的做法， 这是错误的做法，这是错误的做法！！！ 重要的事情说三遍。
        if self.prunning == 'done':

            data = x

            if len(data.shape) == 2:
                exflag = 1
                data = data.unsqueeze(-1).unsqueeze(-1)
            else:
                exflag = 0

            data = data.reshape(data.shape[0], -1)
            
            if exflag == 1:
                data = data.squeeze(-1).squeeze(-1)
           
            return data
        '''

        
        if type(x).__name__ == 'PruTracker':
            
            data = x.m_data
            inptype = x.m_datatype
            
            if len(data.shape) == 2:
                exflag = 1
                data = data.unsqueeze(-1).unsqueeze(-1)
            else:
                exflag = 0

            if self.ibase == None:
                self.ibase = idct2Base(self.inpsize).to(data.device)
                self.base = dct2Base(self.outsize).to(data.device)


            data = chPadding(data, self.inpsize)
            
            data = torch.tensordot(data, self.ibase, dims=([1], [0])).permute(0, 3, 1, 2)

            data = data.reshape(data.shape[0], -1)
            
            data = data @ self.base
            
            self.prunning = 'done'
            
            if exflag == 1:
                data = data.squeeze(-1).squeeze(-1)
           
            return PruTracker(data, 'fr')



        if self.inpsize == -1:
            self.inpsize = x.shape[1]

        x = x.reshape(shape)

        if self.outsize == -1:
            self.outsize = x.shape[1]

        return x


# 由于dct2的线性性，pool在 sp或者fr一样
# 封装只是为了控制输入输出所在的域
class AvgPool2d_dct2(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        super().__init__()
        
        self.pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override)

        self.inp_type = 'sp'
        self.out_type = 'sp'

    def forward(self, x):

        x = self.pool(x)

        return x

class AvgPool2d(AvgPool2d_dct2):
    pass

class ReLU(ReLU_dct2):
    pass

class BatchNorm2d(BatchNorm2d_dct2):
    pass

class Linear(Linear_dct2):
    pass

class Conv2d(Conv2d_dct2):
    pass

class Reshape(Reshape_dct2):
    pass

def main(argc, argv):
    print("hello world")


if __name__ == '__main__':


    argc = len(sys.argv)
    argv = sys.argv

    main(argc, argv)





