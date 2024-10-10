import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.autograd import Variable

from ._dct import idct, idct_2d, idct_3d, idct_4d


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



def getZIdx_3d(data):
     
    zmat = torch.empty(data.shape)


    maxval1 = data.shape[0] + 1.0
    maxval2 = data.shape[1] + 1.0
    maxval3 = data.shape[2] + 1.0


    matdim1 = torch.tensor(range(data.shape[0])) + torch.tensor(range(data.shape[0]))/maxval1
    matdim1 = matdim1.expand(data.shape[1], data.shape[2], data.shape[0])

    matdim2 = torch.tensor(range(data.shape[1])) + torch.tensor(range(data.shape[1]))/(maxval1*maxval2)
    matdim2 = matdim2.expand(data.shape[0], data.shape[2], data.shape[1])

    matdim3 = torch.tensor(range(data.shape[2])) + torch.tensor(range(data.shape[2]))/(maxval1*maxval2*maxval3)
    matdim3 = matdim3.expand(data.shape[0], data.shape[1], data.shape[2])


    matdim = matdim1.permute(2, 0, 1) + matdim2.permute(0, 2, 1) + matdim3

    
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


def ZIdx2Int(inputmat):
    mat = inputmat.view(-1)
    idxpos = torch.tensor(range(mat.numel() ))

    packmat = torch.cat((mat.unsqueeze(-1), idxpos.unsqueeze(-1)), dim=-1)
    packmat = packmat[packmat[:, 0].sort()[1]]
    
    x = torch.cat((packmat, idxpos.unsqueeze(-1)), dim=-1)[:, 1:]
    intZidx = (x[x[:, 0].sort()[1]])[:, 1].squeeze()

    intZidx = intZidx.reshape(inputmat.shape)

    return intZidx




class  Conv2d_FR1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, minrate=0.1, droprate=0.001, dropspeed=-1, dynamicdrop=False):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size).normal_(0, 0.1))
        self.bias = nn.Parameter(torch.empty(out_channels).normal_(0, 0.1))

        self.stride = stride
        self.padding = padding
        self.ifbias = bias

        self.register_buffer("ZMAT",    ZIdx2Int(getZIdx_2d(torch.empty(1, kernel_size)).squeeze()))
        self.register_buffer("IMAT",    torch.zeros(kernel_size) + 1.0)
        self.register_buffer("IDROP",   torch.zeros(kernel_size) + 1.0)

        self.register_buffer("BMAT",  torch.zeros(out_channels) + 1.0)


        self.minrate = minrate
        self.minnum = max(round(kernel_size*minrate), 3)
        
        
        self.dynamicdrop = dynamicdrop
        self.dropcnt = self.ZMAT.numel()

        if dropspeed > 0:
            self.dropspeed = dropspeed
        else:
            self.dropspeed = droprate*self.ZMAT.numel()


        self.weightrate = 0
        self.weightnum = -1

        self.biasrate = 0
        self.biasnum = -1


    def forward(self, data):

        if self.dropcnt <= self.minnum:
            self.dropcnt = self.IDROP.numel()

            if self.dynamicdrop:
                self.IDROP = self.IMAT.clone()
        else:
            idx = ~(self.ZMAT < self.dropcnt)
            self.IDROP[idx] = 0

            self.weightnum = self.IDROP.sum().item()
            self.weightrate = (self.IDROP.sum()/self.IDROP.numel()).item()

            self.dropcnt = self.dropcnt - self.dropspeed


        dropweight = self.weight[:, :, :]*self.IDROP
        weight = idct(dropweight)

        if self.ifbias:
            x = F.conv2d(data, weight, self.bias, stride=self.stride, padding=self.padding)
        else:
            x = F.conv2d(data, weight, stride=self.stride, padding=self.padding)

        return x



class  Conv2d_FR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, minrate=0.1, droprate=0.001, dropspeed=-1, dynamicdrop=False):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size).normal_(0, 0.1))
        self.bias = nn.Parameter(torch.empty(out_channels).normal_(0, 0.1))

        self.stride = stride
        self.padding = padding
        self.ifbias = bias

        self.register_buffer("ZMAT",    ZIdx2Int(getZIdx_2d(torch.empty(kernel_size, kernel_size))))
        self.register_buffer("IMAT",    torch.zeros(kernel_size, kernel_size) + 1.0)
        self.register_buffer("IDROP",   torch.zeros(kernel_size, kernel_size) + 1.0)

        self.register_buffer("BMAT",  torch.zeros(out_channels) + 1.0)


        self.minrate = minrate
        self.minnum = max(round(kernel_size*kernel_size*minrate), 4)
        
        
        self.dynamicdrop = dynamicdrop
        self.dropcnt = self.ZMAT.numel()

        if dropspeed > 0:
            self.dropspeed = dropspeed
        else:
            self.dropspeed = droprate*self.ZMAT.numel()


        self.weightrate = 0
        self.weightnum = -1

        self.biasrate = 0
        self.biasnum = -1


    def forward(self, data):

        if self.dropcnt <= self.minnum:
            self.dropcnt = self.IDROP.numel()

            if self.dynamicdrop:
                self.IDROP = self.IMAT.clone()
        else:
            idx = ~(self.ZMAT < self.dropcnt)
            self.IDROP[idx] = 0

            self.weightnum = self.IDROP.sum().item()
            self.weightrate = (self.IDROP.sum()/self.IDROP.numel()).item()

            self.dropcnt = self.dropcnt - self.dropspeed


        dropweight = self.weight[:, :]*self.IDROP
        weight = idct_2d(dropweight)

        if self.ifbias:
            x = F.conv2d(data, weight, self.bias, stride=self.stride, padding=self.padding)
        else:
            x = F.conv2d(data, weight, stride=self.stride, padding=self.padding)

        return x






class  Conv2d_FR3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, minrate=0.1, droprate=0.001, dropspeed=-1, dynamicdrop=False):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size).normal_(0, 0.1))
        self.bias = nn.Parameter(torch.empty(out_channels).normal_(0, 0.1))

        self.stride = stride
        self.padding = padding
        self.ifbias = bias

        self.register_buffer("ZMAT",    ZIdx2Int(getZIdx_3d(torch.empty(in_channels, kernel_size, kernel_size))))
        self.register_buffer("IMAT",    torch.zeros(in_channels, kernel_size, kernel_size) + 1.0)
        self.register_buffer("IDROP",   torch.zeros(in_channels, kernel_size, kernel_size) + 1.0)

        self.register_buffer("BMAT",  torch.zeros(out_channels) + 1.0)


        self.minrate = minrate
        self.minnum = max(round(in_channels*kernel_size*kernel_size*minrate), 8)
        
        
        self.dynamicdrop = dynamicdrop
        self.dropcnt = self.ZMAT.numel()

        if dropspeed > 0:
            self.dropspeed = dropspeed
        else:
            self.dropspeed = droprate*self.ZMAT.numel()


        self.weightrate = 0
        self.weightnum = -1

        self.biasrate = 0
        self.biasnum = -1


    def forward(self, data):

        if self.dropcnt <= self.minnum:
            self.dropcnt = self.IDROP.numel()

            if self.dynamicdrop:
                self.IDROP = self.IMAT.clone()
        else:
            idx = ~(self.ZMAT < self.dropcnt)
            self.IDROP[idx] = 0

            self.weightnum = self.IDROP.sum().item()
            self.weightrate = (self.IDROP.sum()/self.IDROP.numel()).item()

            self.dropcnt = self.dropcnt - self.dropspeed


        dropweight = self.weight[:]*self.IDROP
        weight = idct_3d(dropweight)

        if self.ifbias:
            x = F.conv2d(data, weight, self.bias, stride=self.stride, padding=self.padding)
        else:
            x = F.conv2d(data, weight, stride=self.stride, padding=self.padding)

        return x








class  Conv2d_FR4d_org(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, minrate=0.1, droprate=0.001, dropspeed=-1, dynamicdrop=False, groups=1, directdrop=False):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels//groups, kernel_size, kernel_size).normal_(0, 0.1))
        self.bias = nn.Parameter(torch.empty(out_channels).normal_(0, 0.1))

        self.stride = stride
        self.padding = padding
        self.ifbias = bias 
        self.groups = groups

        self.register_buffer("ZMAT",    ZIdx2Int(getZIdx_4d(torch.empty(out_channels, in_channels//groups, kernel_size, kernel_size))))
        self.register_buffer("IMAT",    torch.zeros(out_channels, in_channels//groups, kernel_size, kernel_size) + 1.0)
        self.register_buffer("IDROP",  torch.zeros(out_channels, in_channels//groups, kernel_size, kernel_size) + 1.0)

        self.register_buffer("BMAT",  torch.zeros(out_channels) + 1.0)


        self.minrate = minrate
        self.minnum = max(round(out_channels*in_channels*kernel_size*kernel_size*minrate//groups), 511)
        
        self.droprate = droprate 
        self.dynamicdrop = dynamicdrop
        self.dropcnt = self.ZMAT.numel()

        if dropspeed > 0:
            self.dropspeed = dropspeed
        else:
            self.dropspeed = droprate*self.ZMAT.numel()


        self.weightrate = 0
        self.weightnum = -1

        self.biasrate = 0
        self.biasnum = -1


        if directdrop:
            self.dropcnt = self.minnum + 1

    def reset(self):
        self.minnum = max(round(self.weight.numel()*self.minrate//self.groups), 511)
        self.IDROP.fill_(1.0)
        self.dropcnt = self.IDROP.numel()
        self.dropspeed = self.droprate*self.ZMAT.numel()

    def setminrate(self):
        self.minnum = max(round(self.weight.numel()*self.minrate//self.groups), 511)
        self.IDROP.fill_(1.0)
        self.dropcnt = self.minnum + 10


    def forward(self, data):

        if self.dropcnt <= self.minnum:
            if self.dynamicdrop:
                self.IDROP = self.IMAT.clone()
                self.dropcnt = self.IDROP.numel()
                self.dropspeed = self.droprate*self.ZMAT.numel()
        else:
            idx = ~(self.ZMAT < self.dropcnt)
            self.IDROP[idx] = 0



            self.dropcnt = self.dropcnt - self.dropspeed

            if self.dropcnt < self.minnum:
                self.dropcnt = self.dropcnt + self.dropspeed
                #self.dropspeed = max(1, self.dropspeed//10)
                self.dropspeed = 1


        # tracking the drop rate and drop num
        self.weightnum = self.IDROP.sum().item()
        self.weightrate = (self.IDROP.sum()/self.IDROP.numel()).item()


        dropweight = self.weight*self.IDROP
        weight = idct_4d(dropweight)

        if self.ifbias:
            rate = (self.IDROP.sum()/self.IDROP.numel())
            pos = max(round(rate.item()*self.bias.shape[0]), 511)
            BMAT = self.BMAT.clone() 
            BMAT[pos:] = 0

            
            self.biasrate = rate.item()
            self.biasnum = BMAT.sum().item()


            dropbias = self.bias*BMAT
            bias = idct(dropbias)

            x = F.conv2d(data, weight, bias, stride=self.stride, padding=self.padding, groups=self.groups)
        else:
            x = F.conv2d(data, weight, stride=self.stride, padding=self.padding, groups=self.groups)

        return x








class Linear_FR1d(nn.Module):
    def __init__(self, in_features, out_features, bias=True, minrate=0.1, droprate=0.001, dropspeed=-1, dynamicdrop=False):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.1))
        self.bias = nn.Parameter(torch.empty(out_features).normal_(0, 0.1))

        self.ifbias = bias


        self.register_buffer("ZMAT",    ZIdx2Int(getZIdx_2d(torch.empty(1, in_features)).squeeze()))
        self.register_buffer("IMAT",    torch.zeros(1, in_features).squeeze() + 1.0)
        self.register_buffer("IDROP",   torch.zeros(1, in_features).squeeze() + 1.0)
        self.register_buffer("BMAT",    torch.zeros(out_features) + 1.0)


        self.minrate = minrate
        self.minnum = max(round(self.IMAT.numel()*minrate), 16)
        
        
        self.dynamicdrop = dynamicdrop
        self.dropcnt = self.ZMAT.numel()

        if dropspeed > 0:
            self.dropspeed = dropspeed
        else:
            self.dropspeed = droprate*self.ZMAT.numel()
        

        self.weightrate = 0
        self.weightnum = -1

        self.biasrate = 0
        self.biasnum = -1

    def forward(self, data):

        if self.dropcnt <= self.minnum:
            if self.dynamicdrop:
                self.IDROP = self.IMAT.clone()
                self.dropcnt = self.IDROP.numel()
        else:
            idx = ~(self.ZMAT < self.dropcnt)
            self.IDROP[idx] = 0

            self.weightnum = self.IDROP.sum().item()
            self.weightrate = (self.IDROP.sum()/self.IDROP.numel()).item()

            self.dropcnt = self.dropcnt - self.dropspeed



        dropweight = self.weight[:]*self.IDROP
        weight = idct(dropweight)


        if self.ifbias:
            output = F.linear(data, weight, self.bias)
        else:
            output = F.linear(data, weight)


        return output


class Linear_FR2d_org(nn.Module):
    def __init__(self, in_features, out_features, bias=True, minrate=0.1, droprate=0.001, dropspeed=-1, dynamicdrop=False, directdrop=False):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.1))
        self.bias = nn.Parameter(torch.empty(out_features).normal_(0, 0.1))

        self.ifbias = bias


        self.register_buffer("ZMAT",    ZIdx2Int(getZIdx_2d(torch.empty(out_features, in_features)).squeeze()))
        self.register_buffer("IMAT",    torch.zeros(out_features, in_features).squeeze() + 1.0)
        self.register_buffer("IDROP",   torch.zeros(out_features, in_features).squeeze() + 1.0)
        self.register_buffer("BMAT",    torch.zeros(out_features) + 1.0)

        self.scale = nn.Parameter(torch.empty(1,1).fill_(1.0).squeeze())

        self.minrate = minrate
        self.minnum = max(round(self.IMAT.numel()*minrate), 511)
        
        
        self.dynamicdrop = dynamicdrop
        self.dropcnt = self.ZMAT.numel()
        self.droprate = droprate

        if dropspeed > 0:
            self.dropspeed = dropspeed
        else:
            self.dropspeed = droprate*self.ZMAT.numel()
       

        self.weightrate = 0
        self.weightnum = -1

        self.biasrate = 0
        self.biasnum = -1


        if directdrop:
            self.dropcnt = self.minnum + 1


        self.sflag = 100


    def reset(self):
        self.minnum = max(round(self.weight.numel()*self.minrate), 511)
        self.IDROP.fill_(1.0)
        self.dropcnt = self.IDROP.numel()
        self.dropspeed = self.droprate*self.ZMAT.numel()


    def forward(self, data):

        if self.dropcnt <= self.minnum:
            if self.dynamicdrop:
                self.IDROP = self.IMAT.clone()
                self.dropcnt = self.IDROP.numel()
                self.dropspeed = self.droprate*self.ZMAT.numel()
        else:
            idx = ~(self.ZMAT < self.dropcnt)
            self.IDROP[idx] = 0

            self.dropcnt = self.dropcnt - self.dropspeed

            if self.dropcnt < self.minnum:
                self.dropcnt = self.dropcnt + self.dropspeed
                self.dropspeed = 1


        # tracking the drop rate and drop num
        self.weightnum = self.IDROP.sum().item()
        self.weightrate = (self.IDROP.sum()/self.IDROP.numel()).item()


#         #dropweight = F.tanh(self.weight)
#         dropweight = F.tanh(self.weight)
#        
#         #self.sflag=10
#         if self.sflag > 0:
#             #dropweight = F.tanhshrink(dropweight*512.0)*(1/512.0)
#             self.sflag = self.sflag - 1
#             dropweight = dropweight*512.0
#             dropweight = dropweight - torch.sin(dropweight * 2 * 3.14159) / (2 * 3.14159)
#             dropweight = dropweight*(1/512.0)
#         else:
#             dropweight = torch.round(512.0*dropweight)*(1/512.0)
#             self.sflag = 4

        dropweight = self.weight*self.IDROP
        #dropweight = self.scale*dropweight*self.IDROP
        weight = idct_2d(dropweight)


        if self.ifbias:

            rate = (self.IDROP.sum()/self.IDROP.numel())
            pos = max(round(rate.item()*self.bias.shape[0]), 511)
            BMAT = self.BMAT.clone() 
            BMAT[pos:] = 0


            self.biasrate = rate.item()
            self.biasnum = BMAT.sum().item()


            dropbias = self.bias*BMAT
            bias = idct(dropbias)

            output = F.linear(data, weight, self.bias)
        else:
            output = F.linear(data, weight)


        return output




#################################################################################
# borderline                                                                    #
#################################################################################

class  Conv2d_FR4d(nn.Module):
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



        self.register_buffer("ZMAT",    ZIdx2Int(getZIdx_4d(torch.empty(out_channels, in_channels//groups, kernel_size, kernel_size))))
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
        weight = idct_4d(dropweight)

        output = F.conv2d(data, weight, self.bias, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)

        return output


class Linear_FR2d(nn.Module):
    def __init__(self, in_features, out_features, bias=True, minrate=0.01, droprate=0.01, dynamicdrop=False):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.1))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features).normal_(0, 0.1))
        else:
            self.bias = None

        self.register_buffer("ZMAT",    ZIdx2Int(getZIdx_2d(torch.empty(out_features, in_features)).squeeze()))
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
        weight = idct_2d(dropweight)

        output = F.linear(data, weight, self.bias)


        return output



class Conv2d(Conv2d_FR4d):
    pass

class Linear(Linear_FR2d):
    pass



def countParams(model):
    sumnum = 0
    maxnum = -1
    minnum = 10000000000000000
    rate = 0
    totalnum = 0

    for name, layer in model.named_modules():
        try:
            num = layer.IDROP.sum()
            rate = num/layer.IDROP.numel()
            totalnum = totalnum + layer.IDROP.numel()

            if maxnum < num:
                maxnum = num
            
            if minnum > num:
                minnum = num
            
            sumnum += num
        except:
            pass

    rate = sumnum*1.0/totalnum

    return sumnum, minnum, maxnum, rate


def setMinrate(model, minrate=0.01):
    for name, layer in model.named_modules():
        try:
            layer.setminrate(minrate)
        except:
            pass


    return None

def setDroprate(model, droprate):
    for name, layer in model.named_modules():
        try:
            layer.setDroprate(droprate)
        except:
            pass

    return None

def resetParams(model):
    for name, layer in model.named_modules():
        try:
            layer.resetparams()
        except:
            pass

    return None





class  ConvTranspose2d_FR4d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, minrate=0.1, droprate=0.001, dropspeed=-1, dynamicdrop=False, groups=1, directdrop=False):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(in_channels, out_channels//groups, kernel_size, kernel_size).normal_(0, 0.1))
        self.bias = nn.Parameter(torch.empty(out_channels).normal_(0, 0.1))

        self.stride = stride
        self.padding = padding
        self.ifbias = bias 
        self.groups = groups

        self.register_buffer("ZMAT",    ZIdx2Int(getZIdx_4d(torch.empty(in_channels, out_channels//groups, kernel_size, kernel_size))))
        self.register_buffer("IMAT",    torch.zeros(in_channels, out_channels//groups, kernel_size, kernel_size) + 1.0)
        self.register_buffer("IDROP",  torch.zeros(in_channels, out_channels//groups, kernel_size, kernel_size) + 1.0)

        self.register_buffer("BMAT",  torch.zeros(out_channels) + 1.0)


        self.minrate = minrate
        self.minnum = max(round(out_channels*in_channels*kernel_size*kernel_size*minrate//groups), 31)
        
        
        self.dynamicdrop = dynamicdrop
        self.dropcnt = self.ZMAT.numel()

        if dropspeed > 0:
            self.dropspeed = dropspeed
        else:
            self.dropspeed = droprate*self.ZMAT.numel()


        self.weightrate = 0
        self.weightnum = -1

        self.biasrate = 0
        self.biasnum = -1


        if directdrop:
            self.dropcnt = self.minnum + 1

    def reset(self):
        self.minnum = max(round(self.weight.numel()*self.minrate//self.groups), 16)
        self.IDROP.fill_(1.0)
        self.dropcnt = self.minnum + 10


    def forward(self, data):

        if self.dropcnt <= self.minnum:
            if self.dynamicdrop:
                self.IDROP = self.IMAT.clone()
                self.dropcnt = self.IDROP.numel()
        else:
            idx = ~(self.ZMAT < self.dropcnt)
            self.IDROP[idx] = 0



            self.dropcnt = self.dropcnt - self.dropspeed

            if self.dropcnt < self.minnum:
                self.dropcnt = self.dropcnt + self.dropspeed
                #self.dropspeed = max(1, self.dropspeed//10)
                self.dropspeed = 1


        # tracking the drop rate and drop num
        self.weightnum = self.IDROP.sum().item()
        self.weightrate = (self.IDROP.sum()/self.IDROP.numel()).item()


        dropweight = self.weight*self.IDROP
        weight = idct_4d(dropweight)

        if self.ifbias:
            rate = (self.IDROP.sum()/self.IDROP.numel())
            pos = max(round(rate.item()*self.bias.shape[0]), 15)
            BMAT = self.BMAT.clone() 
            BMAT[pos:] = 0

            
            self.biasrate = rate
            self.biasnum = BMAT.sum()


            dropbias = self.bias*BMAT
            bias = idct(dropbias)

#            x = F.conv2d(data, weight, bias, stride=self.stride, padding=self.padding, groups=self.groups)
            x = F.conv_transpose2d(data, weight, bias, stride=self.stride, padding=self.padding, groups=self.groups)

        else:
#            x = F.conv2d(data, weight, stride=self.stride, padding=self.padding, groups=self.groups)
            x = F.conv_transpose2d(data, weight, stride=self.stride, padding=self.padding, groups=self.groups)


        return x


#             x = F.conv_transpose2d(data, weight, bias, stride=self.stride, padding=self.padding)
# 
#             
#         else:
#             x = F.conv_transpose2d(data, weight, stride=self.stride, padding=self.padding)




# class ConvTranspose2d_FR3d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, minrate=0.01, dropspeed=0.01, fixrate=-1.0):
#         super().__init__()
# 
#         self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size).normal_(0, 0.1))
#         self.bias = nn.Parameter(torch.empty(out_channels).normal_(0, 0.1))
# 
#         self.stride = stride
#         self.padding = padding
#         self.ifbias = bias
# 
#         self.register_buffer("ZMAT",    getZIdx_3d(torch.empty(in_channels, kernel_size, kernel_size)))
#         self.register_buffer("ZTRACK",  getZIdx_3d(torch.empty(in_channels, kernel_size, kernel_size)))
#         self.register_buffer("IMAT",    torch.zeros(in_channels, kernel_size, kernel_size) + 1.0)
#         self.register_buffer("theval",  self.ZMAT.max())
#        
#         self.register_buffer("B_IMAT",  torch.zeros(out_channels) + 1.0)
#         
# 
#         self.minrate = minrate
#         
#         self.keeprate = 1.0
#         self.dropspeed = dropspeed
#         self.fixrate = fixrate
# 
#     def forward(self, data):
# 
#         if self.fixrate > 0:
#             self.keeprate = self.fixrate
# 
#         IMAT = self.IMAT.clone()
# 
#         theval = self.ZTRACK.max()
#         idx = ~(self.ZMAT < theval)
# 
#         IMAT[idx] = 0
# 
# 
#         if IMAT.sum()/IMAT.numel() > self.keeprate:
#             self.ZTRACK[idx] = 0
#         else:
#             self.keeprate = self.keeprate - self.dropspeed
#         
#          
#         if self.keeprate <= self.minrate:
#             self.keeprate = 1.0
#             self.ZTRACK = self.ZMAT.clone()
# 
# 
# 
#         dropweight = self.weight[:]*IMAT
# 
#         weight = idct_3d(dropweight)
# 
# 
#         if self.ifbias:
#             
#             pos = round(self.keeprate*self.bias.shape[0])
#             B_IMAT = self.B_IMAT.clone() 
#             B_IMAT[pos:] = 0
# 
#             dropbias = self.bias*B_IMAT
# 
#             bias = idct(dropbias)
# 
#             x = F.conv_transpose2d(data, weight, bias, stride=self.stride, padding=self.padding)
# 
#             
#         else:
#             x = F.conv_transpose2d(data, weight, stride=self.stride, padding=self.padding)
# 
# 
#         return x


class ConvTranspose2d(ConvTranspose2d_FR4d):
    pass
