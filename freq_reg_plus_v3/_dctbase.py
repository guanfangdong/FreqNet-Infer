import numpy as np
import os
import torch

from ._dct import idct, dct1, idct1, dct, dct1_2d, idct1_2d, dct1_4d, dct_2d

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=9999999999999999999)


np.set_printoptions(precision=2, suppress=True)
torch.set_printoptions(precision=2, sci_mode=False)


def dct1Base(N):
    N = N-1

    n = torch.tensor(np.arange(N+1))
    k = n.reshape((N+1, 1))
    
    mat = k * n
    mat = torch.cos(np.pi*mat/N)

    mat[0] *= 0.5
    mat[N] *= 0.5

    return mat


def idct1Base(N):

    mat = dct1Base(N)
    mat = (2*mat)/(N - 1)

    return mat


def dct1Base_gen(N):

    mat = dct1Base(N)

    return mat*2


def idct1Base_gen(N):

    mat = idct1Base(N)

    return mat*0.5



def ifftBase(N):
    n = torch.tensor(np.arange(N))
    k = n.reshape((N, 1))

    temp = k * n

    real_part = torch.cos(2 * np.pi * temp / N)
    imag_part = torch.sin(2 * np.pi * temp / N)

    M = (real_part + 1j * imag_part) / N

    return M



def fftBase(N):

    n = torch.tensor(np.arange(N))
    k = n.reshape((N, 1))

    temp = k * n

    real_part = torch.cos(-2 * np.pi * temp / N)
    imag_part = torch.sin(-2 * np.pi * temp / N)


    M = real_part + 1j * imag_part

    return M




def getTwiFac(a, N):
    
    w = torch.arange(N) * a * 2*np.pi / (N)
    w = torch.exp(w*1j)

    return w


def dctEx(x):

    x = torch.cat([x, x.flip([1])], dim=1)

    return x




def dct2Base(N):

    w = getTwiFac(-0.5, N*2)

    fbase = fftBase(N*2)
    dctbase = fbase * w.unsqueeze(0) * 2
    dctbase = dctbase[:N, :N]

    return dctbase.real


def idct2Base(N):

    iw = getTwiFac(0.5, N*2)

    ifbase = ifftBase(N*2)
    idctbase = ifbase* iw.unsqueeze(1)*2
    idctbase = idctbase[:N, :N]
    idctbase[0] *= 0.5
    
    return idctbase.real






def main_func():
    # 验证dct II 是否正交
    
    x = torch.randn(1, 7)
    N = x.shape[1]


    x_dct = dct(x)


    dctbase = dct2Base(N)
    idctbase = idct2Base(N)


    print(x_dct)
    print((x @ dctbase))

    print(x)
    print(x @ dctbase @ idctbase)



    input("validate the dct base")



    # dct 卷积conv，直接输出dct结果
    x = torch.randn(2, 6)
    y = torch.randn(3, 6)

    tru = x @ y.t()

    tru = dct(tru)



    x_dct = dct(x)
    y_dct = dct_2d(y)


    x_dct[:, 0] *= 0.5

    out_fr = x_dct @ y_dct.t()
    


    ifbase = ifftBase(12)
    out = out_fr*ifbase[0,0].real
    

    print(out)
    print(tru)

    input("check the dct conv")





if __name__ == '__main__':
    main_func()

