import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import pdb
import copy


def irfft(phasors, xx, ff=None, T=None, dim=-1):
    # assert (xx.max() <= 1) & (xx.min() >= 0)
    phasors = phasors.transpose(dim, -1)
    # assert phasors.shape[-1] == len(ff) if ff is not None else True
    device = phasors.device
    xx = xx * (T-1) / T                       # to match torch.fft.fft
    N = phasors.shape[-1]
    if ff is None:
        ff = torch.arange(N).to(device)       # positive freq only
    xx = xx.reshape(-1, 1).to(device)    
    M = torch.exp(2j * np.pi * xx * ff).to(device)
    # indexing in pytorch is slow
    # M[:, 1:-1] = M[:, 1:-1] * 2                # Hermittion symmetry
    M = M * ((ff>0)+1)[None]
    out = F.linear(phasors.real, M.real) - F.linear(phasors.imag, M.imag)
    out = out.transpose(dim, -1)
    return out

def rfft(spatial, xx, ff=None, T=None, dim=-1):
    # assert (xx.max() <= 1) & (xx.min() >= 0)
    spatial = spatial.transpose(dim, -1)
    # assert spatial.shape[-1] == len(xx)
    device = spatial.device
    xx = xx * (T-1) / T
    if ff is None:
        ff = torch.fft.rfftfreq(T, 1/T) # positive freq only
    ff = ff.reshape(-1, 1).to(device)
    M = torch.exp(-2j * np.pi * ff * xx).to(device)
    out = F.linear(spatial, M)
    out = out.transpose(dim, -1) / len(xx)
    return out


def batch_irfft(phasors, xx, ff, T):
    # phaosrs [dim, d, N] # coords  [N,1] # bandwidth d  # norm x to [0,1]
    xx = (xx+1) * 0.5
    xx = xx * (T-1) / T
    if ff is None:
        ff = torch.arange(phasors.shape[1]).to(xx.device)
    twiddle = torch.exp(2j*np.pi*xx * ff)                   # twiddle factor
    # twiddle[:,1:-1] = twiddle[:, 1:-1] * 2                    # hermitian # [N, d]
    twiddle = twiddle  * ((ff>0)+1)[None]
    twiddle = twiddle.transpose(0,1)[None]
    return (phasors.real * twiddle.real).sum(1) - (phasors.imag * twiddle.imag).sum(1)


def getMask_fft(smallSize, largeSize):
    ph_max = [torch.fft.fftfreq(i, 1/i).max() for i in smallSize]
    ph_min = [torch.fft.fftfreq(i, 1/i).min() for i in smallSize]
    tg_ff = torch.stack(torch.meshgrid([torch.fft.fftfreq(i, 1/i) for i in largeSize]))
    mask = torch.ones(largeSize).to(torch.bool)
    for i in range(len(smallSize)):
        mask &= (tg_ff[i] <= ph_max[i]) & (tg_ff[i] >= ph_min[i])
    assert np.array(smallSize).prod() == mask.sum()
    return mask


def grid_sample_cmplx(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True):
            return F.grid_sample(input.real, grid, mode, padding_mode, align_corners) + \
                    1j * F.grid_sample(input.imag, grid, mode, padding_mode, align_corners)