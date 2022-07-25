import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import pdb
import copy

# def grid_sample_cmplx(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True):
#             return F.grid_sample(input.real, grid, mode, padding_mode, align_corners) + \
#                     1j * F.grid_sample(input.imag, grid, mode, padding_mode, align_corners)


# def grid_sample_cmplx1d(input, grid):
#     # map to index
#     assert ((grid >= -1) & (grid <=1)).all()
#     k = (input.shape[-1]-1) / 2
#     index = grid.flatten() * k + k
#     lindx = index.floor().long()
#     rindx = index.ceil().long() 
#     lweight = rindx - index  # if index close to right side, left side weight less
#     rweight = torch.max(index - lindx, 1-lweight)  # if index close to left side, right side weight less
#     assert ((lweight + rweight) == 1.).all()
#     sampled = (input[..., lindx] * lweight[None, None] + input[..., rindx] * rweight[None, None])
#     return sampled


# def pad_hermitian(oneside, gridSize):
#     flip = oneside.clone()
#     flip[..., 1:] =  flip[..., 1:].flip(-1)
#     flip[..., 1:,:] = flip[..., 1:, :].flip(-2)
#     flip[..., 1:,:, :] = flip[..., 1:,:,:].flip(-3)
#     flip = flip[..., 1:]
#     pad_feat = torch.concat([oneside, flip.conj()[..., (gridSize[-2]+1)%2:]], dim=-1)
#     return pad_feat

# def batch_rdftn(phasors, coords, bandwidths, Ts):
#     """last dim is the hermitian dim
#     Args:
#         phasors: [dim, bandwidth[0], bandwidth[1], N]
#         coords:  ([N,1,1], [N,1,1],...)
#         bandwidth: (d1, d2, ...)
#         T: Time (Tx, Ty, ...)
#     Returns:
#         transformed: [dim, N]
#     """
#     assert [i for i in phasors.shape[1:-1]] == bandwidths # check number of dimensions
#     assert len(coords) == len(bandwidths)    # check num of dimensions
#     assert phasors.shape[-1] == coords[0].numel() # check the num of nsamples
#     # pdb.set_trace()
#     device = phasors.device
#     coords = [(x+1) * 0.5 * (Tx-1)/Tx for x, Tx in zip(coords, Ts)] # map to coords
#     if len(bandwidths) > 1:
#         pdb.set_trace()
#         freqs = [torch.fft.fftfreq(d, 1/d).to(device) for d in bandwidths[:-1]] + [torch.fft.rfftfreq(bandwidths[-1], 1/bandwidths[-1]).to(device)]
#     else:
#         freqs = [torch.fft.rfftfreq(bandwidths[0], 1/bandwidths[0]).to(device)]
#     ff = torch.stack(torch.meshgrid(*freqs), dim=-1).unsqueeze(0) # [1, d1, d2, ..., len(bandwidth)]
#     xx = torch.stack(coords, dim=-1)
#     twiddle = torch.exp((2j * np.pi * ff * xx).sum(-1))
#     twiddle[..., 1:] = twiddle[..., 1:] * 2
#     twiddle = twiddle.flatten(1).unsqueeze(0).transpose(1,2)
#     phasors = phasors.flatten(1,-2)
#     return (phasors.real * twiddle.real).sum(1) - (phasors.imag * twiddle.imag).sum(1)
    

# def batch_rdft(phasors, coords, bandwidth, T):
#     # phaosrs [dim, d, N] # coords  [N,1] # bandwidth d  # norm x to [0,1]
#     coords = (coords+1) * 0.5
#     coords = coords * (T-1) / T
#     freqs = torch.arange(bandwidth).to(coords.device)
#     twiddle = torch.exp(2j*np.pi*coords * freqs)          # twiddle factor
#     twiddle[:,1:] = twiddle[:, 1:] * 2                    # hermitian # [N, d]
#     twiddle = twiddle.transpose(0,1)[None]
#     return (phasors.real * twiddle.real).sum(1) - (phasors.imag * twiddle.imag).sum(1)

# def batch_dft(phasors, coords, bandwidth, T):
#     # phaosrs [N, d] # coords  [N, 1] # bandwidth d  # norm x to [0,1]
#     coords = (coords+1) * 0.5
#     coords = coords * (T-1) / T
#     freqs = torch.concat([torch.arange(bandwidth),  torch.arange(-bandwidth+1, 0)])
#     twiddle = torch.exp(2j*np.pi*coords * freqs)    # twiddle factor
#     twiddle = twiddle.transpose(0,1)[None]
#     return (phasors * twiddle).sum(1)


def dft(phasors, inputs, T=None,dim=-1):
    # inputs should be in [0,1]                      # F(f(ax)) = 1/|a| P(w/a)
    phasors = phasors.transpose(dim, -1)
    device = phasors.device
    inputs = inputs * (T-1) / T                      # to match torch.fft.fft
    N = phasors.shape[-1]                            # frequency domain scaling
    pf = torch.arange(0, (N+1)//2).to(device)        # positive freq
    nf = torch.arange(-(N-1)//2, 0).to(device)       # negative freq
    fk = torch.concat([pf,nf])                       # sampling frequencies
    inputs = inputs.reshape(-1, 1).to(device)
    M = torch.exp(2j * np.pi * inputs * fk).to(device)
    out = F.linear(phasors, M)                       # integrate phasors
    out = out.transpose(dim, -1)                     # transpose back
    return out

def rdft(phasors, inputs, T=None, dim=-1):
    phasors = phasors.transpose(dim, -1)
    device = phasors.device
    inputs = inputs * (T-1) / T                      # to match torch.fft.fft
    N = phasors.shape[-1]
    pf = torch.arange(N).to(device)                  # positive freq only
    fk  = pf                                         # sampling frequencies
    inputs = inputs.reshape(-1, 1).to(device)    
    M = torch.exp(2j * np.pi * inputs * fk).to(device)
    # index in pytorch is slow
    # M[:, 1:] = M[:, 1:] * 2                          # Hermittion symmetry
    M = M * ((fk>0)+1)[None]
    out = F.linear(phasors.real, M.real) - F.linear(phasors.imag, M.imag)
    out = out.transpose(dim, -1)
    return out

# def rfft(phasors, inputs, )


def rdft2(phasors, inputs, T=None, dim=-1):
    phasors = phasors.transpose(dim,-1)
    device = phasors.device
    inputs = inputs * (T-1) / T                      # to match torch.fft.fft
    N = phasors.shape[-1]
    pf = torch.arange(N).to(device)                  # positive freq only
    fk  = pf                                         # sampling frequencies
    inputs = inputs.reshape(-1, 1).to(device)    
    M = torch.exp(2j * np.pi * inputs * fk).to(device)
    M[:, 1:] = M[:, 1:] * 2                         # Hermittion symmetry
    out = F.linear(phasors.real, M.real) - F.linear(phasors.imag, M.imag)
    out = out.transpose(dim,-1)
    return out


def irdft3d(phasors, gridSize):
    device = phasors.device
    ifft_crop = phasors
    Nx, Ny, Nz = gridSize
    xx, yy, zz = [torch.linspace(0, 1, N).to(device) for N in gridSize]
    ifft_crop =  dft(ifft_crop, xx, Nx, dim=2)
    ifft_crop =  dft(ifft_crop, yy, Ny, dim=3)
    ifft_crop = rdft(ifft_crop, zz, Nz, dim=4)
    return ifft_crop

def idft3d(phasors, gridSize):
    device = phasors.device
    ifft_crop = phasors
    Nx, Ny, Nz = gridSize
    xx, yy, zz = [torch.linspace(0, 1, N).to(device) for N in gridSize]
    ifft_crop = dft(ifft_crop, xx, Nx, dim=2)
    ifft_crop = dft(ifft_crop, yy, Ny, dim=3)
    ifft_crop = dft(ifft_crop, zz, Nz, dim=4)
    return ifft_crop


def time2freq(gridSize):
    ffSize = copy.deepcopy(gridSize)
    ffSize[-1] = ffSize[-1] // 2 + 1
    return ffSize

def freq2time(freqSize):
    gridSize = copy.deepcopy(freqSize)
    gridSize[-1] = freqSize[-1]*2
    return gridSize


def getMask(smallSize, largeSize, transform=True):
    """
        compute rfft mask, assume the last dim is the hermitian dimension
    """
    if transform:
        smallSize = time2freq(smallSize)
        largeSize = time2freq(largeSize)
    
    ph_max = [torch.fft.fftfreq(i, 1/i).max() for i in smallSize[:-1]]
    ph_min = [torch.fft.fftfreq(i, 1/i).min() for i in smallSize[:-1]]
    ph_max.append(torch.fft.rfftfreq((smallSize[-1]-1)*2, 1/((smallSize[-1]-1)*2)).max())
    ph_min.append(torch.fft.rfftfreq((smallSize[-1]-1)*2, 1/((smallSize[-1]-1)*2)).min())
    tg_ff = torch.stack(torch.meshgrid([torch.fft.fftfreq(i, 1/i) for i in largeSize[:-1]]
            +[torch.fft.rfftfreq(((largeSize[-1]-1)*2), 1/((largeSize[-1]-1)*2))]), dim=-1)
    mask = torch.ones(largeSize).to(torch.bool)
    for i in range(len(smallSize)):
        mask &= (tg_ff[..., i] <= ph_max[i]) & (tg_ff[..., i] >= ph_min[i])
    # pdb.set_trace()
    assert np.array(smallSize).prod() == mask.sum()
    return mask 

def getMask_fft(smallSize, largeSize):
    ph_max = [torch.fft.fftfreq(i, 1/i).max() for i in smallSize]
    ph_min = [torch.fft.fftfreq(i, 1/i).min() for i in smallSize]
    tg_ff = torch.stack(torch.meshgrid([torch.fft.fftfreq(i, 1/i) for i in largeSize]))
    mask = torch.ones(largeSize).to(torch.bool)
    for i in range(len(smallSize)):
        mask &= (tg_ff[i] <= ph_max[i]) & (tg_ff[i] >= ph_min[i])
    try:
        assert np.array(smallSize).prod() == mask.sum()
    except:
        pdb.set_trace()
    return mask

