# PREF as an encoder https://arxiv.org/pdf/2205.13524.pdf
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

class PREF(nn.Module):
    def __init__(self, linear_freqs, reduced_freqs=[1]*3, feature_dim=16, sampling='linear', device='cuda', **kwargs) -> None:
        """
        Notice that a 3D phasor volume is viewed as  2D full specturm and 1D reduced specturm.
        Args: 
            linear_freqs: number of 2D freqeuncies 
            reduced_freqs: number of 1D frequencies 
            sampling: linear or explonential increasing
            feature_dim: output dimension
        """
        super().__init__()
        self.device = device
        self.linear_res = torch.tensor(linear_freqs).to(self.device)
        self.reduced_res = torch.tensor(reduced_freqs).to(self.device)
        
        if sampling == 'linear':
            self.axis = [torch.tensor([0.]+[i+1 for i in torch.arange(d-1)]).to(self.device) for d in reduced_freqs]
        else:
            self.axis = [torch.tensor([0.]+[2**i for i in torch.arange(d-1)]).to(self.device) for d in reduced_freqs]
        
        self.ktraj = self.compute_ktraj(self.axis, self.linear_res)

        self.output_dim = feature_dim
        self.alpha_params = nn.Parameter(torch.tensor([1e-3]).to(self.device))
        self.params = nn.ParameterList(
            self.init_phasor_volume()
            )
        print(self)

    @property
    def alpha(self):
        # adaptively adjust the scale of phasors' magnitude during optimization.
        # not so important when Parsvel loss is imposed.
        return F.softplus(self.alpha_params, beta=10, threshold=1)

    @property
    def phasor(self):
        feature = [feat * self.alpha for feat in self.params]
        return feature

    def forward(self, inputs, bound=1):
        # map to [-1, 1]
        inputs = inputs / bound 
        # obtain embedding from phasor volume
        feature = self.compute_fft(self.phasor, inputs, interp=False)
        return feature.T

    # naive impl of inverse fourier transform
    def compute_spatial_volume(self, features):
        xx, yy, zz = [torch.linspace(0, 1, N).to(self.device) for N in self.res]
        Fx, Fy, Fz = features
        Nx, Ny, Nz = Fy.shape[2], Fz.shape[3], Fx.shape[4]
        d1, d2, d3 = Fx.shape[2], Fy.shape[3], Fz.shape[4]
        kx, ky, kz = self.axis
        kx, ky, kz = kx[:d1], ky[:d2], kz[:d3]
        fx = irfft(torch.fft.ifftn(Fx, dim=(3,4), norm='forward'), xx, ff=kx, T=Nx, dim=2)
        fy = irfft(torch.fft.ifftn(Fy, dim=(2,4), norm='forward'), yy, ff=ky, T=Ny, dim=3)
        fz = irfft(torch.fft.ifftn(Fz, dim=(2,3), norm='forward'), zz, ff=kz, T=Nz, dim=4)
        return (fx, fy, fz)

    # approx IFT as depicted in Eq.5 https://arxiv.org/pdf/2205.13524.pdf
    def compute_fft(self, features, xyz_sampled, interp=True):
        if interp:
            # using interpolation to compute fft = (N*N) log (N) d  + (N*N*d*d) + Nsamples 
            fx, fy, fz = self.compute_spatial_volume(features)
            volume = fx+fy+fz
            points = F.grid_sample(volume, xyz_sampled[None, None, None].flip(-1), align_corners=True).view(-1, *xyz_sampled.shape[:1],)
            # this is somewhat expensive when the xyz_samples is few and a 3D volume stills need computed
        else:
            # this is fast because we did 2d transform and matrix multiplication . (N*N) logN d + Nsamples * d*d + 3 * Nsamples 
            Nx, Ny, Nz = self.linear_res
            Fx, Fy, Fz = features
            d1, d2, d3 = Fx.shape[2], Fy.shape[3], Fz.shape[4]
            kx, ky, kz = self.axis
            kx, ky, kz = kx[:d1], ky[:d2], kz[:d3]
            xs, ys, zs = xyz_sampled.chunk(3, dim=-1)
            Fx = torch.fft.ifftn(Fx, dim=(3,4), norm='forward')
            Fy = torch.fft.ifftn(Fy, dim=(2,4), norm='forward')
            Fz = torch.fft.ifftn(Fz, dim=(2,3), norm='forward')
            fx = grid_sample_cmplx(Fx.transpose(3,3).flatten(1,2), torch.stack([zs, ys], dim=-1)[None]).reshape(Fx.shape[1], Fx.shape[2], -1)
            fy = grid_sample_cmplx(Fy.transpose(2,3).flatten(1,2), torch.stack([zs, xs], dim=-1)[None]).reshape(Fy.shape[1], Fy.shape[3], -1)
            fz = grid_sample_cmplx(Fz.transpose(2,4).flatten(1,2), torch.stack([xs, ys], dim=-1)[None]).reshape(Fz.shape[1], Fz.shape[4], -1)
            fxx = batch_irfft(fx, xs, kx, Nx)
            fyy = batch_irfft(fy, ys, ky, Ny)
            fzz = batch_irfft(fz, zs, kz, Nz)
            return fxx+fyy+fzz

        return points

    @torch.no_grad()
    def init_phasor_volume(self):
        # rough approximation 
        # transform the fourier domain to spatial domain
        Nx, Ny, Nz = self.linear_res
        d1, d2, d3 = self.reduced_res
        # xx, yy, zz = [torch.linspace(0, 1, N).to(self.device) for N in (d1,d2,d3)]
        xx, yy, zz = [torch.linspace(0, 1, N).to(self.device) for N in (Nx,Ny,Nz)]
        XX, YY, ZZ = [torch.linspace(0, 1, N).to(self.device) for N in (Nx,Ny,Nz)]
        kx, ky, kz = self.axis
        kx, ky, kz = kx[:d1], ky[:d2], kz[:d3]
        
        fx = torch.ones(1, self.output_dim, len(xx), Ny, Nz).to(self.device)
        fy = torch.ones(1, self.output_dim, Nx, len(yy), Nz).to(self.device)
        fz = torch.ones(1, self.output_dim, Nx, Ny, len(zz)).to(self.device)
        normx = torch.stack(torch.meshgrid([2*xx-1, 2*YY-1, 2*ZZ-1]), dim=-1).norm(dim=-1)
        normy = torch.stack(torch.meshgrid([2*XX-1, 2*yy-1, 2*ZZ-1]), dim=-1).norm(dim=-1)
        normz = torch.stack(torch.meshgrid([2*XX-1, 2*YY-1, 2*zz-1]), dim=-1).norm(dim=-1)

        fx = fx * normx[None, None] / (3 * self.alpha * np.sqrt(self.output_dim))
        fy = fy * normy[None, None] / (3 * self.alpha * np.sqrt(self.output_dim))
        fz = fz * normz[None, None] / (3 * self.alpha * np.sqrt(self.output_dim))

        fxx = rfft(torch.fft.fftn(fx.transpose(2,4), dim=(2,3), norm='forward'),xx, ff=kx, T=Nx).transpose(2,4)
        fyy = rfft(torch.fft.fftn(fy.transpose(3,4), dim=(2,3), norm='forward'),yy, ff=ky, T=Ny).transpose(3,4)
        fzz = rfft(torch.fft.fftn(fz.transpose(4,4), dim=(2,3), norm='forward'),zz, ff=kz, T=Nz).transpose(4,4)
        return [torch.nn.Parameter(fxx), torch.nn.Parameter(fyy), torch.nn.Parameter(fzz)]


    def compute_ktraj(self, axis, res): # the associated frequency coordinates.
        ktraj2d = [torch.fft.fftfreq(i, 1/i).to(self.device) for i in res]
        ktraj1d = [torch.arange(ax).to(torch.float).to(self.device) if type(ax) == int else ax for ax in axis]
        ktrajx = torch.stack(torch.meshgrid([ktraj1d[0], ktraj2d[1], ktraj2d[2]]), dim=-1)
        ktrajy = torch.stack(torch.meshgrid([ktraj2d[0], ktraj1d[1], ktraj2d[2]]), dim=-1)
        ktrajz = torch.stack(torch.meshgrid([ktraj2d[0], ktraj2d[1], ktraj1d[2]]), dim=-1)
        ktraj = [ktrajx, ktrajy, ktrajz]
        return ktraj

    # def parseval_loss(self):
        # Parseval Loss
        # new_feats = [Fk.reshape(-1, *Fk.shape[2:],1) * 1j * np.pi * wk.reshape(1, *Fk.shape[2:], -1) 
            # for Fk, wk in zip(self.phasor, self.ktraj)]
        # loss = sum([feat.abs().square().mean() for feat in itertools.chain(*new_feats)])
        # return loss


## utilis 
def irfft(phasors, xx, ff=None, T=None, dim=-1):
    assert (xx.max() <= 1) & (xx.min() >= 0)
    phasors = phasors.transpose(dim, -1)
    assert phasors.shape[-1] == len(ff) if ff is not None else True
    device = phasors.device
    xx = xx * (T-1) / T                       # to match torch.fft.fft
    N = phasors.shape[-1]
    if ff is None:
        ff = torch.arange(N).to(device)       # positive freq only
    xx = xx.reshape(-1, 1).to(device)    
    M = torch.exp(2j * np.pi * xx * ff).to(device)
    M = M * ((ff>0)+1)[None]                  # Hermittion symmetry
    out = F.linear(phasors.real, M.real) - F.linear(phasors.imag, M.imag)
    out = out.transpose(dim, -1)
    return out


def batch_irfft(phasors, xx, ff, T):
    # numerial integration 
    # phaosrs [dim, d, N] # coords  [N,1] # bandwidth d  # norm x to [0,1]
    xx = (xx+1) * 0.5
    xx = xx * (T-1) / T
    if ff is None:
        ff = torch.arange(phasors.shape[1]).to(xx.device)
    twiddle = torch.exp(2j*np.pi*xx * ff)                   # twiddle factor
    twiddle = twiddle * ((ff > 0)+1)[None]                  # hermitian # [N, d]
    twiddle = twiddle.transpose(0,1)[None]
    return (phasors.real * twiddle.real).sum(1) - (phasors.imag * twiddle.imag).sum(1)

def rfft(spatial, xx, ff=None, T=None, dim=-1):
    assert (xx.max() <= 1) & (xx.min() >= 0)
    spatial = spatial.transpose(dim, -1)
    assert spatial.shape[-1] == len(xx)
    device = spatial.device
    xx = xx * (T-1) / T
    if ff is None:
        ff = torch.fft.rfftfreq(T, 1/T) # positive freq only
    ff = ff.reshape(-1, 1).to(device)
    M = torch.exp(-2j * np.pi * ff * xx).to(device)
    out = F.linear(spatial, M)
    out = out.transpose(dim, -1) / len(xx)
    return out

def grid_sample_cmplx(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True):
    sampled = F.grid_sample(input.real, grid, mode, padding_mode, align_corners) + \
            1j * F.grid_sample(input.imag, grid, mode, padding_mode, align_corners)
    return sampled
