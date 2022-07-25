import torch
import torch.nn
import torch.nn.functional as F
from .sh import eval_sh_bases
import numpy as np
import time
import pdb

def positional_encoding(positions, freqs):
    
        freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma*dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:,-1:]


def SHRender(xyz_sampled, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(xyz_sampled, viewdirs, features):

    rgb = features
    return rgb

class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device
        self.aabb=aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0/self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1,1,*alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1],alpha_volume.shape[-2],alpha_volume.shape[-3]]).to(self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1,-1,1,1,3).flip(-1), align_corners=True).view(-1)
        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invgridSize - 1


class MLPRender_Fea(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea, self).__init__()

        self.in_mlpC = 2*viewpe*3 + 2*feape*inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLPRender_PE(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, pospe=6, featureC=128):
        super(MLPRender_PE, self).__init__()

        self.in_mlpC = (3+2*viewpe*3)+ (3+2*pospe*3)  + inChanel #
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLPRender(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, featureC=128):
        super(MLPRender, self).__init__()

        self.in_mlpC = (3+2*viewpe*3) + inChanel
        self.viewpe = viewpe
        
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

