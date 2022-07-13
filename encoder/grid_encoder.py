import torch
import torch.nn as nn
import time
import pdb
import numpy as np
import torch.nn.functional as F

class GRID(nn.Module):
    def __init__(self, resolutions, feature_dim, device='cuda', **kwargs):
        super(GRID, self).__init__()
        self.device = device
        self.res = resolutions
        self.dim = feature_dim
        self.init_volume()
        self.mask = torch.ones(1,1, *self.res).to(device)
        self.output_dim = feature_dim
        print(self)

    def init_volume(self):
        xx, yy, zz = [torch.linspace(-1, 1, N) for N in self.res]
        points = torch.stack(torch.meshgrid(xx,yy,zz),dim=-1).norm(dim=-1)
        volume = torch.ones([1,self.dim]+self.res)
        volume = volume * points.reshape(1,-1,*points.shape) / np.sqrt(self.dim)
        self.params = nn.ParameterList(
            [nn.Parameter(volume.to(self.device))]
            )
        assert volume.norm(dim=1).max() <= np.sqrt(3) + 1e-3

    # def mask_nonsee(self, seen_points, maskval=100):
        # return 
        # kernel_size=1
        # grid = ((seen_points.cpu().numpy()+1)/2 * np.array(self.res)[None]).astype(np.int)
        # self.mask = torch.zeros_like(self.mask)
        # self.mask[:,:, grid[:, 0], grid[:, 1], grid[:,2]] = 1
        # self.mask = F.max_pool3d(self.mask, kernel_size=kernel_size, padding=kernel_size // 2, stride=1).to(self.params.device).to(torch.bool)
        
    def tv_loss(self):
        volume = self.params[0]
        dx = volume.diff(2).abs().mean()
        dy = volume.diff(3).abs().mean()
        dz = volume.diff(4).abs().mean()
        return (dx + dy + dz)

    def forward(self, inputs, variance=0, bound=1):
        inputs = inputs / bound # map to [-1, 1]
        feat = F.grid_sample(self.params[0], inputs[None,None,None].flip(-1), padding_mode='zeros', align_corners=True).reshape(self.dim, -1)
        return feat.T


# def grid_sample_3d(image, optical):
#     N, C, ID, IH, IW = image.shape
#     _, D, H, W, _ = optical.shape

#     ix = optical[..., 0]
#     iy = optical[..., 1]
#     iz = optical[..., 2]

#     ix = ((ix + 1) / 2) * (IW - 1);
#     iy = ((iy + 1) / 2) * (IH - 1);
#     iz = ((iz + 1) / 2) * (ID - 1);
#     with torch.no_grad():
        
#         ix_tnw = torch.floor(ix);
#         iy_tnw = torch.floor(iy);
#         iz_tnw = torch.floor(iz);

#         ix_tne = ix_tnw + 1;
#         iy_tne = iy_tnw;
#         iz_tne = iz_tnw;

#         ix_tsw = ix_tnw;
#         iy_tsw = iy_tnw + 1;
#         iz_tsw = iz_tnw;

#         ix_tse = ix_tnw + 1;
#         iy_tse = iy_tnw + 1;
#         iz_tse = iz_tnw;

#         ix_bnw = ix_tnw;
#         iy_bnw = iy_tnw;
#         iz_bnw = iz_tnw + 1;

#         ix_bne = ix_tnw + 1;
#         iy_bne = iy_tnw;
#         iz_bne = iz_tnw + 1;

#         ix_bsw = ix_tnw;
#         iy_bsw = iy_tnw + 1;
#         iz_bsw = iz_tnw + 1;

#         ix_bse = ix_tnw + 1;
#         iy_bse = iy_tnw + 1;
#         iz_bse = iz_tnw + 1;

#     tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
#     tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
#     tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
#     tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
#     bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
#     bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
#     bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
#     bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);


#     with torch.no_grad():

#         torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
#         torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
#         torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

#         torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
#         torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
#         torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

#         torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
#         torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
#         torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

#         torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
#         torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
#         torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

#         torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
#         torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
#         torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

#         torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
#         torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
#         torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

#         torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
#         torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
#         torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

#         torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
#         torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
#         torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

#     image = image.view(N, C, ID * IH * IW)

#     tnw_val = torch.gather(image, 2, (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
#     tne_val = torch.gather(image, 2, (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
#     tsw_val = torch.gather(image, 2, (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
#     tse_val = torch.gather(image, 2, (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
#     bnw_val = torch.gather(image, 2, (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
#     bne_val = torch.gather(image, 2, (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
#     bsw_val = torch.gather(image, 2, (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
#     bse_val = torch.gather(image, 2, (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

#     out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
#                tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
#                tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
#                tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
#                bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
#                bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
#                bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
#                bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

#     return out_val

# if __name__ == "__main__":
#     if True:
#         image = torch.rand(1, 3, 200, 300, 100)
#         grid = torch.rand(1, 100, 100, 2, 3)
#         start = time.time()
#         output1 = grid_sample_3d(image, grid)
#         end = time.time()
#         print('using {}'.format(end - start))

#         start = time.time()
#         output2 = F.grid_sample(image, grid, padding_mode='border', align_corners=True)
#         end = time.time()
#         print('using {}'.format(end - start))