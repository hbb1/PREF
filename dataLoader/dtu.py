
import torch
import cv2 as cv
import numpy as np
import os
from glob import glob
from .ray_utils import *
from torch.utils.data import Dataset


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

class DTUDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1, is_stack=False):
        """
        img_wh should be set to a tuple ex: (1152, 864) to enable test mode!
        """
        self.split = split
        self.root_dir = datadir
        self.is_stack = is_stack
        self.downsample = downsample
        self.white_bg = True
        self.camera_dict = np.load(os.path.join(self.root_dir, 'cameras_sphere.npz'))

        self.img_wh = (int(1600 / downsample), int(1200 / downsample))

        # self.scan = os.path.basename(datadir)
        # self.split = split
        #
        # self.img_wh = (int(640 / downsample), int(512 / downsample))
        # self.downsample = downsample
        #
        # self.scale_factor = 1.0 / 200
        # self.define_transforms()

        # self.scene_bbox = np.array([[-1.01, -1.01, -1.01], [1.01,  1.01,  1.01]])
        self.near_far = [0.1, 10]
        #
        # self.re_centerMat = np.array([[0.311619, -0.853452, 0.417749, -1.4379079],
        #                               [0.0270351, 0.44742498, 0.893913, -2.801856],
        #                               [-0.949823, -0.267266, 0.162499, -0.35806254],
        #                               [0., 0., 0., 1.]])
        self.read_meta()
        self.get_bbox()

    # def define_transforms(self):
    #     self.transform = T.ToTensor()

    def get_bbox(self):
        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.root_dir, 'cameras_sphere.npz'))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.scene_bbox = torch.from_numpy(np.stack((object_bbox_min[:3, 0],object_bbox_max[:3, 0]))).float()
        # self.near_far = [2.125, 4.525]

    def gen_rays_at(self, intrinsic, c2w, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        W,H = self.img_wh
        tx = torch.linspace(0, W - 1, W // l)+0.5
        ty = torch.linspace(0, H - 1, H // l)+0.5
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        intrinsic_inv = torch.inverse(intrinsic)
        p = torch.matmul(intrinsic_inv[None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(c2w[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = c2w[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1).reshape(-1,3), rays_v.transpose(0, 1).reshape(-1,3)

    def read_meta(self):

        images_lis = sorted(glob(os.path.join(self.root_dir, 'image/*.png')))
        images_np = np.stack([cv.resize(cv.imread(im_name),self.img_wh) for im_name in images_lis]) / 255.0
        masks_lis = sorted(glob(os.path.join(self.root_dir, 'mask/*.png')))
        masks_np = np.stack([cv.resize(cv.imread(im_name),self.img_wh) for im_name in masks_lis])>128

        self.all_rgbs = torch.from_numpy(images_np.astype(np.float32)[...,[2,1,0]])  # [n_images, H, W, 3]
        self.all_masks  = torch.from_numpy(masks_np>0)   # [n_images, H, W, 3]
        self.img_wh = [self.all_rgbs.shape[2],self.all_rgbs.shape[1]]

        # world_mat is a projection matrix from world to image
        n_images = len(images_lis)
        world_mats_np = [self.camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
        self.scale_mats_np = [self.camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

        # W,H = self.img_wh
        self.all_rays = []
        self.intrinsics, self.poses = [],[]
        for img_idx, (scale_mat, world_mat) in enumerate(zip(self.scale_mats_np, world_mats_np)):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsic, c2w = load_K_Rt_from_P(None, P)

            c2w = torch.from_numpy(c2w).float()
            intrinsic = torch.from_numpy(intrinsic).float()
            intrinsic[:2] /= self.downsample

            self.poses.append(c2w)
            self.intrinsics.append(intrinsic)

            # center = intrinsic[:2,-1]
            # directions = get_ray_directions(H, W, [intrinsic[0,0], intrinsic[1,1]], center=center)  # (h, w, 3)
            # directions = directions / torch.norm(directions, dim=-1, keepdim=True)
            # rays_o, rays_d = get_rays(directions, c2w)  # both (h*w, 3)
            rays_o, rays_d = self.gen_rays_at(intrinsic,c2w)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)



        self.intrinsics, self.poses = torch.stack(self.intrinsics), torch.stack(self.poses)

        self.all_rgbs[~self.all_masks] = 1.0
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = self.all_rgbs.reshape(-1,3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = self.all_rgbs.reshape(-1, *self.img_wh[::-1],3)  # (len(self.meta['frames]),h,w,3)

    def __len__(self):
        return len(self.all_rays)

    def __getitem__(self, idx):
        if self.split == 'train':
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}
        else:
            sample = {'rays': self.all_rays[idx]}
        return sample