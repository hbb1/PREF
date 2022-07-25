
import os
from tqdm.auto import tqdm
from opt import config_parser


import json, random
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime
from dataLoader import dataset_dict
import sys


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]


@torch.no_grad()
def export_mesh(args):

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    phasorf = eval(args.model_name)(**kwargs)
    phasorf.load(ckpt)

    alpha,_ = phasorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply',bbox=phasorf.aabb.cpu(), level=0.005)


@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    phasorf = eval(args.model_name)(**kwargs)
    phasorf.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,phasorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset,phasorf, args, renderer, f'{logfolder}/{args.expname}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        PSNRs_test = evaluation_path(test_dataset,phasorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

def reconstruction(args, return_bbox=False, return_memory=False, bbox_only=False):

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list

    
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'
    

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)
    # save config files
    json.dump(args.__dict__, open(f'{logfolder}/config.json',mode='w'),indent=2)


    # init parameters
    if not bbox_only and args.dataset_name=='blender':
        # use tight bbox pre-extracted and stored in meta.py, which takes 2k iters
        data = args.datadir.split('/')[-1]
        from meta import blender_aabb
        aabb = torch.tensor(blender_aabb[data]).reshape(2,3).to(device)
    else:
        # run bbox from scratch
        aabb = train_dataset.scene_bbox.to(device)

        
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        phasorf = eval(args.model_name)(**kwargs)
        phasorf.load(ckpt)
    else:
        phasorf = eval(args.model_name)(aabb, reso_cur, device,
                    # modeling
                    den_num_comp=args.den_num_comp, 
                    app_num_comp=args.app_num_comp, 
                    app_dim=args.app_dim, 
                    softplus_beta=args.softplus_beta,
                    app_aug=args.app_aug,
                    app_ksize = args.app_ksize,
                    den_ksize = args.den_ksize,
                    alpha_init=args.alpha_init,
                    den_scale=args.den_scale,
                    app_scale=args.app_scale,
                    update_dd=args.update_dd, 
                    # rendering 
                    near_far=near_far,
                    shadingMode=args.shadingMode, 
                    alphaMask_thres=args.alpha_mask_thre, 
                    density_shift=args.density_shift, 
                    distance_scale=args.distance_scale,
                    pos_pe=args.pos_pe, 
                    view_pe=args.view_pe, 
                    fea_pe=args.fea_pe, 
                    featureC=args.featureC, 
                    step_ratio=args.step_ratio, 
                    fea2denseAct=args.fea2denseAct)


    grad_vars = phasorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))


    #linear in logrithmic space
    if upsamp_list:
        N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), 
            np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]

    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if not args.ndc_ray:
        allrays, allrgbs = phasorf.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)


    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")


    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:


        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

        rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(rays_train, phasorf, chunk=args.batch_size,
                                N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True)

        loss = torch.mean((rgb_map - rgb_train) ** 2)

        # loss
        total_loss = loss
        loss_tv = torch.tensor([0.0]).cuda()
        if TV_weight_density>0 and (iteration % args.TV_step == 0):
            TV_weight_density *= lr_factor
            loss_tv = phasorf.Parseval_Loss() * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
        if TV_weight_app>0:
            TV_weight_app *= lr_factor
            raise NotImplementedError('not implemented')
            
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss = loss.detach().item()
        
        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse', loss, global_step=iteration)


        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' mse = {loss:.6f}'
                + f' tv_loss = {loss_tv.detach().item():.6f}'
            )
            PSNRs = []


        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
            PSNRs_test = evaluation(test_dataset,phasorf, args, renderer, 
                                    f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                    prtx=f'{iteration:06d}_', N_samples=nSamples, 
                                    white_bg = white_bg, ndc_ray=ndc_ray, 
                                    compute_extra_metrics=args.compute_extra_metric)
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)

        # TODO: to accelerate 
        if update_AlphaMask_list is not None and iteration in update_AlphaMask_list:

            if reso_cur[0] * reso_cur[1] * reso_cur[2]<256**3:# update volume resolution
                reso_mask = reso_cur
            new_aabb = phasorf.updateAlphaMask(tuple(reso_mask))

            if bbox_only:
                return new_aabb

            if return_bbox:
                return (new_aabb[1]-new_aabb[0]).prod().cpu().numpy()

            if iteration == update_AlphaMask_list[0]:
                # use tight aabb already
                # phasorf.shrink(new_aabb)
                if args.TV_weight_density_reset >= 0:
                    TV_weight_density = args.TV_weight_density_reset
                    print(f'TV weight density reset to {args.TV_weight_density_reset}')

            if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                # filter rays outside the bbox
                allrays,allrgbs = phasorf.filtering_rays(allrays,allrgbs)
                trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size)

        # TODO:
        if upsamp_list is not None and iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, phasorf.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
            phasorf.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1 #0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
                print(f'lr set {lr_scale}')
            grad_vars = phasorf.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
        

    phasorf.save(f'{logfolder}/{args.expname}.th')


    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,phasorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset, phasorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

        if return_memory:
            memory = np.sum([v.numel() * v.element_size() for k, v in phasorf.named_parameters()])/2**20
            return np.mean(PSNRs_test), memory

        return np.mean(PSNRs_test)

    if args.render_path:
        c2ws = test_dataset.render_path
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset, phasorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
    
    if not args.render_test:
        PSNRs_test = evaluation(test_dataset,phasorf, args, renderer, 
                                    f'{logfolder}/imgs_vis_all/', N_vis=10, N_samples=nSamples, 
                                    white_bg = white_bg, ndc_ray=ndc_ray, 
                                    compute_extra_metrics=args.compute_extra_metric)
        if return_memory:
            memory = np.sum([v.numel() * v.element_size() for k, v in phasorf.named_parameters()])/2**20
            return np.mean(PSNRs_test), memory

        return np.mean(PSNRs_test)

if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    seed = 2020233254
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    args = config_parser()
    print(args)

    if  args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args)
