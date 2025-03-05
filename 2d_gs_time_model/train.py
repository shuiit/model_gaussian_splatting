#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import pickle
import numpy as np
import matplotlib.pyplot as plt
import model.Utils as Utils

# sys.path.insert(0, 'D:/Documents/gaussian_splat/model/fly_model') # added the pass manually to vscode, should fix that



try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,data_dict, wing_body_pose):
    first_iter = 0

    # joint_list,root_no_bone,list_joints_pitch_update,right_wing_root,left_wing_root = initilize_skeleton(pitch_body = 0)
    # body,right_wing,left_wing = initilize_ptcloud(right_wing_root,left_wing_root)
    # root_no_bone,all_skin_points = initilize_joints(root_no_bone,body,right_wing,left_wing)
    # rot_mat_ew_to_lab,root_no_bone = find_2d_cm_and_update_location(root_no_bone)


    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree,dataset.dist2_th_min)
    scene = Scene(dataset, gaussians,data_dict = data_dict,wing_body_pose = wing_body_pose)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    model_flag = True
    initial_flag = False
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        # if iteration == 198:
        #     wakk = 2
        # if iteration == 200:
        #     model_flag = False
        #     op.opacity_reset_interval = 500
        #     opt.position_lr_init = 0.0000016
        #     opt.position_lr_final = 0.0000000016 
        #     opt.scaling_lr = 0.005
        #     opt.feature_lr = 0.0025
        #     opt.opacity_lr = 0.05
        #     lambda_dist = 5
        #     lambda_normal = 0.05
            
        #     opt.model_position_lr_init = 0
        #     opt.model_rotation_lr = 0.
        #     opt.model_rotation_lr_rwing = 0
        #     opt.model_rotation_lr_lwing = 0
        #     initial_flag = True
        # if iteration > 200:
        #     initial_flag = False

        iter_start.record()
        # gaussians.update_model_location()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, model = model_flag, initial_flag = initial_flag)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) 
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 50 else 0.0 # 7000
        lambda_dist = opt.lambda_dist if iteration > 50 else 0.0 # 3000

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # loss
        total_loss = loss + dist_loss + normal_loss
        
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
          

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save






            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/scale_skin', gaussians.scale_skin, iteration)

                tb_writer.add_scalars('wing_body_pose/wing_angles_right', {'phi':gaussians.right_wing_angles[0],'theta':gaussians.right_wing_angles[1],'psi':gaussians.right_wing_angles[2]}, iteration)
                tb_writer.add_scalars('wing_body_pose/wing_angles_left', {'phi':gaussians.left_wing_angles[0],'theta':gaussians.left_wing_angles[1],'psi':gaussians.left_wing_angles[2]}, iteration)


                tb_writer.add_scalars('wing_body_pose/body_location', {coord:location for coord,location in zip(['x','y','z'],gaussians.body_location)}, iteration)
                tb_writer.add_scalars('wing_body_pose/body_angles', {coord:angles for coord,angles in zip(['yaw','pitch','roll'],gaussians.body_angles)}, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background),ema_dist_for_log,ema_normal_for_log)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

    return gaussians

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs,ema_dist_for_log,ema_normal_for_log):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                metrics = {'loss': l1_test, 'psnr': psnr_test, 'dist_loss': ema_dist_for_log, 'normal_loss':ema_normal_for_log}
                tb_writer.add_hparams(hparam_dict=hparams, metric_dict=metrics,run_name = '.')

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()
    

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_0,1_00,5_00,1_000,1_500,2_000])#[1_0,5_0,1_00,2_00,3_00,4_00,5_00,1_000,2_000,3_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_0,1_00,5_00,1_000,1_500,2_000])#[1_0,5_0,1_00,2_00,3_00,4_00,5_00,1_000,2_000,3_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    lp = lp.extract(args)
    op = op.extract(args)
    pp = pp.extract(args)


    # percent_dense = 0.0001
    # densify_grad_threshold = 0.0002
    print("Optimizing " + args.model_path)
    lp.white_background = True
    lp.dist2_th_min = 0.00000001
    op.position_lr_init =0# 0.0000016
    op.position_lr_final =0# 0.0000000016 
    # op.opacity_lr = 0

    op.iterations  = 2000


    # op.feature_lr = 0
    # op.opacity_lr = 0.1


    # Initialize system state (RNG)
    safe_state(args.quiet)
    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    lambda_dist = 0#5
    lambda_normal = 0#0.05
    path = f'{lp.source_path}/dict/frames_model.pkl'
    with open(path, 'rb') as file:
        data_dict_original = pickle.load(file)

    right_wing_angle = []
    left_wing_angle = []
    body_angles = []
    body_locaiton = []
    wing_body_pose = None
    frames = list(range(900,970,1))
    for depth_ratio in [1]:
        for densify_until_iter in [4000]:#[4000]:
            for opacity_reset_interval in [5000]:
                for scaling_lr in [0]:#[0.001,0.0005]:#[0.00005,0.0005,0.001]:
                    for densification_interval in [0]:#[5,10,15]:#[0,10,20]:
                        for lambda_normal in [0]:#[0.05,0.1,0.15]:#[0.005,0.1]:
                            data_dict = data_dict_original.copy()
                            for idx,key in enumerate(frames):
                                pp.depth_ratio = depth_ratio
                                if idx != 0:
                                    data_dict[key][2] = [xyz,color]

                                    
                                # op.scaling_lr = 0.005
                                # op.feature_lr = 0.0025
                                # op.opacity_lr = 0.05
                                
                                # op.model_position_lr_init = 0
                                # op.model_rotation_lr = 0
                                # op.model_rotation_lr_rwing = 0
                                # op.model_rotation_lr_lwing = 0
                                op.lambda_normal = lambda_normal
                                op.lambda_dist = lambda_dist
                                op.scaling_lr = scaling_lr
                                op.densify_from_iter = 30000
                                op.densify_until_iter = densify_until_iter
                                op.opacity_reset_interval = opacity_reset_interval
                                op.densification_interval = densification_interval
                                # op.model_position_lr_init = 0.01
                                hparams = {
                                    'iterations': op.iterations,
                                    'opacity_reset_interval': op.opacity_reset_interval,
                                    'learning_rate': op.scaling_lr,
                                    'densify_grad_threshold': op.densify_grad_threshold,
                                    'position_lr_init':op.position_lr_init,
                                    'densify_from_iter': op.densify_from_iter,
                                    'percent_dense':op.percent_dense,
                                    'min dist':lp.dist2_th_min,
                                    'densification_interval' : op.densification_interval,
                                    'opacity_lr' : op.opacity_lr,
                                    'feature_lr': op.feature_lr,
                                    'lambda_dist': op.lambda_dist,
                                    'lambda_normal': op.lambda_normal}
                                
                                # lambda_normal_no_dec = f'{lambda_normal}'.split('.')[1]
                                # lambda_scaling_lr_no_dec = f'{scaling_lr}'.split('.')[1]

                                # name_folder = f'iterations_{op.iterations}_densification_interval_{op.densification_interval}_ld_{lambda_dist}_densify_until{op.densify_until_iter}_ln{lambda_normal_no_dec}_lr{lambda_scaling_lr_no_dec}_ori_{opacity_reset_interval}_dr{pp.depth_ratio}'
                                # name_folder = f'iterations_{op.iterations}_ld_{lambda_dist}_ln{lambda_normal_no_dec}_lr{lambda_scaling_lr_no_dec}_ori_{opacity_reset_interval}_dr{pp.depth_ratio}'
                                name_folder = f'model_pose_time_color_scale_v3'


                                lp.model_path = os.path.join(f"G:/My Drive/Research/gaussian_splatting/gaussian_splatting_output/fly_gaussian/2d_output/{name_folder}/", f'{key}/')
                                lp.model_path = os.path.join(f"D:/Documents/gaussian_splattimg_output_model/{name_folder}/", f'{key}/')
                                
                                gaussians = training(lp, op, pp, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint,data_dict[key],wing_body_pose) 
                                xyz = gaussians.get_xyz.cpu().detach().numpy()
                                color = (0.28209479177387814*gaussians.get_features[:,0,:].cpu().detach().numpy() + 0.5)*255
                                
                                right_wing_angle.append(np.array(gaussians.right_wing_angles.detach().cpu()))
                                left_wing_angle.append(np.array(gaussians.left_wing_angles.detach().cpu()))
                                body_angles.append(np.array(gaussians.body_angles.detach().cpu()))
                                body_locaiton.append(np.array(gaussians.body_location.detach().cpu()))
                                wing_body_pose = [np.array(gaussians.right_wing_angles.detach().cpu()),np.array(gaussians.left_wing_angles.detach().cpu())
                                                  ,np.array(gaussians.body_angles.detach().cpu()),np.array(gaussians.body_location.detach().cpu())]
                                
                                
                                
                                del gaussians
        
    pose = {}
    pose['right_wing'] = np.vstack(right_wing_angle)
    pose['left_wing'] = np.vstack(left_wing_angle)


    pose['body_location'] = np.vstack(body_locaiton)
    pose['body_angle'] = np.vstack(body_angles)
    
    with open(f'{lp.model_path}/body_wing_pose.pkl', 'wb') as f:
        pickle.dump(pose, f)
    # All done
    print("\nTraining complete.")