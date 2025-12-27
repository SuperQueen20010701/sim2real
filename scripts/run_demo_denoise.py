# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys
import argparse
import imageio
import torch
import logging
import cv2
import numpy as np
import open3d as o3d
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import set_logging_format, set_seed, vis_disparity, depth2xyzmap, toOpen3dCloud
from core.foundation_stereo import FoundationStereo
from typing import List, Tuple,Optional
import json
import shutil

median_kernel = 3 
consistency_threshold = 0.1
# 加入中值滤波以及局部一致性 (disperity pic)
def filter_disperity(disp: np.asarray):
  disp_filter = disp.copy()
  valid_mask = np.isfinite(disp) & (disp > 0)
  if not valid_mask.any():
    return disp_filter
  if (median_kernel > 0) :
   
    if(median_kernel % 2 == 1) and median_kernel > 1 :
      disp_median_filter = cv2.medianBlur(disp_filter.astype(np.float32),median_kernel)
      disp_filter[valid_mask] = disp_median_filter[valid_mask]
  else:
    pass
  disp_c_filtered  = disp_filter.copy()

  grad_x = np.abs(np.gradient(disp_c_filtered,axis=1))
  grad_y = np.abs(np.gradient(disp_c_filtered,axis=0))
  mean_kernel = np.ones((5,5),dtype=np.float32) / 25.0
  local_mean_disp = cv2.filter2D(disp_c_filtered , -1 , mean_kernel)
  local_mean_disp[~valid_mask] = 1.0 
  grad_x_norm = grad_x / (local_mean_disp + 1e-6)
  grad_y_norm = grad_y / (local_mean_disp + 1e-6)

  high_grad_mask = valid_mask &((grad_x_norm > consistency_threshold) | (grad_y_norm > consistency_threshold))
  kernel_check = np.ones((3,3),dtype=np.float32)
  num_high_grad = cv2.filter2D(high_grad_mask.astype(np.float32),-1,kernel_check)
  outlier_mask = high_grad_mask & (num_high_grad < 3)
  disp_c_filtered[outlier_mask] = np.inf

  return disp_c_filtered

# 对于3D点云加入统计滤波
nb_neighbors = 5
std_ratio=2.5
def filter_point_cloud(pcd , args):
  if len(pcd.points) == 0:
    return pcd 
  # 使用统计滤波计算离群点 
  try:
    cl , ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd = pcd.select_by_index(ind)
    logging.info(f"after statistical outlier remove , {len(pcd.points)} number points remaining")
  except Exception as e:
    logging.warning(f"statistical outlier remove failed :{e}")

  return pcd

def find_image_pairs(data_root: str) -> List[Tuple[str, str]]:
  """Find every directory containing both left.png/right.png under data_root."""
  pairs: List[Tuple[str, str]] = []
  for dirpath, _, filenames in os.walk(data_root):
    if "left.png" in filenames and "right.png" in filenames:
      pairs.append((os.path.join(dirpath, "left.png"), os.path.join(dirpath, "right.png")))
  pairs.sort()
  return pairs

def parse_json_to_param(K_json_path :str) -> Tuple[np.ndarray , Optional[float]]:
  with open(K_json_path, 'r') as f:
    param = json.load(f)

    if 'left_intrinsic_matrix' in param:
      K_robot = param['left_intrinsic_matrix']
    elif 'depth_intrinsic_matrix' in param:
      K_robot = param['depth_intrinsic_matrix']
    else:
      K_robot = None
      logging.warning("ERROR !! no param found in param")
    if K_robot is not None:
      K_robot = np.array(K_robot, dtype=np.float32)
    if 'base_line' in param:
      baseline_robot= param['base_line']
    else:
      baseline_robot = None
      logging.warning("ERROR !!no baseline found in param")
    return K_robot ,baseline_robot 

if __name__=="__main__":
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('--left_file', default=f'{code_dir}/../assets/left.png', type=str)
  parser.add_argument('--right_file', default=f'{code_dir}/../assets/right.png', type=str)
  parser.add_argument('--intrinsic_file', default=f'{code_dir}/../assets/K.txt', type=str, help='camera intrinsic matrix and baseline file')
  parser.add_argument('--out_dir', default=f'{code_dir}/../output/', type=str, help='the directory to save results')
  parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
  parser.add_argument('--hiera', default=0, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
  parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
  parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
  parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
  parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')


  parser.add_argument('--data_root_dir',type=str , default='',help="data root directory")
  parser.add_argument('--output_root_dir',type=str ,default='',help="output root directory")
  parser.add_argument('--filter_disperity',type=int,default=0,help="filter disparity map")
  parser.add_argument('--filter_point_cloud',type=int,default=0,help="filter point cloud")
  parser.add_argument('--save_disp_tiff',type=int,default=1,help="save disparity map as tiff file")
  parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
  parser.add_argument('--denoise_cloud', type=int, default=0, help='whether to denoise the point cloud')
  parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
  parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth', type=str, help='pretrained model path')
  parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use (default: 0)')
  parser.add_argument('--index_success_file',type=str , default='',help='record the successful disperity output path')
  parser.add_argument('--invalid_index_file',type=str,default='',help='record the unsuccessful disperity output path')
  args = parser.parse_args()
  # choose gpu 
  gpu_id = args.gpu
  device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
  if torch.cuda.is_available():
    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()
    logging.info(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
  else:
    logging.warning("CUDA not available, using CPU")
  set_logging_format()
  set_seed(0)
  torch.autograd.set_grad_enabled(False)

  ckpt_dir = args.ckpt_dir
  cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
  if 'vit_size' not in cfg:
    cfg['vit_size'] = 'vitl'
  for k in args.__dict__:
    cfg[k] = args.__dict__[k]
  args = OmegaConf.create(cfg)
  logging.info(f"args:\n{args}")
  logging.info(f"Using pretrained model from {ckpt_dir}")
  # load model
  model = FoundationStereo(args)
  # Load checkpoint to CPU first to avoid GPU memory issues
  ckpt = torch.load(ckpt_dir, map_location='cpu')
  logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
  model.load_state_dict(ckpt['model'])
  del ckpt  # Free memory immediately after loading

  model.cuda()
  model.eval()
  # batch process 
  if (args.data_root_dir is not None )and (args.data_root_dir != ''):
    data_root_dir = args.data_root_dir
    # 输出目录设置
    if (args.output_root_dir != ''):
      output_path_dir = args.output_root_dir
      if os.path.exists(output_path_dir):
        logging.warning(f"Output directory already exists: {output_path_dir}. Files may be overwritten.")
      os.makedirs(output_path_dir, exist_ok=True)
    else:
      default_output_name = os.path.basename(os.path.normpath(data_root_dir)) + '_output'
      default_output_parent = os.path.dirname(os.path.normpath(data_root_dir))
      args.output_root_dir = os.path.join(default_output_parent, default_output_name)
      logging.info(f"No output directory specified, using default: {args.output_root_dir}")
      if os.path.exists(args.output_root_dir):
        logging.warning(f"Default output directory already exists: {args.output_root_dir}. Files may be overwritten.")
      os.makedirs(args.output_root_dir, exist_ok=True)

    # 寻找对应的照片对
    image_pairs = find_image_pairs(data_root_dir)
    param_abormal_list : List[str] = []
    success_index : List[str] = []
    invalid_index : List[str] = []

    for idx, (left_path_,right_path_) in enumerate(image_pairs):
      try:
        logging.info(f"Processing [{idx+1}/{len(image_pairs)}]: {left_path_}")
        image_l = imageio.imread(left_path_)
        image_r = imageio.imread(right_path_)
        scale = args.scale
        assert scale<=1, "scale must be <=1"
        image_l = cv2.resize(image_l, fx=scale, fy=scale, dsize=None)
        image_r = cv2.resize(image_r, fx=scale, fy=scale, dsize=None)
        H,W = image_l.shape[:2]
        image_l_ori = image_l.copy()
        image_r_ori = image_r.copy()
        logging.info(f"image_l: {image_l.shape}")
        logging.info(f"image_r: {image_r.shape}")

        image_l =torch.as_tensor(image_l).cuda().float()[None].permute(0,3,1,2)
        image_r = torch.as_tensor(image_r).cuda().float()[None].permute(0,3,1,2)
        padder = InputPadder(image_l.shape, divis_by=32, force_square=False)
        image_l, image_r = padder.pad(image_l, image_r)

        with torch.cuda.amp.autocast(True):
          if not args.hiera:
            disp = model.forward(image_l, image_r, iters=args.valid_iters, test_mode=True)
          else:
            disp = model.run_hierachical(image_l, image_r, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H,W)

        # remove invisible points 
        if args.remove_invisible:
          yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
          us_right = xx-disp
          invalid = us_right<0
          disp[invalid] = np.inf

        # save disparity 
        disp_saved = False
        if args.save_disp_tiff:
          if args.filter_disperity:
            disp = filter_disperity(disp)
          disp_parent_dir = os.path.dirname(left_path_)
          rel_dir_path = os.path.relpath(disp_parent_dir,data_root_dir) # relative path 
          out_disp_dir = os.path.join(args.output_root_dir,rel_dir_path)
          os.makedirs(out_disp_dir,exist_ok=True)
          disp_out_path = os.path.join(out_disp_dir,"disp.tiff")
          try:
            imageio.imwrite(f'{disp_out_path}', disp.astype(np.float32))
            if os.path.exists(disp_out_path):
              success_index.append((os.path.abspath(left_path_),os.path.abspath(right_path_),os.path.abspath(disp_out_path)))
              disp_saved = True
            else:
              logging.warning(f'WARNNING !! disp.tiff file not created: {disp_out_path}')
              invalid_index.append(os.path.abspath(os.path.dirname(left_path_)))
          except Exception as e:
            logging.warning(f'WARNNING !! failed to save disp.tiff: {e}')
            invalid_index.append(os.path.abspath(os.path.dirname(left_path_)))
        
        vis = vis_disparity(disp)
        vis = np.concatenate([image_l_ori, vis], axis=1)

        vis_parent_dir = os.path.dirname(left_path_)
        rel_dir_path = os.path.relpath(vis_parent_dir,data_root_dir)
        out_vis_dir = os.path.join(args.output_root_dir,rel_dir_path)
        os.makedirs(out_vis_dir,exist_ok=True)
        vis_out_path = os.path.join(out_vis_dir,"vis.png")
        imageio.imwrite(vis_out_path, vis)
        
        logging.info(f"Output saved to {out_vis_dir}")

        # save point cloud
        if args.get_pc:
          left_img_parent_dir = os.path.dirname(left_path_)
          json_path = os.path.join(left_img_parent_dir,'param.json')
          K,baseline = parse_json_to_param(json_path)
          # 错误的param json 跳过 记录异常
          if K is None or baseline is None:
            logging.warning(f'WARNNING !! detected no K or baseine param.json !! json path is {json_path}')
            param_abormal_list.append(json_path)
            continue
          K[:2] *= scale
          depth = K[0,0]*baseline/disp
          
          pcd_parent_dir = os.path.dirname(left_path_)
          rel_dir_path = os.path.relpath(pcd_parent_dir,data_root_dir)
          pcd_out_dir = os.path.join(args.output_root_dir,rel_dir_path)
          os.makedirs(pcd_out_dir,exist_ok=True)
          depth_out_path = os.path.join(pcd_out_dir,"depth_meter.npy")
          np.save(depth_out_path, depth)
          
          xyz_map = depth2xyzmap(depth, K)
          pcd = toOpen3dCloud(xyz_map.reshape(-1,3), image_l_ori.reshape(-1,3))
          keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.z_far)
          keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
          pcd = pcd.select_by_index(keep_ids)
          if args.filter_point_cloud:
            pcd = filter_point_cloud(pcd, args)

          pcd_out_path = os.path.join(pcd_out_dir,"cloud.ply")
          o3d.io.write_point_cloud(pcd_out_path, pcd)
          logging.info(f"PCL saved to {pcd_out_dir}")
      except Exception as e:
        logging.error(f"Error processing {left_path_}: {e}", exc_info=True)
        invalid_index.append(os.path.abspath(os.path.dirname(left_path_)))
        continue 
    
    if len(param_abormal_list) != 0:
      invalid_param_path = os.path.join(args.output_root_dir, 'invalid_param.txt')
      with open(invalid_param_path, 'w') as f:
        for param_path in param_abormal_list:
          f.write(f'{param_path}\n')
      logging.info(f'INFO !! saved {len(param_abormal_list)} invalid param.json paths to {invalid_param_path}')
    else:
      logging.info("INFO !! no invalid param.json file found")
    
    if len(success_index) != 0:
      idx_path_rec = args.index_success_file if (args.index_success_file is not None and args.index_success_file != '') else os.path.join(args.output_root_dir, 'success_index.txt')
      with open(idx_path_rec, 'w') as f:
        for single_entry in success_index:
          left_path, right_path, disp_path = single_entry
          f.write(f'{left_path}\t{right_path}\t{disp_path}\n')
      logging.info(f'INFO !! successfully saved index for {len(success_index)} items to {idx_path_rec}')

    if len(invalid_index) != 0:
      invalid_idx_path = args.invalid_index_file if (args.invalid_index_file is not None and args.invalid_index_file != '') else os.path.join(args.output_root_dir, 'invalid_index.txt')
      with open(invalid_idx_path, 'w') as f:
        for err_path in invalid_index:
          f.write(f'{err_path}\n')
      logging.info(f'INFO !! saved invalid index for {len(invalid_index)} items to {invalid_idx_path}')
    sys.exit(0)
  else:
    logging.info("INFO !! processing single file ")
    os.makedirs(args.out_dir, exist_ok=True)
    code_dir = os.path.dirname(os.path.realpath(__file__))
    img0 = imageio.imread(args.left_file)
    img1 = imageio.imread(args.right_file)
    scale = args.scale
    assert scale<=1, "scale must be <=1"
    img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
    img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
    H,W = img0.shape[:2]
    img0_ori = img0.copy()
    logging.info(f"img0: {img0.shape}")

    img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
    img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)
    
    # disperity 处理逻辑
    with torch.cuda.amp.autocast(True):
      if not args.hiera:
        disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
      else:
        disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H,W)

    # 对不不可见部分的移除
    if args.remove_invisible:
      yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
      us_right = xx-disp
      invalid = us_right<0
      disp[invalid] = np.inf
    # 保存视差 & 对视差图进行滤波
    if args.save_disp_tiff:
      if args.filter_disperity:
        disp = filter_disperity(disp)
      imageio.imwrite(f'{args.out_dir}/disp.tiff', disp.astype(np.float32))
    # 视差可视化
    vis = vis_disparity(disp)
    vis = np.concatenate([img0_ori, vis], axis=1)
    imageio.imwrite(f'{args.out_dir}/vis.png', vis)
    logging.info(f"Output saved to {args.out_dir}")

    # 获取点云信息
    if args.get_pc:
      K,baseline = parse_json_to_param(args.intrinsic_file)
      # with open(args.intrinsic_file, 'r') as f:
      #   lines = f.readlines()
      #   K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
      #   baseline = float(lines[1])
      K[:2] *= scale
      depth = K[0,0]*baseline/disp
      np.save(f'{args.out_dir}/depth_meter.npy', depth)
      xyz_map = depth2xyzmap(depth, K)
      pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0_ori.reshape(-1,3))
      keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.z_far)
      keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
      pcd = pcd.select_by_index(keep_ids)
      if args.filter_point_cloud:
        pcd = filter_point_cloud(pcd, args)
      o3d.io.write_point_cloud(f'{args.out_dir}/cloud.ply', pcd)
      logging.info(f"PCL saved to {args.out_dir}")

      if args.denoise_cloud:
        logging.info("[Optional step] denoise point cloud...")
        cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
        inlier_cloud = pcd.select_by_index(ind)
        o3d.io.write_point_cloud(f'{args.out_dir}/cloud_denoise.ply', inlier_cloud)
        pcd = inlier_cloud