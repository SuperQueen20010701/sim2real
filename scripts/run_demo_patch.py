# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *
from typing import List, Tuple, Optional
import cv2
import imageio
import numpy as np
import json

# robot_1 的 K.txt 文件路径
K_TXT_PATH = '/DATA/disk0/zhaobojun/FoundationStereo_ws/src/FoundationStereo/assets/K.txt'

# 视差滤波参数
median_kernel = 3 
consistency_threshold = 0.1

# 点云滤波参数
nb_neighbors = 5
std_ratio = 2.5

def filter_disperity(disp: np.ndarray):
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

def parse_k_txt(k_txt_path: str) -> Tuple[np.ndarray, float]:
  with open(k_txt_path, 'r') as f:
    lines = f.readlines()
    if len(lines) < 2:
      raise ValueError(f"ERROR !! K.txt file format error : {k_txt_path}")
    K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3, 3)
    baseline = float(lines[1])
    return K, baseline

def parse_json_to_param(json_path : str,robot_idx = '', require_baseline: bool = True) -> Tuple[np.ndarray , Optional[float]]:
  with open(json_path, 'r') as f:
    params = json.load(f)
    
    if 'left_intrinsic_matrix' in params:
      K_robot = params['left_intrinsic_matrix']
    elif 'depth_intrinsic_matrix' in params:
      K_robot = params['depth_intrinsic_matrix']
    else:
      K_robot = np.eye(3)
      raise ValueError("param.json 中未找到 left_intrinsic_matrix 或 depth_intrinsic_matrix")
    
    if len(K_robot) != 3 or any(len(row) != 3 for row in K_robot):
      raise ValueError("Value Error in intrinsic matrix")
    K_robot = np.array(K_robot, dtype=np.float32)
    
    if 'base_line' in params:
      baseline_robot = params['base_line']
    else:
      if require_baseline:
        if robot_idx =='robot_10':
          baseline_robot = 0.060192495584487915
        else:
          base_line = None
          raise ValueError("param.json 中未找到 base_line 信息")
      else:
        baseline_robot = None
    
    return K_robot, baseline_robot

# 寻找对应图像对
def find_image_pairs(data_root: str) -> List[Tuple[str, str]]:
  pairs: List[Tuple[str, str]] = []
  for dirpath, _, filenames in os.walk(data_root):
    if "left.png" in filenames and "right.png" in filenames:
      pairs.append((os.path.join(dirpath, "left.png"), os.path.join(dirpath, "right.png")))
  pairs.sort()
  return pairs

# 按照 robot id 进行划分  
def classify_paires_by_robot_id(iamge_paires : List[Tuple[str, str]]) -> dict:
  robot_dict_assemble = {
    'robot_1' : [],
    'robot_5' : [],
    'robot_10' : []
  }

  for left_png , right_png in iamge_paires:
    pair_img = (left_png , right_png)
    # 检查 robot_10 (需要先检查，避免匹配到 robot_1)
    if ('robot_10' in left_png) and ('robot_10' in right_png):
      robot_dict_assemble['robot_10'].append(pair_img)
    # 检查 robot_5 或 robot_05
    elif ('robot_5' in left_png) and ('robot_5' in right_png):
      robot_dict_assemble['robot_5'].append(pair_img)
    # 检查 robot_1 或 robot_01
    elif ('robot_1' in left_png) and ('robot_1' in right_png):
      robot_dict_assemble['robot_1'].append(pair_img)
    else:
      logging.warning(f"WARNNING !! find the path not consistency with robot 01 , 05, 10: left={left_png}, right={right_png}")
      continue

  for robot_id in robot_dict_assemble :
    robot_dict_assemble[robot_id].sort()
    logging.info(f"Found {len(robot_dict_assemble[robot_id])} image pairs for {robot_id}")
  return robot_dict_assemble

def infer_disperity(left_img_path:str ,right_img_path:str,args,model,device) -> Tuple[np.ndarray, np.ndarray, int, int]:
    img_l = imageio.imread(left_img_path)
    img_r = imageio.imread(right_img_path)
    scale_ = args.scale

    assert scale_ <= 1,"scale must be <=1"
    img_l = cv2.resize(img_l , fx=scale_ , fy = scale_ ,dsize = None)
    img_r = cv2.resize(img_r ,fx = scale_ , fy = scale_ , dsize = None)

    H,W = img_l.shape[:2]
    image_l_ori = img_l.copy()

    img_l = torch.as_tensor(img_l).to(device).float()[None].permute(0,3,1,2)
    img_r = torch.as_tensor(img_r).to(device).float()[None].permute(0,3,1,2)

    padder = InputPadder(img_l.shape , divis_by=32 , force_square = False)
    img_l , img_r = padder.pad(img_l ,img_r)  # 实现图像填充

    # 视差图推理模式
    disp = None
    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
      if not args.hiera :
        disp = model.forward(img_l , img_r ,iters = args.valid_iters ,  test_mode=True)
      else:
        disp = model.run_hierachical(img_l, img_r, iters=args.valid_iters, test_mode=True, small_ratio=0.5)

    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H,W) 
    # 不可见视差点的处理
    if args.remove_invisible:
      yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
      us_right = xx-disp  
      invalid = us_right<0 
      disp[invalid] = np.inf

    return disp, image_l_ori, H, W

if __name__=="__main__":
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('--left_file', default=f'{code_dir}/../assets/left.png', type=str)
  parser.add_argument('--right_file', default=f'{code_dir}/../assets/right.png', type=str)
  parser.add_argument('--intrinsic_file', default=f'{code_dir}/../assets/K.txt', type=str, help='camera intrinsic matrix and baseline file')
  parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth', type=str, help='pretrained model path')
  parser.add_argument('--out_dir', default=f'{code_dir}/../output/', type=str, help='the directory to save results')
  parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
  parser.add_argument('--hiera', default=0, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
  parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
  parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
  parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
  parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
  parser.add_argument('--denoise_cloud', type=int, default=1, help='whether to denoise the point cloud')
  parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
  parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
  
  # 单文件 / 多文件
  parser.add_argument('--batch_process',type=int ,default =0, help='choice to batch processing')
  parser.add_argument('--filter_disperity',type=int , default =1 , help='whether to filter the disperity')
  parser.add_argument('--filter_point_cloud',type= int ,default = 1 , help='whether to filter the point cloud')
  parser.add_argument('--output_root_dir',type=str , default= '',help='the save dir for the disperity')
  parser.add_argument('--index_success_file',type=str,default='',help='record the succeed save disperity paires')
  parser.add_argument('--invalid_index_file',type=str,default='',help='record the un successfull disperity paires')
  parser.add_argument('--data_root_dir',type=str,default='',help='the input data root directory')
  parser.add_argument('--save_disp_tiff',type=int ,default=1 ,help='whether to save the disperity tiff picture')
  parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use (default: 0)')
  args = parser.parse_args()
  gpu_id = args.gpu

  # Set GPU device
  device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
  if torch.cuda.is_available():
    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()
    logging.info(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
  else:
    logging.warning("CUDA not available, using CPU")

  # load config 
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
  del ckpt
  if torch.cuda.is_available():
    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()

  model.to(device)
  model.eval()

  # batch process this channel 
  if args.batch_process :
    if (args.data_root_dir is not None) and (args.data_root_dir != ''):
      data_root_dir = args.data_root_dir
      os.makedirs(args.output_root_dir , exist_ok = True)

      # 存储有效计算视差和无效视差的index 
      index_entries:List[str] = []
      invalid_entries :List[str] = [] 

      # 寻找对应的图片对
      iamges_pairs = find_image_pairs(data_root_dir)
      logging.info(f"Found {len(iamges_pairs)} total image pairs")
      # 按照robot id 划分
      png_pair_cls_id = classify_paires_by_robot_id(iamges_pairs)

      for robot_idx ,image_paires in png_pair_cls_id.items():
        logging.info(f"Processing {robot_idx} with {len(image_paires)} image pairs")
        for left_img_path ,right_img_path in image_paires:
          # 判断左右图像是否能够有效计算视差
          try:
            disp,image_l_ori , H , W = infer_disperity(left_img_path , right_img_path , args, model, device)
            disp_out_path = None
            # 保存视差tiff
            if args.save_disp_tiff:
              # 是否对视差图进行filter 
              if args.filter_disperity:
                disp = filter_disperity(disp)
              img_parent_name = os.path.dirname(left_img_path)
              rel_dir_path = os.path.relpath(img_parent_name,data_root_dir)
              out_disp_dir = os.path.join(args.output_root_dir,rel_dir_path)
              os.makedirs(out_disp_dir,exist_ok=True)
              disp_out_path = os.path.join(out_disp_dir,"disp.tiff")
              imageio.imwrite(f'{disp_out_path}', disp.astype(np.float32))
                
              #保存可视化视差视图vis
              vis = vis_disparity(disp)
              vis = np.concatenate([image_l_ori, vis], axis=1) 
              imageio.imwrite(f'{out_disp_dir}/vis.png', vis)
              logging.info(f"Output disparity map and visibile disparity map saved to {out_disp_dir}")
              # 保存点云信息
              if args.get_pc:
                # 判断robot id
                if robot_idx == 'robot_1':
                  # robot_1 从 K.txt 文件读取内参和baseline
                  try:
                    K_robot, baseline_robot = parse_k_txt(K_TXT_PATH)
                    logging.info(f"robot_1 从 {K_TXT_PATH} 读取参数: fx={K_robot[0,0]:.2f}, baseline={baseline_robot:.6f}")
                  except Exception as e:
                    logging.error(f"读取 robot_1 K.txt 文件失败 ({K_TXT_PATH}): {e}")
                    continue
                elif (robot_idx == 'robot_5') or( robot_idx == 'robot_10'):
                  # robot_5 和 robot_10 从 param.json 读取
                  try:
                    K_robot, baseline_robot = parse_json_to_param(os.path.join(img_parent_name,'param.json'),robot_idx)
                    logging.info(f"{robot_idx} 从 param.json 读取参数: fx={K_robot[0,0]:.2f}, baseline={baseline_robot:.6f}")
                  except Exception as e:
                    logging.error(f"读取 {robot_idx} param.json 失败: {e}")
                    continue
                else:
                  logging.warning(f"WARNNING !! No image pair satisfied ")
                  continue
                scale = args.scale
                K_robot[:2] *= scale
                baseline_robot *= scale
                depth = K_robot[0,0]*baseline_robot/disp
                # 保存深度信息
                np.save(f'{out_disp_dir}/depth_meter.npy', depth)
                xyz_map = depth2xyzmap(depth, K_robot)
                pcd = toOpen3dCloud(xyz_map.reshape(-1,3), image_l_ori.reshape(-1,3))
                keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.z_far)
                keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
                pcd = pcd.select_by_index(keep_ids)
                # 记录每个点对应的原图像素 flat id（v*W+u），用于“无再投影误差”的反推视差
                flat_ids = keep_ids.copy()
                # 对点云进行滤波
                if args.filter_point_cloud:
                  try:
                    _cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
                    pcd = pcd.select_by_index(ind)
                    flat_ids = flat_ids[np.asarray(ind, dtype=np.int64)]
                    logging.info(f"after statistical outlier remove , {len(pcd.points)} number points remaining")
                  except Exception as e:
                    logging.warning(f"statistical outlier remove failed :{e}")
                o3d.io.write_point_cloud(f'{out_disp_dir}/cloud.ply', pcd)
                logging.info(f"PCL saved to {out_disp_dir}")

                # 滤波/降噪后的点云”反推视差
                disp_from_pcd = disparity_from_pcd_flat_ids(
                  pcd=pcd,
                  flat_ids=flat_ids,
                  H=H,
                  W=W,
                  fx=K_robot[0, 0],
                  baseline=baseline_robot,
                  invalid_value=np.inf,
                )
                # 保存滤波后的disp.tiff 
                disp_from_pcd_path = os.path.join(out_disp_dir, "disp_from_pcd.tiff")
                imageio.imwrite(disp_from_pcd_path, disp_from_pcd.astype(np.float32))
                logging.info(f"Disparity map from filtered point cloud saved to {disp_from_pcd_path}")

                # 保存滤波后的vis png 可视化
                vis_disp_from_pcd = vis_disparity(disp_from_pcd)
                vis_disp_from_pcd = np.concatenate([image_l_ori, vis_disp_from_pcd], axis=1) 
                imageio.imwrite(f'{out_disp_dir}/vis_disp_from_pcd.png', vis_disp_from_pcd)
                logging.info(f"Output disparity map and visibile disparity map saved to {out_disp_dir}")
            if disp_out_path:
              # filter out the filter from pcd path 
              index_entries.append((os.path.abspath(left_img_path), os.path.abspath(right_img_path), os.path.abspath(disp_from_pcd_path)))
          except Exception as e:
            logging.warning(f"ERROR!!  calculate the disperity : {e}")
            invalid_entries.append(os.path.dirname(left_img_path))
            continue
      #将index 写入文件 方便后续处理
      if index_entries:
        index_path = args.index_success_file if args.index_success_file else os.path.join(args.output_root_dir, "success_index.txt")
        with open(index_path, "w") as f:
          for entry in index_entries:
            left_path, right_path, disp_path = entry
            f.write(f"{left_path}\t{right_path}\t{disp_path}\n")
        logging.info(f"Saved index for {len(index_entries)} items to {index_path}")
      if invalid_entries:
        invalid_index_path = args.invalid_index_file if args.invalid_index_file else os.path.join(args.output_root_dir, "invalid_index.txt")
        with open(invalid_index_path, "w") as f:
          for error_dir in invalid_entries:
            f.write(f"{error_dir}\n")
        logging.info(f"Saved invalid index for {len(invalid_entries)} items to {invalid_index_path}")
