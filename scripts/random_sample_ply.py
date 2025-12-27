import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from typing import List ,Tuple,Optional
import logging 

# Add parent directory to path to import Utils
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from Utils import set_logging_format, set_seed


def list_subdirs(dir_path: str) -> List[str]:
    if not os.path.exists(dir_path):
        return []
    items: List[str] = []
    for name in os.listdir(dir_path):
        p = os.path.join(dir_path, name)
        if os.path.isdir(p):
            items.append(name)
    return sorted(items)

def copytree_safe(src_dir: str, dst_dir: str) -> bool:
    try:
        # 确保源目录存在
        if not os.path.exists(src_dir):
            logging.error(f"Source directory does not exist: {src_dir}")
            return False
        
        # 如果目标目录已存在，先删除
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        
        # 确保目标目录的父目录存在
        parent_dir = os.path.dirname(dst_dir)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        
        # 复制整个目录树
        shutil.copytree(src_dir, dst_dir)
        return True
    except Exception as e:
        logging.error(f"Error copytree {src_dir} -> {dst_dir}: {e}")
        return False

def random_smaple(data_input_root_dir :str , target_output_dir : str ,
                  num_samples : int =10,random_seed :Optional[int] =None):
    if random_seed is not None:
        random.seed(random_seed)

    data_input_root_dir = Path(data_input_root_dir)
    if not data_input_root_dir.exists():
        logging.error(f'输入的文件路径存在错误:{data_input_root_dir}')
        return 
    
    date_folder_id : List[str] = []

    for id_sub_path in os.listdir(str(data_input_root_dir)):
        robot_id_path = os.path.join(str(data_input_root_dir),id_sub_path)
        if(os.path.isdir(robot_id_path)):
            date_folder_id.append(id_sub_path)
    
    date_folder_id = sorted(date_folder_id)
    
    if not date_folder_id:
        logging.warning(f"ERROR !! no robot id folder path is finding !!")
        return
    
    for robot_id in date_folder_id:
        total_copied = 0
        total_failed = 0
        robot_id_dir = os.path.join(str(data_input_root_dir),robot_id)
        if os.path.exists(robot_id_dir) :
            date_folder = []
            for sub_dir in os.listdir(robot_id_dir):
                sub_abs_path = os.path.join(robot_id_dir ,sub_dir)
                if os.path.isdir(sub_abs_path):
                    # 日期
                    date_folder.append(sub_dir)
            if not date_folder:
                continue

            for single_date in date_folder:
                # 当天的data
                data_dir = os.path.join(robot_id_dir,single_date ,"data")
                # 执行随机抽样
                total_sample = list_subdirs(data_dir)
                if not total_sample:
                    continue
                
                # 选择样本
                if len(total_sample) <= num_samples:
                    selected_sample = total_sample
                    logging.info(f"select total sample (sample <= {num_samples})")
                else:
                    selected_sample = random.sample(total_sample, num_samples)
                    logging.info(f'select {len(selected_sample)} / {len(total_sample)}')
                
                out_single_date = single_date
                date_copied = 0
                date_failed = 0

                for st in selected_sample:
                    source_dir_path = os.path.join(data_dir, st)
                    target_dir_path = os.path.join(target_output_dir, robot_id, out_single_date, "data", st)
                    ok = copytree_safe(source_dir_path, target_dir_path)
                    if ok:
                        date_copied += 1
                        total_copied += 1
                    else:
                        date_failed += 1
                        total_failed += 1

                logging.info(f'successfully copied in {robot_id} / {out_single_date}: {date_copied}')
                logging.info(f'unsuccessfully copied in {robot_id} / {out_single_date}: {date_failed}')
            
            logging.info(f'Total copied in {robot_id}: {total_copied}, Total failed: {total_failed}')


if __name__ =="__main__":

    parser = argparse.ArgumentParser(
    description="从output目录下的每个日期文件夹的data目录中随机抽样"
    )
    parser.add_argument('--data_input_root_dir',type=str , 
    default='',required = True ,help = '输入提取数据的目录')

    parser.add_argument('--target_output_dir',type=str,
    default='' ,required=True , help='输出随机抽样的目录')

    parser.add_argument('--num_samples',type=int,
    default=10,help='默认选择抽样文件的数量为10'
    )

    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    if (args.target_output_dir is not None )and (args.target_output_dir != ''):
        os.makedirs(args.target_output_dir , exist_ok = True)

        random_smaple(args.data_input_root_dir , args.target_output_dir, args.num_samples)