import shutil
import torch
from torch import nn
import torch.optim as optim
import scipy.stats
import numpy as np
from logger import Logger
import utils
import lm
import os
import inspect
import Model_training
import remove_model
# import stega
import stega_lstm
import extract_bits
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def claer_dirs(dirs_modes):
    del_list = os.listdir(dirs_modes)
    for f in del_list:
        # print('f:',f)
        file_path = os.path.join(dirs_modes, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
def count_lines(file_path):
    """计算文件的总行数"""
    with open(file_path, 'r') as f:
        line_count = sum(1 for line in f)
    return line_count

if __name__ == '__main__':
    File = []
# 第二个文件少了一些数据
    # file_names = ['ASM141792v1', 'ASM286374v1']  
    file_names = ['ASM400647v1']
    # file_names = ['ASM141792v1', 'ASM286374v1', 'ASM400647v1','ASM949793v1' ,'ASM1821919v1']
    Path_ori_file = '/home/fan/Code/VAE_Synthetic_Steganography/0_Data'  # 原始数据路径
    Path_results_save = '/home/fan/Code/VAE_Synthetic_Steganography/ExperimentData'  # 实验数据保存路径
    Path_model_save = '/home/fan/Code/VAE_Synthetic_Steganography/ExperimentData/LSTM'  # 模型保存路径
    

    # Range = [2,3]  # 重复实验次数
    Range = [1]  # 重复实验次数
    index = ['3']  # SL 值
    # index = ['3', '4', '5', '6']  # SL 值
    Clear_File = False  # 是否清空模型保存文件夹

    for file_name in file_names:
        for ind in index:
            code = 'vaetest' + ind
            if int(ind) % 3 == 0:
                SeqLength = int(198 / int(ind))  # 单个短序列中的基本单元数量
                file_ = os.path.join(Path_ori_file, file_name, f"{file_name}_198_{ind}.txt")  # 原始DNA序列数据集路径
            else:
                SeqLength = int(200 / int(ind))
                file_ = os.path.join(Path_ori_file, file_name, f"{file_name}_200_{ind}.txt")

            num_rows = count_lines(file_)
            for i in Range:
                model_save_dir = os.path.join(Path_model_save, file_name, f"read_{ind}", f"M_{i}")
                os.makedirs(model_save_dir, exist_ok=True)  # 创建模型保存路径
                dirs_modes = model_save_dir


                if Clear_File:
                    if os.path.exists(dirs_modes):
                        shutil.rmtree(dirs_modes)  # 清空目录
                    os.makedirs(dirs_modes)  # 重新创建空目录
                else:
                    os.makedirs(dirs_modes, exist_ok=True)  # 仅创建目录
  
                # print('hello1')
                # Model_training.main(file_, code, SeqLength, ind, Path_model_save, file_name, modelnum=i, lr=0.0003, running_mode='s')
                # # print('hello2')

                # # 删除冗余模型，仅保留损失函数收敛时的模型
                # remove_model.qingchu_Model(model_save_dir)

            # 开始嵌入秘密信息并生成伪序列
                stega_lstm.main_stega(file_name, Ranges=i, index=ind, file_=file_, Path_save=Path_model_save, test=False, coding='adg',num_rows=100)
                # extract_bits.main_extract_bits(file_name, Ranges=Range, index=[ind], file_=file_, Path_save=Path_model_save)
