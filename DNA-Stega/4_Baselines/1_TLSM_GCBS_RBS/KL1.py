import numpy
import numpy as np
import pandas as pd
import scipy.stats
import random
import collections
import math
import os
import csv
import pandas
# 参数
import cg_tm_kl
import warnings

# 禁用 FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
len_ori = 48
len_sc = 48

beg_sc = 0
end_sc = 48

beg_ori = 0
end_ori = 5000
dictComplement = {}

dictComplement['A'] = 'T'
dictComplement['T'] = 'A'
dictComplement['C'] = 'G'
dictComplement['G'] = 'C'

def txt_process_sc(lines):
    ALL = []

    temp_out = ''
    for line in lines:
        temp = ''
        temp1 = ''
        for i in range(len(line)):
            if line[i] == 'A' or line[i] == 'T' or line[i] == 'C' or line[i] == 'G':
                temp += line[i]

        # temp = temp[:len_sc]
        # temp_out += temp
        # if len(temp) == len_sc:
        #    for i in range(0, len(temp) - 1, 2):
        #        temp1 += temp[i] + temp[i + 1] + ' '

        #    ALL.append(temp1)


        temp_out += temp
        for i in range(0, len(temp) - 1, 2):
            temp1 += temp[i] + temp[i + 1] + ' '

        ALL.append(temp1)

    # ALL = ALL[beg_sc:end_sc]
    return ALL, temp_out


def txt_process(lines):
    ALL = []
    num = []
    # numpy.random.shuffle(lines)
    temp_out = ''
    for line in lines:
        temp = ''
        temp1 = ''

        for i in range(len(line)):
            if line[i] == 'A' or line[i] == 'T' or line[i] == 'C' or line[i] == 'G':
                temp += line[i]
        # len1 = len(temp)

        # num.append(len1)
        # most_num = np.argmax(np.bincount(num))
        # min_num = np.min(num)

        # for i in range(0, len(temp) - 1, 2):
        #   temp1 += temp[i] + temp[i + 1] + ' '

        # ALL.append(temp1)

        temp = temp[:len_ori]
        temp_out += temp
        if len(temp) == len_ori:
            for i in range(0, len(temp) - 1, 2):
                temp1 += temp[i] + temp[i + 1] + ' '

            ALL.append(temp1)

    ALL = ALL[beg_ori:end_ori]

    return ALL, temp_out


def str_to_list(lines):
    out = []
    for line in lines:
        line = line.split(" ")
        out += line

    return out



def pxy_doubleseq(line, sigle, two_base):
    BaseX = ['A', 'T', 'C', 'G']
    BaseY = ['A', 'T', 'C', 'G']
    BaseMartix = []
    DicSingleBaseNum = {}
    DicTwoBaseNum = {}
    DicP = {}

    for x in BaseX:
        for y in BaseY:
            bases = x + y
            BaseMartix.append(bases)

    for base, num in sigle:
        DicSingleBaseNum[base] = num

    for bases, num in two_base:
        DicTwoBaseNum[bases] = num

    for bases in BaseMartix:
        FirstBase = bases[0]
        SecondBase = bases[1]
        FirstBaseComplement = dictComplement[FirstBase]
        SecondBaseComplement = dictComplement[SecondBase]
        basesComplement = str(FirstBaseComplement + SecondBaseComplement)
        fXYDoubleseq = 0.5 * (DicTwoBaseNum[bases] + DicTwoBaseNum[basesComplement])
        fXDoubleseq  = 0.5 * (DicSingleBaseNum[FirstBase] + DicSingleBaseNum[FirstBaseComplement])
        fYDoubleseq  = 0.5 * (DicSingleBaseNum[SecondBase] + DicSingleBaseNum[SecondBaseComplement])

        PXY = (fXYDoubleseq * 2 * len(line)) / (fXDoubleseq * fYDoubleseq)

        DicP[bases] = PXY

    return DicP


def pxy(line, sigle, two_base):
    BaseX = ['A','T','C','G']
    BaseY = ['A', 'T', 'C', 'G']
    BaseMartix = []
    DicSingleBaseNum = {}
    DicTwoBaseNum = {}
    DicP = {}

    for x in BaseX:
        for y in BaseY:
            bases = x + y
            BaseMartix.append(bases)

    for base,num in sigle:
        DicSingleBaseNum[base] = num

    for bases , num in two_base:
        DicTwoBaseNum[bases] = num

    for bases in BaseMartix:
        FirstBase  = bases[0]
        SecondBase = bases[1]
        PXY = (DicTwoBaseNum[bases] * 2*len(line)) / (  DicSingleBaseNum[FirstBase] * DicSingleBaseNum[SecondBase])
        DicP[bases] = PXY

    return DicP



def KL(DICsc,DICori):
    sc = []
    ori = []
    for bases, Pxy in DICsc.items():
        sc.append(Pxy)

    for bases, Pxy in DICori.items():
        ori.append(Pxy)

    ori = ori / np.sum(ori)
    sc = sc / np.sum(sc)
    ZipScore = list(zip(sc,ori))

    KLD = 0
    KLD1 = 0
    for ScScore, OriScore in ZipScore:
        TEMP = OriScore / ScScore
        KLD += -( ScScore * math.log(OriScore / ScScore,math.e ) )
        KLD1 += ScScore * np.log( ScScore / OriScore )

    OUT = scipy.stats.entropy(ori ,sc)
    '''
    x = [0.14285714, 0.04761905, 0.15873016, 0.07936508, 0.15873016, 0.06349206,0.11111111, 0.0952381,  0.12698413, 0.01587302]
    y = [0.0952381 , 0.07936508, 0.15873016, 0.01587302, 0.11111111, 0.14285714, 0.14285714, 0.0952381,  0.03174603, 0.12698413]
    ZIPxy = list(zip(x,y))
    KLD = 0
    KLD1 = 0
    for ScScore, OriScore in ZIPxy:
        TEMP = OriScore / ScScore
        KLD += -(ScScore * math.log(OriScore / ScScore, math.e))
        KLD1 += ScScore * np.log(ScScore / OriScore)
    OUT = scipy.stats.entropy(x, y)
    '''
    return OUT

def SequenceComplement(lines):
    lines_list, line_str = txt_process_sc(lines)

    #line_sc = str_to_list(lines_list) # 原始生成数据，需要进行处理ori_two.txt

    #line_sc = line_sc[: len(line_sc) - 1]

    # str_temp_ori = list(''.join(line_ori))

    line_str_tolist = list(line_str)



    linecomplement = []

    for character in line_str_tolist:
        if len(character) == 1:
            complement = dictComplement[character]
            linecomplement.append(complement)
        else:
            complement0 = dictComplement[character[0]]
            complement1 = dictComplement[character[1]]
            temp = complement0 + complement1
            linecomplement.append(temp)

    linecomplement = ''.join(linecomplement)
    return linecomplement,line_str

def KLDoubleStrand(line_sc,line_ori):
    line_sc_complement,line_sc = SequenceComplement(line_sc)

    line_sc_doublesequence = line_sc + line_sc_complement

    temp = []
    temp.append(line_sc_doublesequence)

    line_sc_doublesplit, _ = txt_process_sc(temp)
    line_sc_doublesplit_list = line_sc_doublesplit[0].split(' ')[ : -1]

    str_temp_sc = list(_)

    singlebase = sorted(collections.Counter(str_temp_sc).items(), key=lambda x: x[1], reverse=True)

    word_distribution_sc = sorted(collections.Counter(line_sc_doublesplit_list).items(), key=lambda x: x[1],
                                  reverse=True)  # 获得文件中各个单词的分布

    sc_pxy = pxy_doubleseq(str_temp_sc, singlebase, word_distribution_sc)

    ############
    line_ori_complement, line_ori = SequenceComplement(line_ori)

    line_ori_doublesequence = line_ori + line_ori_complement

    temp = []
    temp.append(line_ori_doublesequence)

    line_ori_doublesplit, __ = txt_process_sc(temp)
    line_ori_doublesplit_list = line_ori_doublesplit[0].split(' ')[: -1]

    str_temp_ori = list(__)

    singlebase = sorted(collections.Counter(str_temp_ori).items(), key=lambda x: x[1], reverse=True)

    word_distribution_ori = sorted(collections.Counter(line_ori_doublesplit_list).items(), key=lambda x: x[1],
                                  reverse=True)  # 获得文件中各个单词的分布

    ori_pxy = pxy_doubleseq(str_temp_ori, singlebase, word_distribution_ori)

    kl_ = KL(sc_pxy, ori_pxy)

    return kl_

def KL_(line_sc, line_ori):
    line_sc_, all_sc = txt_process_sc(line_sc)
    line_ori_, all_ori = txt_process(line_ori)

    line_sc = str_to_list(line_sc_)  # 原始生成数据，需要进行处理ori_two.txt
    line_ori = str_to_list(line_ori_)


    line_sc = line_sc[ : len(line_sc)-1]
    line_ori = line_ori[ : len(line_ori)-1]
    str_temp_sc = list(''.join(line_sc))
    singlebase = sorted(collections.Counter(str_temp_sc).items(), key=lambda x: x[1], reverse=True)

    word_distribution_sc = sorted(collections.Counter(line_sc).items(), key=lambda x: x[1],
                                  reverse=True)  # 获得文件中各个单词的分布

    sc_pxy = pxy(str_temp_sc, singlebase, word_distribution_sc)

    str_temp_ori = list(''.join(line_ori))

    singlebase = sorted(collections.Counter(str_temp_ori).items(), key=lambda x: x[1], reverse=True)

    word_distribution_ori = sorted(collections.Counter(line_ori).items(), key=lambda x: x[1],
                                   reverse=True)  # 获得文件中各个单词的分布

    ori_pxy = pxy(str_temp_ori, singlebase, word_distribution_ori)

    kl_ = KL(sc_pxy,ori_pxy)

    return kl_

def count_lines(file_path):
    """统计文件的实际行数"""
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)

if __name__ == '__main__':
    # 文件名列表
    # files_name = ['ASM141792v1']
    files_name = ['ASM141792v1', 'ASM286374v1', 'ASM400647v1', 'ASM949793v1', 'ASM1821919v1']
    read_files = []
    file_line_counts = {}  # 用于记录每个文件的行数
    # pd_kl = pd.DataFrame(columns=['name', 'kl', 'klds'])

    for base_name in files_name:
        # 原始文件所在目录
        base_dir = f'/home/fan/Code/VAE_Synthetic_Steganography/0_Data/{base_name}'
        pd_kl_dict = {}  # 用于每个 prefix 独立的 DataFrame 存储

        for file in os.listdir(base_dir):
            if file.endswith('.txt'):
                # 处理每个原始文件的路径
                path_ori = os.path.join(base_dir, file)
                read_files.append(path_ori)  # 记录文件名
                
                # 提取分割参数
                try:
                    length_str = file.split('_')[1]
                    devide_num = int(file.split('_')[2].split('.')[0])
                    
                    # 动态设置 len_sc
                    len_sc = 198 if length_str == '198' else 200
                except (IndexError, ValueError):
                    raise ValueError(f"文件名 {file} 不符合预期格式，无法提取长度和分割参数。")

                # 获取文件的实际行数，作为处理长度
                actual_lines = count_lines(path_ori)
                file_line_counts[path_ori] = actual_lines  # 记录文件的行数
                end_sc = actual_lines

                # 输出每个文件的行数和读取的内容
                # print(f"Processing original file: {path_ori}")
                # print(f"Actual lines in file: {actual_lines}")
                # print("Content sample (first 5 lines):")
                # with open(path_ori, 'r') as f:
                #     for _ in range(5):
                #         # print(f.readline().strip())

                # 获取原始文件内容
                raw_ori = cg_tm_kl.txt_process_sc_duo(path_ori, len_sc=len_sc, beg_sc=0, end_sc=end_sc, PADDING=False, flex=0, devide_num=devide_num)

                # 获取对比文件路径
                # dirsSc = f'/home/fan/Code/VAE_Synthetic_Steganography/ExperimentData/{base_name}/{len_sc}/LSTM'
                # dirsSc = f'/home/fan/Code/VAE_Synthetic_Steganography/ExperimentData/{base_name}/{len_sc}/VAE/lr_0.0003'
                dirsSc = f'/home/fan/Code/VAE_Synthetic_Steganography/ExperimentData/{base_name}/{len_sc}/VAE/discop'
                PathSc = [os.path.join(dirsSc, sc_file) for sc_file in os.listdir(dirsSc) if sc_file.endswith('.txt')]
                # print(f"Files found in {dirsSc}: {PathSc}")
                for p in PathSc:

                    filename = os.path.splitext(os.path.basename(p))[0]
                    # 获取文件的前缀并创建对应的保存目录
                    prefix = filename.split('_')[0]
                    prefix_len_sc = f"{prefix}_{len_sc}"

                    if prefix_len_sc not in pd_kl_dict:
                        pd_kl_dict[prefix_len_sc] = pd.DataFrame(columns=['name', 'kl', 'klds'])


                    # 创建对应的保存目录
                    save_dir = os.path.join(
                        '/home/fan/Code/VAE_Synthetic_Steganography/ExperimentData/file_evaluate',
                        base_name,
                        str(len_sc),
                        prefix
                    )
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # 读取对比文件数据，动态设置 `len_sc` 和 `devide_num`
                    length_str_sc = filename.split('_')[1]
                    devide_num_sc = int(filename.split('_')[2])
                    len_sc = 198 if length_str_sc == '198' else 200

                    raw_sc = cg_tm_kl.txt_process_sc_duo(p, len_sc=len_sc, beg_sc=0, end_sc=end_sc, PADDING=False, flex=0, devide_num=devide_num_sc)
                    
                    # 计算 KL 散度
                    kl = KL_(raw_sc, raw_ori)
                    klds = KLDoubleStrand(raw_sc, raw_ori)
                    # 记录 KL 散度结果
                    # 如果 prefix 的 DataFrame 是空的，则直接赋值而不是拼接
                    # 如果这个前缀的 DataFrame 还不存在，就创建一个
                    if prefix not in pd_kl_dict:
                        pd_kl_dict[prefix] = pd.DataFrame(columns=['name', 'kl', 'klds'])

                    # 创建新的行作为字典
                    new_row = {'name': filename, 'kl': kl, 'klds': klds}
                    # 检查 new_row 中是否有空值或无效列，去除它们
                    new_row_cleaned = {key: value for key, value in new_row.items() if pd.notna(value)}
                    # 如果清理后的 new_row 有变化，打印出相关信息
                    if len(new_row) != len(new_row_cleaned):
                        print(f"Original row (with possible NA values): {new_row}")
                        print(f"Cleaned row (after removing NA values): {new_row_cleaned}")
                    # 检查这行数据是否已经存在（通过匹配 'name'、'kl' 和 'klds'）
                    if not ((pd_kl_dict[prefix]['name'] == filename) & 
                            (pd_kl_dict[prefix]['kl'] == kl) & 
                            (pd_kl_dict[prefix]['klds'] == klds)).any():
                        # 如果不存在，才追加到 DataFrame 中
                        pd_kl_dict[prefix] = pd.concat([pd_kl_dict[prefix], pd.DataFrame([new_row_cleaned])], ignore_index=True)

                    # 保存结果到 CSV 文件
                    csv_path = os.path.join(save_dir, 'kl_2.csv')
                    pd_kl_dict[prefix].to_csv(csv_path, index=False)
        # 每处理完一个 base_name，输出状态并清空 pd_kl_dict
        print(f"完成 {base_name} 的 KL 散度计算，结果已保存。")
    
    print("KL 散度计算完成，结果已保存。")
    print("\n每个文件的实际行数:")
    for file_path, line_count in file_line_counts.items():
        print(f"{file_path}: {line_count} 行")
# if __name__ == '__main__':
#     # path_ori = r'D:\Destop\seqs\888kb_ASM400647v1\read_4\OriginalData\read_4.txt'
#     path_ori = r'/home/fan/Code/DNA-Synthetic-Steganography-Based-on-Conditional-Probability-Adaptive-Coding-main/4_Baselines/ASM141792v1_200_4.txt'
#     if os.path.exists(path_ori):
#         print('文件存在')
#     else:
#         print('文件不存在')
#     # with open(path_ori, "r") as f2:
#     #    line_ori = f2.readlines()
#     PathSc = []
#     dirsSc = r'/home/fan/Code/DNA-Synthetic-Steganography-Based-on-Conditional-Probability-Adaptive-Coding-main/ExperimentData/18/read_4/file_evaluate'
#     # dirsSc = r'/home/fan/Code/DNA-Synthetic-Steganography-Based-on-Conditional-Probability-Adaptive-Coding-main/ExperimentData/vae/read_3/file_evaluate'
#     # dirsSc = r'D:\Destop\seqs\888kb_ASM400647v1\ss_read_4\ss'
#     for root, dirs, files in os.walk(dirsSc):
#         for file in files:
#             PathSc.append(os.path.join(root, file))
#             print('hello1')
#     raw_ori = cg_tm_kl.txt_process_sc_duo(path_ori,len_sc=200,beg_sc=0,end_sc=1000,PADDING=False,flex=0,devide_num=4)
#     # raw_ori = cg_tm_kl.txt_process_sc_duo(path_ori,len_sc=198,beg_sc=0,end_sc=1000,PADDING=False,flex=0,devide_num=3)
#     pd_kl = pd.DataFrame(columns=['name','kl','klds'])
#     for p in PathSc:
#         # with open(p, "r") as f1:
#         #   line_sc = f1.readlines()
#         filename = os.path.splitext(os.path.basename(p))[0]
#         # 创建新的文件夹路径
#         save_dir = os.path.join(
#             '/home/fan/Code/DNA-Synthetic-Steganography-Based-on-Conditional-Probability-Adaptive-Coding-main/ExperimentData/18/read_4',
#             filename
#         )
#         os.makedirs(save_dir, exist_ok=True)
#         raw_sc = cg_tm_kl.txt_process_sc_duo(p, len_sc=200, beg_sc=0, end_sc=1000, PADDING=False, flex=0,
#                                               devide_num=4)
#         # raw_sc = cg_tm_kl.txt_process_sc_duo(p, len_sc=198, beg_sc=0, end_sc=1000, PADDING=False, flex=0,
#         #                                       devide_num=3)
#         kl = KL_(raw_sc, raw_ori)
#         klds = KLDoubleStrand(raw_sc, raw_ori)
#         # pd_kl = pd_kl.append({'name':p[ p.rfind('\\') + 1 : -4], 'kl':kl,'klds':klds },ignore_index=True)
#         pd_kl = pd.concat([pd_kl,pd.DataFrame({'name':[filename], 'kl':[kl],'klds':[klds]})],ignore_index=True)
#         print(pd_kl)
#             # 文件路径
#         csv_path = os.path.join(save_dir, 'kl_2.csv')
#         print('hello2')
#         pd_kl.to_csv(csv_path,index=False)
#     # pd_kl.to_csv(r'/home/fan/Code/DNA-Synthetic-Steganography-Based-on-Conditional-Probability-Adaptive-Coding-main/ExperimentData/18/read_4/ss/kl_2.csv')
