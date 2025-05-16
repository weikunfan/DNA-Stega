import numpy
import random
import copy
import math
import numpy as np
#import KL
import cg_tm_kl
import KL1
import os
ALL = ['A','T','C','G']

def count_lines(file_path):
    """统计文件的实际行数"""
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)
    
def random_sub(k,rate):
    k = list(''.join(k.strip().split(' ')))
    index = random.sample(range(0, len(k)), int(math.ceil(rate * len(k))))
    for i in index:
        t = k[i]
        ALL.remove(k[i])
        temp = ALL[random.randint(0,2)]
        k[i] = temp
        ALL.append(t)

    line = ''.join(k)
    return line

def DOUBLESUB(linglist):
    linglist = linglist.strip().split(' ')
    linglist = ''.join(linglist)
    subindex = []
    i = 0
    while  i < len(linglist):
        try:
            if linglist[i] == linglist[i+1]:
                subindex.append(i+1)
                i += 2
            else:
                i += 1
        except:
            break

    linglist = list(linglist)
    for index in subindex:
        linglist[index] = ALL[random.randint(0,3)]

    linglist = ''.join(linglist)
    return linglist


if __name__ == '__main__':
    files_name = ['ASM141792v1', 'ASM286374v1', 'ASM400647v1', 'ASM949793v1', 'ASM1821919v1']
    read_files = []
    file_line_counts = {}  # 用于记录每个文件的行数

    for base_name in files_name:
        base_dir = f'/home/fan/Code/VAE_Synthetic_Steganography/0_Data/{base_name}'

        for file in os.listdir(base_dir):
            if file.endswith('.txt'):
                file_path = os.path.join(base_dir, file)
                read_files.append(file_path)  # 记录文件名
                
                # 提取分割参数
                try:
                    length_str = file.split('_')[1]
                    devide_ = int(file.split('_')[2].split('.')[0])
                    len_sc = 198 if length_str == '198' else 200
                except (IndexError, ValueError):
                    raise ValueError(f"文件名 {file} 不符合预期格式，无法提取长度和分割参数。")

                # 获取文件的实际行数，作为处理长度
                actual_lines = count_lines(file_path)
                file_line_counts[file_path] = actual_lines  # 记录文件的行数
                end_sc = actual_lines
            # 输出每个文件的行数和读取的内容
                # print(f"File: {file_path}")
                # print(f"Actual lines in file: {actual_lines}")
                # print("Content sample (first 5 lines):")
                # with open(file_path, 'r') as f:
                #     for _ in range(5):
                #         print(f.readline().strip())
                # 对DNA序列进行隐写处理
                split_suq = cg_tm_kl.txt_process_sc_duo(file_path, len_sc=len_sc, beg_sc=0, end_sc=end_sc, PADDING=False, flex=0, devide_num=devide_)
                print(f"Processing file: {file_path}, Length: {end_sc}, Divide: {devide_}")
                print(f"Split sequence length: {len(split_suq)}")

                for i in range(1, 4):
                    GCBSOUT = []
                    TLSMOUT = []
                    DOUBLESUNOUT = []

                    for line in split_suq:
                        line_DOU = DOUBLESUB(line)
                        line_GCBS = random_sub(line, 0.50)
                        line_TLSM = random_sub(line, 0.82)

                        GCBSOUT.append(line_GCBS)
                        DOUBLESUNOUT.append(line_DOU)
                        TLSMOUT.append(line_TLSM)

                    # 保存结果文件
                    output_dir = f'/home/fan/Code/VAE_Synthetic_Steganography/ExperimentData/{base_name}/{length_str}'
                    os.makedirs(output_dir, exist_ok=True)

                    # 定义格式化函数，将每行按 `len_sc` 个碱基保存，每 `devide_` 个碱基加一个空格，但不计入空格
                    def format_sequence(sequence, len_sc, devide_):
                        # 截取 len_sc 个碱基
                        sequence = sequence[:len_sc]

                        # 每 `devide_` 个碱基添加一个空格，但不计入碱基总数
                        formatted = ' '.join(sequence[i:i + devide_] for i in range(0, len(sequence), devide_))
                        
                        return formatted
                    # 写入 GCBS 文件
                    write_gcbs = os.path.join(output_dir, f'GCBS_{length_str}_{devide_}_{i}.txt')
                    with open(write_gcbs, 'w') as file2:
                        for line in GCBSOUT:
                            formatted_line = format_sequence(line, len_sc, devide_)
                            file2.write(formatted_line + '\n')

                    # 写入 TLSM 文件
                    write_tlsm = os.path.join(output_dir, f'TLSM_{length_str}_{devide_}_{i}.txt')
                    with open(write_tlsm, 'w') as file3:
                        for line in TLSMOUT:
                            formatted_line = format_sequence(line, len_sc, devide_)
                            file3.write(formatted_line + '\n')

                    # 写入 DOUBLESUB 文件
                    write_double = os.path.join(output_dir, f'DOUBLESUB_{length_str}_{devide_}_{i}.txt')
                    with open(write_double, 'w') as file4:
                        for line in DOUBLESUNOUT:
                            formatted_line = format_sequence(line, len_sc, devide_)
                            file4.write(formatted_line + '\n')

    print("隐写处理完成，所有文件已保存。")
    
    # 输出每个文件的实际行数
    print("\n每个文件的实际行数:")
    for file_path, line_count in file_line_counts.items():
        print(f"{file_path}: {line_count} 行")


    # split_suq = cg_tm_kl.txt_process_sc_duo(path,len_sc=200,beg_sc=0,end_sc=length_,PADDING=False,flex=0,devide_num = devide)


    # for i in range(0,3):
    #     GCBSOUT = []
    #     TLSMOUT = []
    #     DOUBLESUNOUT = []
    #     for line in split_suq:
    #         temp = ''.join(line.strip().split(' '))
    #         line_DOU = DOUBLESUB(line)
    #         line_GCBS = random_sub(line,
    #                                0.50)  # The pseudo-sequence of information encoded by GCBS method is equal in length to the original sequence,
    #         # so it can be considered as random substitution at 50% modification rate
    #         line_TLSM = random_sub(line,
    #                                0.82)  # The BPN of TLSM is 1.64, hence it can be considered as random substitution at 82% modification rate

    #         # line_1_percent_randomsub = random_sub(line, 0.01) #non-specialized methods

    #         # line_5_percent_randomsub = random_sub(line,0.05) #non-specialized methods

    #         # line_10_percent_randomsub = random_sub(line, 0.10) #non-specialized methods

    #         # line_20_percent_randomsub = random_sub(line, 0.20) #non-specialized methods

    #         # OnePercent.append(line_1_percent_randomsub)

    #         # FivePercent.append(line_5_percent_randomsub)

    #         # TenPercent.append(line_10_percent_randomsub)

    #         # TwentyPercnt.append(line_20_percent_randomsub)


    #         GCBSOUT.append(line_GCBS)
    #         DOUBLESUNOUT.append(line_DOU)
    #         TLSMOUT.append(line_TLSM)


    #     # write_gcbs = r'D:\Destop\file\科研相关\论文代码\ExperimentData\{}\baselines\GCBS_{}_{}.txt'.format(mode, mode, str(i))
    #     write_gcbs = r'GCBS_{}_{}.txt'.format(18, mode, str(i))
    #     with open(write_gcbs, 'w') as file2:
    #         for line in GCBSOUT:
    #             file2.write(line)
    #             file2.write('\n')

    #     write_tlsm = r'TLSM_{}-{}.txt'.format(18, mode,str(i))
    #     with open(write_tlsm, 'w') as file3:
    #         for line in TLSMOUT:
    #             file3.write(line)
    #             file3.write('\n')

    #     write = r'DOUBLESUB_{}-{}.txt'.format(18, mode,str(i))
    #     with open(write, 'w') as file4:
    #         for line in DOUBLESUNOUT:
    #             file4.write(line)
    #             file4.write('\n')

        # Files_Names = [write_1,write_5,write_10,write_20]
        # for file_name in Files_Names:
        #     with open(path, 'r') as f1:
        #         linesori = f1.readlines()

        #     with open(file_name, 'r') as f2:
        #         linessc = f2.readlines()

        #     # CGd = cg_tm_kl.CG_b(linesori,linessc,len_sc=200)
        #     # Tmd = cg_tm_kl.Tmb(linesori, linessc, len_sc=200)

        #     # klds = cg_tm_kl.KLDoubleStrand(linessc,linesori)

        #     lineall = list(zip(linesori, linessc))

        #     All = []
        #     for l in lineall:
        #         lineori = list(''.join(l[0].split(' ')))
        #         linesc = list(''.join(l[1].split(' ')))
        #         num = 0
        #         for i in range(len(lineori)):
        #             if lineori[i] != linesc[i]:
        #                 num += 1

        #         All.append(num / len(lineori))

        #     modificationrate = np.mean(np.array(All))

        #     print(modificationrate)
        #     # print('Tmd:',Tmd)
        #     # print('CGD:',CGd)
        #     # print('KLDS',klds)










