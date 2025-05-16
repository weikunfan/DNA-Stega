import copy
import os
import numpy
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import kl
import pandas as pd
from pandas import DataFrame,Series
from logger import Logger
import cg_tm_kl
import datetime
def get_key (dict, value):
    return [k for k, v in dict.items() if v == value]

def dict_slice(adict, start, end):
    keys = list(adict.keys())
    dict_slice = {}

    for k in keys[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice

def main(WriteFile,OriginalFile,miniorigal,len_ori,devide_):
    log_file = WriteFile
    logger = Logger(log_file)

    dict_bit = {}
    dict_bit['00000'] = 0.0220
    dict_bit['00001'] = 0.0399
    dict_bit['00010'] = 0.0305
    dict_bit['00011'] = 0.0269
    dict_bit['00100'] = 0.0301
    dict_bit['00101'] = 0.0350
    dict_bit['00110'] = 0.0332
    dict_bit['00111'] = 0.0292
    dict_bit['01000'] = 0.0292
    dict_bit['01001'] = 0.0332
    dict_bit['01010'] = 0.0355
    dict_bit['01011'] = 0.0355
    dict_bit['01100'] = 0.0292
    dict_bit['01101'] = 0.0287
    dict_bit['01110'] = 0.0332
    dict_bit['01111'] = 0.0373

    dict_bit['10000'] = 0.0269
    dict_bit['10001'] = 0.0278
    dict_bit['10010'] = 0.0215
    dict_bit['10011'] = 0.0323
    dict_bit['10100'] = 0.0301
    dict_bit['10101'] = 0.0323
    dict_bit['10110'] = 0.0256
    dict_bit['10111'] = 0.0310
    dict_bit['11000'] = 0.0359
    dict_bit['11001'] = 0.0314
    dict_bit['11010'] = 0.0319
    dict_bit['11011'] = 0.0287
    dict_bit['11100'] = 0.0431
    dict_bit['11101'] = 0.0265
    dict_bit['11110'] = 0.0328
    dict_bit['11111'] = 0.0337

    dict_bit_sor = sorted(dict_bit.items(), key=lambda item: item[1], reverse=True)
    # print(dict_bit_sor)

    ALL = ['A', 'C', 'G', 'T']
    ALL_BASE = []
    for x in range(4):
        for y in range(4):
            for z in range(4):
                BASE = ALL[x] + ALL[y] + ALL[z]
                if BASE not in ALL_BASE:
                    ALL_BASE.append(BASE)

    # with open(r"D:\Destop\seqs\660kb_3300\read_in_5\read_5.txt",'r') as f:
    #     lines_ = f.readlines()
    dp = OriginalFile
    lines_ = cg_tm_kl.txt_process_sc_duo(dp,len_sc=198,beg_sc=0,end_sc=len_ori,PADDING=False,flex=0,num1=devide_)
    lines = ''
    for line in lines_:
        lines += line
    # num = 0
    CHART = []
    for i in range(0, len(lines),3):
        Chart = lines[i : i+3]
        CHART.append(Chart)

    dict1 = {}
    dict2 = {}
    for key in ALL_BASE:
        dict1[key] = dict1.get(key, 0)

    for key in CHART:
        dict2[key] = dict2.get(key, 0) + 1

    for key in ALL_BASE:
        dict1[key] = dict2.get(key, 0)

    # print(dict1)
    # print(dict2)
    value_sum = 0
    for key in dict1:
        value_sum += dict1.get(key, 0)
    num = []
    for key in dict1:
        dict1[key] = dict1.get(key, 0) / value_sum
        num.append(dict1[key])
    # =====#
    final_dict = {key: dict1[key] for key in dict1 if key not in ['ATG', 'TTA', 'TAG', 'TGA']}
    temp_fal = 0
    for key ,value in final_dict.items():
        temp_fal += value
    guiyihua_final_dict = {}
    for key,value in final_dict.items():
        guiyihua_final_dict[key] = value / temp_fal

    # 统计目标非编码区中的密码子出现概率，并且按照从高到低的顺序排序

    # suiji = np.random.randint(0,60,size=32)
    minum = miniorigal
    cunchu = []
    ###
    '''
    #   归一化密码子概率
    dict_bit_sorted_d ={}
    dict_bit_sorted = sorted(dict_bit.items(), key=lambda x: x[1], reverse=True)
    for k ,v in dict_bit_sorted:
        dict_bit_sorted_d[k] = v

    BaseProb = copy.copy(final_dict)
    n = 2
    for i in range(int(32 / n)): #做范围为n的局部贪婪算法
        starttime = datetime.datetime.now()
        dict_bit_SplitN = list(dict_slice(dict_bit_sorted_d,i*n,(i+1)*n).items())
        num_ = 0
        rep = 0
        BaseInDEX = []
        while rep < int(math.factorial(60) / math.factorial(60-n)):
            final_dict_temp = copy.copy(final_dict)

            for j in range(n):
                #####

                ####
                RegionalPick = random.sample(final_dict_temp.items(), 1)

                BaseInDEX.append(RegionalPick[0][0])
                num_ += (RegionalPick[0][1] - dict_bit_SplitN[j][1]) ** 2

                final_dict_temp.pop(RegionalPick[0][0])

            dist = num_ ** 0.5

            if minum > dist:
                minum = dist
                # cunchu.append(BaseInDEX)

            rep += 1

        BaseFinal = BaseInDEX[len(BaseInDEX)-n:]

        cunchu.append(BaseFinal)

        for key in BaseFinal:
            final_dict.pop(key)

        endtime = datetime.datetime.now()
        TimeAll = (endtime-starttime) * int(60 / n)
    '''
    # 寻找最优解
    minum = float('inf')
    best_solution = None
    
    for i in range(500000):
        num_ = 0
        temp = 0
        guiyihua_list = {}
        random_list = random.sample(final_dict.items(), 32)

        for i in range(len(random_list)):
            temp += random_list[i][1]

        for i in range(len(random_list)):
            guiyihua_list[random_list[i][0]] = random_list[i][1] / temp

        random_list_paixu = sorted(guiyihua_list.items(), key=lambda x: x[1], reverse=True)

        for j in range(32):
            num_ += (random_list_paixu[j][1] - dict_bit_sor[j][1]) ** 2

        dist = num_ ** 0.5

        if minum > dist:
            minum = dist
            best_solution = random_list_paixu  # 保存最优解
    
    # 使用最优解
    map_base_bit = {}
    base_shiyong = []
    if best_solution:
        for i in range(32):
            base_shiyong.append(best_solution[i][0])
            base = best_solution[i][0]
            map_base_bit[base] = dict_bit_sor[i][0]
    else:
        logger.error("No valid solution found")
        return miniorigal
    ######


    bits_num = 168

    # 对二进制码进行后处理
    random_bit = []
    for j in range(15000):
        bit = ''
        for i in range(bits_num):
            temp = random.randint(0, 1)
            bit += str(temp)
        bit += '00'

        random_bit.append(bit)
    bpn = []

    for line_bits in random_bit:
        BIT = []
        shuzhi = []
        dict_bit = {}
        dict_bit_shuju = {}
        chazhi = {}
        base = []
        for i in range(0, len(line_bits) - 1, 5):
            bit = line_bits[i:i + 5]
            BIT.append(bit)
        for key in BIT:
            dict_bit[key] = dict_bit.get(key,0) + 1
            dict_bit_shuju[key] = (dict_bit.get(key, 0) + 1) / len(BIT)

        for key ,value in dict_bit.items():
            key_base = get_key(map_base_bit,key)
            k = key_base[0]

            for key1 ,value1 in best_solution[:]:
                if key1 == k:
                    value2 = dict_bit_shuju[key]
                    chazhi[key1] = np.abs(value2 - value1)
                    base.append(key1)
                    shuzhi.append(np.abs(value2 - value1))
                    break

        shuju_key,value_key = [],[]
        for key , value in dict_bit.items():
            shuju_key.append(key)
            value_key.append(value)

        data = { ' 1' : shuju_key,
                 ' 2' : value_key,
                 ' 3' : base,
                 ' 4' : shuzhi

        }

        df = DataFrame(data)

        #print(df)

        p = df.loc[:,' 4'].min()

        q = df[df[' 4'] == p].index.tolist()

        bizhi_1 = df.iloc[q[0],1]
        bizhi_2 = df.iloc[q[0],3]

        dic_buzu = {}
        for key1, value1 in guiyihua_final_dict.items():
            if key1 in base_shiyong:
                continue
            else:
                dic_buzu[key1] = value1
        dic_buzu1 = {}
        for key,value in dic_buzu.items():
            dic_buzu1[key] = (bizhi_1 / bizhi_2) * value

        for key,value in dic_buzu.items():
            dic_buzu[key] = round((bizhi_1 / bizhi_2) * value)

        dic_buzu = sorted(dic_buzu.items(),key = lambda x : x[1],reverse= True)
        buzu_base = []
        for key,value in dic_buzu:
            for i in range(value):
                buzu_base.append(key)

        map_base_bit.items()
        yuanxinxi_base = []
        map_bitstobase = dict(zip(map_base_bit.values(),map_base_bit.keys()))
        for bits in BIT:
            yuanxinxi_base.append(map_bitstobase[bits])

        for base in buzu_base:
            choice = random.choice(range(0,len(yuanxinxi_base)))
            yuanxinxi_base.insert(choice,base)

        fanal_ = ''.join(yuanxinxi_base)
        bpn.append(bits_num / len(fanal_))
        logger.info("fal: {}".format(fanal_))

    mean_bpn = np.mean(np.array(bpn))
    logger.info("bpn: {}".format(mean_bpn))
    
    return minoriginal, mean_bpn  # 返回两个值

def count_lines(file_path):
    """统计文件的实际行数"""
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)
    
if __name__ == '__main__':
    file_names = ['ASM949793v1']
    # file_names = ['ASM141792v1', 'ASM286374v1', 'ASM400647v1', 'ASM949793v1']
    
    # 定义不同len_sc对应的devide值
    params = {
        # 198: [3, 6],
        # 198: [6],
        200: [5]
        # 200: [4, 5]
    }
    
    for file_name in file_names:
        # 创建结果文件路径
        results_file = os.path.join(f'{file_name}_results_CCRS.csv')
        
        # 如果结果文件已存在，读取它作为基础
        if os.path.exists(results_file):
            all_results = pd.read_csv(results_file)
        else:
            all_results = pd.DataFrame(columns=['stego_file', 'cgb', 'Tmb', 'kl', 'bpn'])
            
        for len_sc, devide_list in params.items():
            for devide in devide_list:
                print(f"\nProcessing {file_name} with len_sc={len_sc}, devide={devide}")
                
                output_dir = f'/home/fan/Code/VAE_Synthetic_Steganography/ExperimentData/{file_name}/{len_sc}'
                os.makedirs(output_dir, exist_ok=True)
                
                OriginalFile = f'/home/fan/Code/VAE_Synthetic_Steganography/0_Data/{file_name}/{file_name}_{len_sc}_{devide}.txt'
                
                # 检查原始文件是否存在
                if not os.path.exists(OriginalFile):
                    print(f"Warning: {OriginalFile} does not exist, skipping...")
                    continue
                    
                len_ori = count_lines(OriginalFile)
                minoriginal = 1000
                
                for i in range(1,4):
                    # WriteFile = f'CCRS_{len_sc}_{devide}_{i}.txt'
                    WriteFile = f'{output_dir}/CCRS_{len_sc}_{devide}_{i}.txt'
                    
                    minoriginal, mean_bpn = main(WriteFile, OriginalFile, minoriginal, len_ori, devide)
                    WriteLine = cg_tm_kl.txt_process_sc_duo(WriteFile, len_sc=len_sc, beg_sc=0, end_sc=len_ori, PADDING=False, flex=0, num1=devide)
                    
                    with open(WriteFile, 'w') as f:
                        for line in WriteLine:
                            f.write(line)
                            f.write('\n')
                    print('write done')
                    
                    OriginalLine = cg_tm_kl.txt_process_sc_duo(OriginalFile, len_sc=len_sc, beg_sc=0, end_sc=len_ori, PADDING=False, flex=0, num1=devide)
                    WriteLine = open(WriteFile).readlines()
                    
                    # 确保序列格式正确
                    def clean_sequence(seq):
                        if isinstance(seq, str):
                            return seq.strip()
                        return seq
                    
                    OriginalLine = [clean_sequence(seq) for seq in OriginalLine]
                    WriteLine = [clean_sequence(seq) for seq in WriteLine]
                    
                    Tmb = cg_tm_kl.Tmb(OriginalLine, WriteLine, len_sc=len_sc)
                    CGb = cg_tm_kl.CG_b(OriginalLine, WriteLine, len_sc=len_sc)
                    kll = kl.KL_(WriteLine,OriginalLine)
                    filename_only = os.path.basename(WriteFile)

                    # 创建新的结果行
                    new_row = pd.DataFrame([{
                        'stego_file': filename_only,
                        'cgb': CGb, 
                        'Tmb': Tmb, 
                        'kl': kll,
                        'bpn': mean_bpn
                    }])
                    
                    # 将新结果添加到总结果中
                    all_results = pd.concat([all_results, new_row], ignore_index=True)
                    
                    # 立即保存所有结果到CSV文件
                    all_results.to_csv(results_file, index=False)
                    
                    print(f'File: {WriteFile}')
                    print(f'CGb: {CGb}, Tmb: {Tmb}')
                    print(f'Results saved to: {results_file}')
                
                print("\nCurrent Results:")
                print(all_results)
