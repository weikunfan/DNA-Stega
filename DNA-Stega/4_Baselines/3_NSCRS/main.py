import FindStartCondon
import random
from cg_tm_kl import txt_process_sc_FromKl
import os

def AdjustCandidatePoll_Hutong(GeneratedBase,CandidatePoll):
    AdjustedPoll  = []
    for Candidate in CandidatePoll:
        if len(GeneratedBase) < 2:
            AdjustedPoll.append(Candidate)
        else:
            CandidateTemp = GeneratedBase[len(GeneratedBase)-2 : ] + Candidate
            if FindStartCondon.FindStartCondon(CandidateTemp) == 0:
                AdjustedPoll.append(Candidate)

    return AdjustedPoll

def adjust_poll(x,vocabulary):
    #x为以生成的序列
    candidate_poll = ['A', 'T', 'C', 'G']
    if len(x) == 0:
        GeneratedBase = ''
    else:
        GeneratedBase = x[-1]

    Adjusted_poll = FindStartCondon.AdjustCandidatePoll(GeneratedBase=GeneratedBase,
                                                        CandidatePoll=candidate_poll)

    return Adjusted_poll

def hutong_baseline(single_line_num,total_num):
    #加载比特流
    with open(r'/home/fan/Code/VAE_Synthetic_Steganography/bit_stream/bit_stream.txt','r') as f1:
        bits = f1.readlines()

    bits = list(bits[0])
    random.shuffle(bits)
    bits_len = len(bits)
    
    #候选碱基池
    candidate_poll = ['A','T','C','G']
    Generated_Base_Line = []
    i = 0
    bits_used = 0  # 统计使用的比特数
    while( len(Generated_Base_Line) < total_num):
        GeneratedBase = ''
        while( len(GeneratedBase) < single_line_num):
            # 碱基&二进制编码对应字典
            dict_base_bits = {}

            Adjusted_poll = AdjustCandidatePoll_Hutong(GeneratedBase=GeneratedBase, CandidatePoll=candidate_poll)
            if len(Adjusted_poll) == 4:
                dict_base_bits['00'] = 'A'
                dict_base_bits['01'] = 'T'
                dict_base_bits['10'] = 'C'
                dict_base_bits['11'] = 'G'
                # 使用循环索引
                bit1 = bits[i % bits_len]
                bit2 = bits[(i + 1) % bits_len]
                GeneratedBase = GeneratedBase + dict_base_bits[str(bit1) + str(bit2)]
                i += 2
                bits_used += 2
            elif len(Adjusted_poll) == 3:
                dict_base_bits['0'] = 'A'
                dict_base_bits['10'] = 'T'
                dict_base_bits['11'] = 'C'
                # 使用循环索引
                bit1 = bits[i % bits_len]
                if str(bit1) == '0':
                    GeneratedBase = GeneratedBase + 'A'
                    i += 1
                    bits_used += 1
                else:
                    bit2 = bits[(i + 1) % bits_len]
                    GeneratedBase = GeneratedBase + dict_base_bits[str(bit1) + str(bit2)]
                    i += 2
                    bits_used += 2
            elif len(Adjusted_poll) == 1:
                dict_base_bits[''] = 'C'
                GeneratedBase = GeneratedBase + 'C'

        Generated_Base_Line.append(GeneratedBase)

    print(f"Total bits used: {bits_used}")
    print(f"Original bit stream length: {bits_len}")
    print(f"Number of times bit stream was cycled: {bits_used // bits_len}")
    return Generated_Base_Line, bits_used

def count_lines(file_path):
    """计算文件的总行数"""
    with open(file_path, 'r') as f:
        line_count = sum(1 for line in f)
    return line_count

if __name__ == '__main__':
    len_sc = 200
    devide = 5
    # file_names = ['ASM141792v1', 'ASM286374v1', 'ASM400647v1', 'ASM949793v1']

    file_name = 'ASM949793v1'
    len_ori = count_lines(f'/home/fan/Code/VAE_Synthetic_Steganography/0_Data/{file_name}/{file_name}_{len_sc}_{devide}.txt')
    output_dir = f'/home/fan/Code/VAE_Synthetic_Steganography/ExperimentData/{file_name}/{len_sc}'
    
    # 创建或打开日志文件
    log_file = f'{file_name}_bits_usage_log.txt'
    # 如果文件不存在，写入表头
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("Filename\tBits Used\n")
    
    for i in range(1, 4):
        print(f"\nGenerating sequence set {i}:")
        ls, bits_used = hutong_baseline(len_sc, len_ori)
        processed_ls, _ = txt_process_sc_FromKl(ls, len_sc, devide)
        output_file = f'{output_dir}/NSCRS_{len_sc}_{devide}_{i}.txt'
        
        # 保存生成的序列
        with open(output_file, 'w') as f:
            for l in processed_ls:
                f.write(l)
                f.write('\n')
        
        # 记录比特使用情况，仅记录文件名
        with open(log_file, 'a') as f:
            filename_only = os.path.basename(output_file)
            f.write(f"{filename_only}\t{bits_used}\n")