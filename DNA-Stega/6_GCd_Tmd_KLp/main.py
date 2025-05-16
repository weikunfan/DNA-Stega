import os
import numpy
import random
import copy
import numpy as np
import cg_tm_kl
import pandas as pd

def count_lines(file_path):
    """统计文件的实际行数"""
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)

def get_processed_files(csv_file):
    """获取已经处理过的文件列表"""
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        return set(df['path'].values)
    return set()

if __name__ == '__main__':
    files_name = ['ASM1821919v1']
    # files_name = ['ASM141792v1', 'ASM286374v1', 'ASM400647v1', 'ASM949793v1']

    file_line_counts = {}
    asm_to_gca = {
        'ASM141792v1': 'GCA_001417925',
        'ASM286374v1': 'GCA_002863745',
        'ASM400647v1': 'GCA_004006475',
        'ASM949793v1': 'GCA_009497935',
        'ASM1821919v1': 'GCA_018219195'
    }

    for base_name in files_name:
        # 获取对应的GCA编号
        gca_name = asm_to_gca[base_name]
        
        # 结果文件路径
        writefile = f'/home/fan/Code/VAE_Synthetic_Steganography/results/{gca_name}_mono_di_tri_gcb_tmb_klp.csv'
        
        # 获取已处理的文件列表
        processed_files = get_processed_files(writefile)
        print(f"\n已处理的文件数量：{len(processed_files)}")

        # 初始化结果DataFrame
        result = pd.DataFrame(columns=['path', 'mono', 'di', 'tri', 'GCd', 'Tmd', 'KLp'])
        if os.path.exists(writefile):
            result = pd.read_csv(writefile)

        print(f"\n正在处理：{base_name}")
        base_dir = f'/home/fan/Code/VAE_Synthetic_Steganography/0_Data/{base_name}'

        # 遍历每个文件
        for file in os.listdir(base_dir):
            if file.endswith('.txt'):
                file_path = os.path.join(base_dir, file)
                
                # 提取文件参数
                try:
                    length_str = file.split('_')[1]
                    devide_ = int(file.split('_')[2].split('.')[0])
                    len_sc = 198 if length_str == '198' else 200
                except (IndexError, ValueError):
                    print(f"警告：文件名 {file} 不符合预期格式，跳过处理。")
                    continue

                # 处理原始DNA序列
                actual_lines = count_lines(file_path)
                file_line_counts[file_path] = actual_lines
                end_sc = actual_lines
                print(f"\n处理文件：{file_path}")
                print(f"文件行数：{actual_lines}")

                linesori = cg_tm_kl.txt_process_sc_duo(file_path, len_sc=len_sc, beg_sc=0, end_sc=end_sc, 
                                                     PADDING=False, flex=0, devide_num=devide_)

                # 隐写DNA序列路径列表
                dirs_sc_list = [
                    f'/home/fan/Code/VAE_Synthetic_Steganography/ExperimentData/{base_name}/{len_sc}',
                    f'/home/fan/Code/VAE_Synthetic_Steganography/ExperimentData/{base_name}/{len_sc}/huffman',
                ]
                
                for dirs_sc in dirs_sc_list:
                    if not os.path.exists(dirs_sc):
                        print(f"路径不存在，跳过：{dirs_sc}")
                        continue
                        
                    print(f"\n处理路径：{dirs_sc}")
                    
                    # 遍历目录下的文件
                    for stega_file in os.listdir(dirs_sc):
                        for i in range(1, 4):
                            if stega_file.endswith(f'_{len_sc}_{devide_}_{i}.txt'):
                                full_path = os.path.join(dirs_sc, stega_file)
                                filename = full_path[full_path.rfind('/') + 1: -4]
                                
                                # 检查是否已处理
                                if filename in processed_files:
                                    print(f"文件已处理，跳过：{filename}")
                                    continue
                                    
                                print(f"处理新文件：{filename}")

                                # 处理隐写序列文件
                                linessc = cg_tm_kl.txt_process_sc_duo(full_path, len_sc=len_sc, beg_sc=0, 
                                                                    end_sc=end_sc, PADDING=False, flex=0, 
                                                                    devide_num=devide_)

                                # 计算度量
                                CGd = cg_tm_kl.CG_b(linesori, linessc, len_sc=len_sc)
                                Tmd = cg_tm_kl.Tmb(linesori, linessc, len_sc=len_sc)
                                klds = cg_tm_kl.KLDoubleStrand(linessc, linesori)
                                mono_bias, di_bias, tri_bias = cg_tm_kl.nucleotide_composition_bias(linesori, linessc, len_sc=len_sc)
                                
                                # jsd = cg_tm_kl.JSDDoubleStrand(linessc, linesori)
                                # 添加新结果
                                new_row = pd.DataFrame({
                                    'path': [filename], 
                                    'mono': [mono_bias],
                                    'di': [di_bias],
                                    'tri': [tri_bias],
                                    'GCd': [CGd], 
                                    'Tmd': [Tmd], 
                                    'KLp': [klds],
                                    # 'JSD': [jsd]
                                })
                                result = pd.concat([result, new_row], ignore_index=True)
                                
                                # 即时保存结果
                                result.sort_values(by='path').to_csv(writefile, index=False)
                                print(f"已保存结果：{filename}")

        # 计算并保存统计结果
        stats_writefile = f'/home/fan/Code/VAE_Synthetic_Steganography/results/{gca_name}_stats.csv'
        print(f"\n生成统计结果：{stats_writefile}")
        
        result['base_name'] = result['path'].apply(lambda x: '_'.join(x.split('_')[:-1]))
        stats = []
        for name in result['base_name'].unique():
            group = result[result['base_name'] == name]
            stats_dict = {
                'method': name,
                'mono': f"{group['mono'].mean():.2f}±{group['mono'].std():.3f}",
                'di': f"{group['di'].mean():.2f}±{group['di'].std():.3f}",
                'tri': f"{group['tri'].mean():.2f}±{group['tri'].std():.3f}",
                'GCd': f"{group['GCd'].mean():.2f}±{group['GCd'].std():.3f}",
                'Tmd': f"{group['Tmd'].mean():.2f}±{group['Tmd'].std():.3f}",
                'KLp': f"{group['KLp'].mean()*1000:.2f}±{group['KLp'].std()*1000:.2f}",
                # 'JSD': f"{group['JSD'].mean():.2f}±{group['JSD'].std():.3f}"
            }
            stats.append(stats_dict)
        
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(stats_writefile, index=False)

print("\n处理完成！所有文件已更新。")