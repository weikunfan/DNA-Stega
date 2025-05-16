import os
import pandas as pd
import numpy as np
import random
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from EMD import getEMD
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import cg_tm_kl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import ot
colors = ['blue', 'orange']
marker = ['.', '.']

def getEMD(xs, xt, num_samples, path,filename):
    n = num_samples  # Number of samples

    # Uniform distribution on samples
    a, b = np.ones((n,)), np.ones((n,))

    # 源和目标分布图
    plt.figure(figsize=(8, 6))
    plt.scatter(xs[:, 0], xs[:, 1], color='blue', marker='+', label='Source samples', s=40, alpha=0.7)
    plt.scatter(xt[:, 0], xt[:, 1], color='red', marker='x', label='Target samples', s=40, alpha=0.7)
    plt.legend(loc='best', fontsize=10, frameon=True, fancybox=True, shadow=True)
    plt.title('Source and Target Distributions', fontsize=14, fontweight='bold')
    plt.xlabel('X-axis', fontsize=12)
    plt.ylabel('Y-axis', fontsize=12)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # 保存文件
    savename = os.path.splitext(os.path.basename(path))[0]
    # save_dir = 'PICC'
    save_dir = f'/home/fan/Code/VAE_Synthetic_Steganography/results/{filename}/PCA'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{savename}_distribution.jpg'))
    plt.close()

    # 计算 SWD
    n_seed = 50
    n_projections_arr = np.logspace(0, 3, 25, dtype=int)
    res = np.empty((n_seed, 25))
    for seed in range(n_seed):
        for i, n_projections in enumerate(n_projections_arr):
            res[seed, i] = ot.sliced_wasserstein_distance(xs, xt, a, b, n_projections, seed=seed)

    res_mean = np.mean(res, axis=0)
    res_std = np.std(res, axis=0)

    # 绘制 SWD 曲线（保持不变）
    plt.figure(2)
    plt.plot(n_projections_arr, res_mean, label="SWD")
    plt.fill_between(n_projections_arr, res_mean - 2 * res_std, res_mean + 2 * res_std, alpha=0.5)
    plt.legend()
    plt.xscale('log')
    plt.xlabel("Number of projections")
    plt.ylabel("Distance")
    plt.title('Sliced Wasserstein Distance with 95% confidence interval')

    return res_mean[-1]

def split_words(line, num):
    words = [line[i:i+num] for i in range(0, len(line), num)]
    return words

def total_vector(words, word2vec):
    vec = np.zeros(300).reshape((1, 300))
    for word in words:
        try:
            vec += word2vec.wv[word].reshape((1, 300))
        except KeyError:
            continue
    return vec

def process_sequence(path, word2vec, split_length, seq_length, base_name=None, max_lines=None):
    """
    处理序列数据，对于ASM286374v1限制处理前5400行
    
    Args:
        path: 文件路径
        word2vec: Word2Vec模型
        split_length: 分割长度
        seq_length: 序列长度
        base_name: 基础名称（用于判断是否是ASM286374v1）
        max_lines: 最大处理行数
    """
    with open(path, 'r') as f:
        lines = f.readlines()
        
    # 如果是ASM286374v1，只处理前5400行
    if base_name == 'ASM286374v1' and max_lines:
        lines = lines[:max_lines]
        print(f"限制处理文件 {path} 的前 {max_lines} 行")
        
    raw_pos = cg_tm_kl.txt_process_sc_duo(path, len_sc=seq_length, beg_sc=0, end_sc=len(lines), 
                                         PADDING=False, flex=10, num1=seq_length, tiqu=False)
    raw_pos = list(filter(lambda x: x not in ['', None], raw_pos))

    pos_df = pd.DataFrame(raw_pos, columns=[0])
    pos_df['words'] = pos_df[0].apply(lambda x: split_words(x, split_length))

    vectors = [total_vector(words, word2vec) for words in pos_df['words']]
    return np.squeeze(vectors)


def plot_with_labels(embedded_data, labels, path, output_dir):
    plt.cla()
    fig, ax = plt.subplots()
    ax.axis("off")

    data_df = pd.DataFrame({'x': embedded_data[:, 0], 'y': embedded_data[:, 1], 'label': labels})
    for index in [0, 1]:
        subset = data_df[data_df['label'] == index]
        plt.scatter(subset['x'], subset['y'], marker=marker[index], color=colors[index], alpha=0.65)
    # 将 .csv 文件路径转换为 .jpg 文件路径
    base_filename = path.replace('.csv', '.jpg')
    savename = os.path.join(output_dir, base_filename)
    plt.savefig(savename)

def save_emd_result(pd_EMD, emd_result_path):
    # 如果文件已经存在，读取现有文件并追加数据
    if os.path.exists(emd_result_path):
        pd_EMD.to_csv(emd_result_path, mode='a', header=False, index=False)
    else:
        # 文件不存在时，写入表头
        pd_EMD.to_csv(emd_result_path, mode='w', header=True, index=False)

if __name__ == '__main__':
    files_name = ['ASM286374v1']
    # files_name = ['ASM141792v1', 'ASM286374v1', 'ASM400647v1', 'ASM949793v1']
    read_files = []
    # files_name = ['ASM141792v1', 'ASM286374v1', 'ASM400647v1', 'ASM949793v1', 'ASM1821919v1']
    asm_to_gca = {
        'ASM141792v1': 'GCA_001417925',
        'ASM286374v1': 'GCA_002863745',
        'ASM400647v1': 'GCA_004006475',
        'ASM949793v1': 'GCA_009497935',
        'ASM1821919v1': 'GCA_018219195'
    }
    for base_name in files_name:
        base_dir = f'/home/fan/Code/VAE_Synthetic_Steganography/0_Data/{base_name}'
        for file in os.listdir(base_dir):
            if file.endswith('.txt'):
                file_path = os.path.join(base_dir, file)
                read_files.append(file_path)
                
                try:
                    length_str = file.split('_')[1]  # 提取长度字符串
                    devide_ = int(file.split('_')[2].split('.')[0])  # 提取分块参数
                    len_sc = 198 if length_str == '198' else 200  # 确定序列长度
                except (IndexError, ValueError):
                    raise ValueError(f"文件名 {file} 不符合预期格式，无法提取长度和分割参数。")
                
                # 创建模型文件名
                model_name = f"{base_name}_{len_sc}_{devide_}.model"
                embedding_model_path = os.path.join(base_dir, model_name)

                # 检查模型是否存在
                if os.path.exists(embedding_model_path):
                    # 加载已存在的模型
                    word2vec_model = Word2Vec.load(embedding_model_path)
                    print(f"已加载模型: {embedding_model_path}")
                else:
                    # 生成新模型并保存
                    print(f"训练新模型: {embedding_model_path}")
                    all_sequences = [split_words(line.strip(), devide_) for line in open(file_path, 'r')]
                    word2vec_model = Word2Vec(all_sequences, vector_size=300, window=devide_, min_count=5, sg=1, hs=1, epochs=10)
                    word2vec_model.save(embedding_model_path)
                
                # 仅处理与当前模型匹配的数据文件
                if f"_{len_sc}_{devide_}" in file:
                    print(f"处理文件: {file_path} 使用模型: {model_name}")
                    # 处理当前数据文件
                    # 设置最大行数
                    max_lines = 5400 if base_name == 'ASM286374v1' else None
                    
                    # 处理原始序列
                    original_vectors = process_sequence(file_path, word2vec_model, 
                                                     split_length=devide_, 
                                                     seq_length=len_sc,
                                                     base_name=base_name,
                                                     max_lines=max_lines)
                    original_2D = PCA(n_components=2).fit_transform(original_vectors)

                    # 隐写文件路径列表
                    steganography_dirs = [
                        # f'/home/fan/Code/VAE_Synthetic_Steganography/ExperimentData/{base_name}/{len_sc}/VAE/lr_0.003',
                        # f'/home/fan/Code/VAE_Synthetic_Steganography/ExperimentData/{base_name}/{len_sc}/VAE/lr_0.003/discop2',
                        # f'/home/fan/Code/VAE_Synthetic_Steganography/ExperimentData/{base_name}/{len_sc}/VAE/lr_0.003/ssDiscop',
                        # f'/home/fan/Code/VAE_Synthetic_Steganography/ExperimentData/{base_name}/{len_sc}/VAE/lr_0.003/standardDiscop',
                        f'/home/fan/Code/VAE_Synthetic_Steganography/ExperimentData/{base_name}/{len_sc}',
                        f'/home/fan/Code/VAE_Synthetic_Steganography/ExperimentData/{base_name}/{len_sc}/huffman'
                    ]

                    # 创建结果存储目录
                    base_results_dir = os.path.join('/home/fan/Code/VAE_Synthetic_Steganography/ExperimentData', base_name)
                    csv_dir = os.path.join(base_results_dir, 'CSV')
                    jpg_dir = os.path.join(base_results_dir, 'JPG')
                    os.makedirs(csv_dir, exist_ok=True)
                    os.makedirs(jpg_dir, exist_ok=True)

                    # 创建一个 DataFrame 来存储 EMD 结果
                    pd_EMD = pd.DataFrame(columns=['name', 'emd'])

                    # 处理每个目录下的隐写文��
                    for steganography_dir in steganography_dirs:
                        if not os.path.exists(steganography_dir):
                            print(f"路径不存在，跳过：{steganography_dir}")
                            continue
                            
                        print(f"处理目录：{steganography_dir}")
                        # 处理隐写文件
                        steganography_files = [f for f in os.listdir(steganography_dir) if f.endswith('.txt')]
                        for stego_file in steganography_files:
                            if f"_{len_sc}_{devide_}" in stego_file:
                                # 检查对应的输出文件是否已存在
                                csv_filename = stego_file.replace('.txt', '_PCA.csv')
                                jpg_filename = stego_file.replace('.txt', '_PCA.jpg')
                                csv_path = os.path.join(csv_dir, csv_filename)
                                jpg_path = os.path.join(jpg_dir, jpg_filename)
                                
                                # 如果输出文件都已存在，则跳过此文件
                                if os.path.exists(csv_path) and os.path.exists(jpg_path):
                                    print(f"文件 {stego_file} 的处理结果已存在，跳过处理")
                                    continue

                                try:
                                    stego_path = os.path.join(steganography_dir, stego_file)
                                    stego_vectors = process_sequence(stego_path, word2vec_model, 
                                                                  split_length=devide_, 
                                                                  seq_length=len_sc,
                                                                  base_name=base_name,
                                                                  max_lines=max_lines)
                                    stego_2D = PCA(n_components=2).fit_transform(stego_vectors)

                                    emd_value = getEMD(original_2D, stego_2D, len(original_2D), stego_path, base_name)
                                    emd_row = pd.DataFrame({'name': [stego_file], 'emd': [emd_value]})
                                    pd_EMD = pd.concat([pd_EMD, emd_row], ignore_index=True)

                                    # 保存嵌入结果到CSV目录
                                    combined_df = pd.DataFrame({
                                        'ori_x': original_2D[:, 0], 'ori_y': original_2D[:, 1],
                                        'stego_x': stego_2D[:, 0], 'stego_y': stego_2D[:, 1]
                                    })
                                    csv_filename = stego_file.replace('.txt', '_PCA.csv')
                                    csv_path = os.path.join(csv_dir, csv_filename)
                                    combined_df.to_csv(csv_path, index=False)

                                    # 绘制PCA降维图并保存到JPG目录
                                    jpg_filename = stego_file.replace('.txt', '_PCA.jpg')
                                    jpg_path = os.path.join(jpg_dir, jpg_filename)
                                    plot_with_labels(original_2D, np.zeros(len(original_2D)), jpg_path, jpg_dir)
                                    plot_with_labels(stego_2D, np.ones(len(stego_2D)), jpg_path, jpg_dir)
                                    print(f"完成 {stego_file} 的处理")
                                except Exception as e:
                                    print(f"错误发生在文件: {stego_path}")
                                    print(f"错误类型: {type(e).__name__}")
                                    print(f"错误信息: {str(e)}")
                                    continue  # 跳过这个文件，继续处理下一个
                    # 保存EMD结果
                    emd_result_path = os.path.join(base_results_dir, f'{asm_to_gca[base_name]}_emd_results.csv')
                    save_emd_result(pd_EMD, emd_result_path)
                    print(f"{base_name} 的所有EMD结果已保存到 {emd_result_path}。")

                    # 计算EMD的统计结果
                    pd_EMD['base_name'] = pd_EMD['name'].apply(lambda x: '_'.join(x.split('_')[:-1]))
                    stats = []
                    for name in pd_EMD['base_name'].unique():
                        group = pd_EMD[pd_EMD['base_name'] == name]
                        print('group:',group)
                        stats_dict = {
                            'method': name,
                            'EMD': f"{group['emd'].mean():.2f}±{group['emd'].std():.2f}"
                        }
                        stats.append(stats_dict)
                    
                    # 创建统计结果DataFrame并保存
                    stats_df = pd.DataFrame(stats)
                    stats_writefile = os.path.join(base_results_dir, f'{asm_to_gca[base_name]}_emd_stats.csv')
                    print(f"保存EMD统计结果到：{stats_writefile}")
                    # 如果文件存在则追加，不存在则创建新文件
                    if os.path.exists(stats_writefile):
                        stats_df.to_csv(stats_writefile, mode='a', header=False, index=False)
                    else:
                        stats_df.to_csv(stats_writefile, mode='w', header=True, index=False)