import cg_tm_kl
import os

def main(Path):
    print(f"Processing file: {Path}")

    # 根据文件名中的特定字符串设置参数
    if '198' in Path:
        index = ['3', '6']
        len_sc_ = 198
    else:
        index = ['4', '5']
        len_sc_ = 200

    for len_seq_ in index:
        lines = cg_tm_kl.txt_process_sc_duo(dp_sc=Path, len_sc=len_sc_, beg_sc=0, end_sc=9999999, PADDING=False, flex=0,
                                            num1=int(len_seq_))
        print(f"Processing sequence length: {len_seq_}")

        P_write = Path[: -4] + '_' + str(len_seq_) + '.txt'
        with open(P_write, 'w') as f1:
            for line in lines:
                f1.write(line)
                f1.write('\n')
    # 删除源文件
    try:
        os.remove(Path)
        print(f"Deleted source file: {Path}")
    except Exception as e:
        print(f"Failed to delete {Path}: {e}")
if __name__ == '__main__':
    name = ['ASM141792v1','ASM286374v1','ASM400647v1','ASM949793v1','ASM1821919v1']

    for n in name:
        FileAll = r'/home/fan/Code/VAE_Synthetic_Steganography/0_Data/{}'.format(str(n))
        # 检查目录是否存在
        if not os.path.exists(FileAll):
            print(f"Directory {FileAll} does not exist. Skipping...")
            continue
        else:
            print(f"Searching in directory: {FileAll}")
        Path = []
        for root, dirs, files in os.walk(FileAll):
            print('root:', root)
            for file in files:
                Path.append(os.path.join(root, file))

        for p in Path:
            main(p)
