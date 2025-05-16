import os

def qingchu_Model(path):
    # 存储所有模型文件的路径
    File = []
    for root, dirs, files in os.walk(path):
        for file in files:
            File.append(os.path.join(root, file))

    # 过滤出 .pkl 模型文件
    Models = [f for f in File if f.endswith('.pkl')]
    
    # 记录所有文件的名称和对应的信息
    model_info = []  # 存储每个文件的 (模型名称, epoch 数, 损失值) 元组
    for m in Models:
        m_name = os.path.basename(m)[:-4]  # 去掉文件扩展名
        parts = m_name.split('-')

        # 检查文件名是否符合 'modelname-experiment-epoch-loss' 的格式
        try:
            base_name = '-'.join(parts[:2])  # 基本名称（模型名称 + 实验编号）
            epoch_num = int(parts[2])  # epoch 数
            loss_val = float(parts[3].split('_')[1])  # 提取损失值
            model_info.append((base_name, epoch_num, loss_val, m))
        except (ValueError, IndexError):
            print(f"Skipping file {m} due to incompatible format.")
    
    # 查找最小损失值的模型
    if model_info:
        # 按 base_name 对模型进行分组
        models_by_name = {}
        for base_name, epoch_num, loss_val, filepath in model_info:
            if base_name not in models_by_name:
                models_by_name[base_name] = []
            models_by_name[base_name].append((epoch_num, loss_val, filepath))
        
        # 对每个模型名称保留最小损失值的文件，删除其他文件
        for base_name, epoch_loss_files in models_by_name.items():
            min_loss_file = min(epoch_loss_files, key=lambda x: x[1])[2]  # 找到最小损失值对应的文件
            print(f"Keeping file: {min_loss_file}")
            for _, _, filepath in epoch_loss_files:
                if filepath != min_loss_file:
                    os.remove(filepath)  # 删除其他文件
                    print(f"Removed file: {filepath}")
    else:
        print("No valid model files found.")
