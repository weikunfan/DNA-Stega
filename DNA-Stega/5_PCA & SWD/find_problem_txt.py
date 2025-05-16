def count_lines_in_directory(directory_path):
    """
    统计指定目录下所有txt文件的行数
    
    Args:
        directory_path: 目录路径
        
    Returns:
        pandas DataFrame，包含文件名和行数两列
    """
    import os
    import pandas as pd
    
    results = []
    
    # 遍历目录下的所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r') as f:
                    line_count = sum(1 for line in f)
                results.append({
                    'filename': filename,
                    'line_count': line_count
                })
            except Exception as e:
                print(f"读取文件 {filename} 时出错: {str(e)}")
    
    # 创建DataFrame并按文件名排序
    df = pd.DataFrame(results)
    df = df.sort_values('filename')
    
    return df

# 使用示例：
directory = "/home/fan/Code/VAE_Synthetic_Steganography/ExperimentData/ASM286374v1/198"
result_df = count_lines_in_directory(directory)
print(result_df)

# # 可选：保存到CSV
# result_df.to_csv('file_lines.csv', index=False)