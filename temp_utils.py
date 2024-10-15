import os

def rename_images_to_six_digits(folder_path):
    """
    将文件夹中的所有以四位数命名的图片重命名为六位数格式（前面补0）
    例如 0855.png -> 000855.png
    """
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)

    for file_name in files:
        # 提取文件名和扩展名
        name, ext = os.path.splitext(file_name)

        # 检查文件是否为四位数字命名且扩展名为 .png
        if ext == '.png' and name.isdigit() and len(name) == 4:
            # 将文件名变为六位数格式，前面补0
            new_name = name.zfill(6) + ext
            # 重命名文件
            os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_name))
            print(f"Renamed: {file_name} -> {new_name}")

# 调用示例
# rename_images_to_six_digits('your_folder_path_here')

# 请将上面注释中的 'your_folder_path_here' 替换为你实际的文件夹路径，调用该函数即可。
