import numpy as np

def generate_file_paths(txt_file, npy_file):
    file_paths = []
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.split()
        folder_name = parts[0]
        num_images = int(parts[1])
        
        for i in range(num_images-1):
            file_name = f"{folder_name}_{(i+1):03d}.png"
            file_paths.append(file_name)
            
    np.save(npy_file, file_paths)
    #return file_paths

# 使用示例
file_paths = generate_file_paths('meta_info_LDVD_test_GT.txt','LDV_D.npy')
print(file_paths)
