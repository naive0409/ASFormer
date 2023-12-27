import json

# 打开JSON文件
with open(r"D:\MLdata\thumos\annotations\thumos14.json", 'r') as file:
    # 从文件中加载JSON数据
    data = json.load(file)

# 现在，'data'变量包含了从JSON文件中读取的数据
print(data)

# if __name__ == '__main__':

'''
import numpy as np
feature_thumos = np.load(r"D:\MLdata\thumos\i3d_features\video_test_0000006.npy")
import json
with open(r"D:\MLdata\thumos\annotations\thumos14.json", 'r') as file:
    thumos = json.load(file)
    
file_ptr = open(r"D:\MLdata\ASFormer\data\gtea\groundTruth\S1_Cheese_C1.txt", 'r')
content = file_ptr.read().split('\n')[:-1]
file_ptr.close()
features = np.load(r"D:\MLdata\ASFormer\data\gtea\features\S1_Cheese_C1.npy")
'''