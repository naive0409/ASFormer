import json

# 打开JSON文件
with open(r"D:\MLdata\thumos\annotations\thumos14.json", 'r') as file:
    # 从文件中加载JSON数据
    data = json.load(file)

# 现在，'data'变量包含了从JSON文件中读取的数据
print(data)
data = data['database']

d = dict()

for video in data:
    for i in data[video]['annotations']:
        label = i['label']
        label_id = i['label_id']
        
        if label not in d:
            d[label] = -1

        # 检查label_id是否在对应的数组中，如果不在则添加
        if label_id != d[label]:
            d[label] = label_id
        #
        # if not d[i['label']]:
        #     d[i['label']] = []
        # if not i['label_id'] in d[i['label']]:
        #     d[i['label']].append(i['label_id'])

        pass

print(d)

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