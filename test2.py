import os


def get_all_files_in_directory(directory):
    file_list = []

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        # 判断是否为文件
        if os.path.isfile(filepath):
            name, extension = os.path.splitext(filename)
            file_list.append(name)

    return file_list

actions = []

names = get_all_files_in_directory(r'D:/MLdata/multithumos/groundTruth_per_clip/')
gts = [r'D:/MLdata/multithumos/groundTruth_per_clip/' + s + '.txt' for s in names]
for s in gts:
    file_ptr = open(s, 'r')
    content = file_ptr.read().split('\n')[:-1]
    for act in content:
        if act not in actions:
            actions.append(act)

# actions = ['CricketBowling', 'CricketShot', 'Background', 'VolleyballSpiking', 'JavelinThrow', 'Shotput', 'TennisSwing',
#            'GolfSwing', 'ThrowDiscus', 'Billiards', 'CleanAndJerk', 'LongJump', 'Diving', 'CliffDiving', 'BasketballDunk',
#            'HighJump', 'HammerThrow', 'SoccerPenalty', 'BaseballPitch', 'FrisbeeCatch', 'PoleVault']
# dict_keys(['CricketBowling', 'CricketShot', 'VolleyballSpiking', 'JavelinThrow', 'Shotput', 'TennisSwing', 'GolfSwing', 'ThrowDiscus', 'Billiards', 'CleanAndJerk', 'LongJump', 'Diving', 'CliffDiving', 'BasketballDunk', 'HighJump', 'HammerThrow', 'SoccerPenalty', 'BaseballPitch', 'FrisbeeCatch', 'PoleVault', 'Background'])
# actions_dict = {'CricketBowling': 5, 'CricketShot': 6, 'VolleyballSpiking': 19, 'JavelinThrow': 12, 'Shotput': 15,
#                 'TennisSwing': 17, 'GolfSwing': 9, 'ThrowDiscus': 18, 'Billiards': 2, 'CleanAndJerk': 3, 'LongJump': 13,
#                 'Diving': 7, 'CliffDiving': 4, 'BasketballDunk': 1, 'HighJump': 11, 'HammerThrow': 10,
#                 'SoccerPenalty': 16, 'BaseballPitch': 0, 'FrisbeeCatch': 8, 'PoleVault': 14,
#                 'Background': 20}
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

pass