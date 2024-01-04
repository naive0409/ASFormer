actions_dict = {'CricketBowling': 5, 'CricketShot': 6, 'VolleyballSpiking': 19, 'JavelinThrow': 12, 'Shotput': 15,
                'TennisSwing': 17, 'GolfSwing': 9, 'ThrowDiscus': 18, 'Billiards': 2, 'CleanAndJerk': 3, 'LongJump': 13,
                'Diving': 7, 'CliffDiving': 4, 'BasketballDunk': 1, 'HighJump': 11, 'HammerThrow': 10,
                'SoccerPenalty': 16, 'BaseballPitch': 0, 'FrisbeeCatch': 8, 'PoleVault': 14,
                'Background': 20}

import eval
import myargs
import json
import numpy as np

import csv

def find_equal_value_segments(lst):
    segments = []
    start_frame = None
    end_frame = None
    current_label = None

    for i, value in enumerate(lst):
        if start_frame is None or value != current_label:
            if (start_frame is not None) and (current_label != 20):
                segments.append({'start_frame': start_frame, 'end_frame': end_frame, 'label': int(current_label)})
            start_frame = i
            current_label = value
        end_frame = i

    if (start_frame is not None) and (current_label != 20):
        segments.append({'start_frame': start_frame, 'end_frame': end_frame, 'label': int(current_label)})

    return segments

def save_to_csv(segments, filename='output.csv'):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['start_frame', 'end_frame', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for segment in segments:
            writer.writerow(segment)

# 示例用法
# my_list = [20, 20, 4, 4, 4, 17, 17, 17, 20, 20, 20, 4, 4, 20, 17, 17, 20, 20]
# segments = find_equal_value_segments(my_list)
# save_to_csv(segments)



dataset_dir = r'D:\MLdata\thumos'
with open(dataset_dir + r"/annotations/thumos14.json", 'r') as file:
    myargs.thumos = json.load(file)
features = list(myargs.thumos['database'].keys())
# if action == 'train':
# features = [s for s in features if 'validation' in s]
# if action == 'test':
features = [s for s in features if 'test' in s]
names = features
pass

for name in names:

    content = myargs.thumos['database'][name]
    # batch_target = np.ones(len_) * actions_dict['Background']
    # for i in thumos['databse'][]
    total_frames = content['fps'] * content['duration']
    batch_target = np.ones(int(total_frames)) * actions_dict['Background']
    predicted_frames = np.ones(int(total_frames)) * actions_dict['Background']

    for anno in content['annotations']:
        id_begin = anno['segment(frames)'][0]
        id_end = anno['segment(frames)'][1]

        # classes_by_frame[int(id_begin):int(id_end)] = int(anno['label_id'])

        # id_begin = id_begin / total_frames * len_
        # id_end = id_end / total_frames * len_

        # id_begin = anno['segment'][0] * content['fps'] / 16 * 4
        # id_end = anno['segment'][1] * content['fps'] / 16 * 4

        batch_target[int(id_begin):int(id_end)] = int(anno['label_id'])

    file_ptr = open(r"./results/thumos/0102/" + name, 'r')
    predicted = file_ptr.read().split('\n')[1].split(' ')
    len_predicted = len(predicted)
    for index in range(len_predicted):
        predicted[index] = int(actions_dict[predicted[index]])
    # predicted = np.ones(l)

    # 上采样
    up_sample_factor = total_frames / len_predicted
    down_sample_factor = len_predicted / total_frames
    for index in range(int(total_frames)):
        predicted_frames[index] = predicted[int(index * down_sample_factor)]
        pass

    segments = find_equal_value_segments(predicted_frames)
    save_to_csv(segments, filename=r"./results/thumos/0102/" + name + '.csv')

    eval.segment_bars_with_confidence(r"./results/thumos/0102/" + name + ".png",
                                      [0],  # confidence.tolist(),
                                      batch_target, predicted_frames)

    pass
