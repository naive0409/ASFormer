'''
    Adapted from https://github.com/yabufarha/ms-tcn
'''

import torch
import numpy as np
import random
from grid_sampler import GridSampler, TimeWarpLayer

import os
import myargs

class BatchGenerator(object):
    # def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate):
    def __init__(self, num_classes, actions_dict, features_path, sample_rate):
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        # self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.features = []
        self.names = []

        self.timewarp_layer = TimeWarpLayer()

    def reset(self):
        self.index = 0
        self.my_shuffle()

    def has_next(self):
        if self.index < len(self.names):
            return True
        return False

    def read_data_(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        # ['S2_Cheese_C1.txt', 'S2_CofHoney_C1.txt', 'S2_Coffee_C1.txt', 'S2_Hotdog_C1.txt', 'S2_Pealate_C1.txt', 'S2_Peanut_C1.txt', 'S2_Tea_C1.txt', 'S3_Cheese_C1.txt', 'S3_CofHoney_C1.txt', 'S3_Coffee_C1.txt', 'S3_Hotdog_C1.txt', 'S3_Pealate_C1.txt', 'S3_Peanut_C1.txt', 'S3_Tea_C1.txt', 'S4_Cheese_C1.txt', 'S4_CofHoney_C1.txt', 'S4_Coffee_C1.txt', 'S4_Hotdog_C1.txt', 'S4_Pealate_C1.txt', 'S4_Peanut_C1.txt', 'S4_Tea_C1.txt']
        file_ptr.close()

        self.gts = [self.gt_path + vid for vid in self.list_of_examples]
        self.features = [self.features_path + vid.split('.')[0] + '.npy' for vid in self.list_of_examples]
        self.my_shuffle()

    def get_all_files_in_directory(self, directory):
        file_list = []

        # 遍历目录中的所有文件
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)

            # 判断是否为文件
            if os.path.isfile(filepath):
                name, extension = os.path.splitext(filename)
                file_list.append(name)

        return file_list

    def read_data(self, action):
        # self.features = self.get_all_files_in_directory(self.features_path)
        self.features = list(myargs.thumos['database'].keys())
        if action == 'train':
            self.features = [s for s in self.features if 'validation' in s]
        if action == 'test':
            self.features = [s for s in self.features if 'test' in s]
        self.my_shuffle()
        self.names = self.features
        # ['video_validation_0000262', 'video_validation_0000051', 'video_validation_0000163',
        self.features = [self.features_path + '/' + s + '.npy' for s in self.features]
        # ['D:\\MLdata\\thumos/i3d_features/video_validation_0000262.npy', 'D:\\MLdata\\thumos/i3d_features/video_validation_0000051.npy',
        pass


    def my_shuffle(self):
        # shuffle list_of_examples, gts, features with the same order
        randnum = random.randint(0, 100)
        # random.seed(randnum)
        # random.shuffle(self.list_of_examples)
        # random.seed(randnum)
        # random.shuffle(self.gts)
        random.seed(randnum)
        random.shuffle(self.features)

    def warp_video(self, batch_input_tensor, batch_target_tensor):
        '''
        :param batch_input_tensor: (bs, C_in, L_in)
        :param batch_target_tensor: (bs, L_in)
        :return: warped input and target
        '''
        bs, _, T = batch_input_tensor.shape
        grid_sampler = GridSampler(T)
        grid = grid_sampler.sample(bs)
        grid = torch.from_numpy(grid).float()

        warped_batch_input_tensor = self.timewarp_layer(batch_input_tensor, grid, mode='bilinear')
        batch_target_tensor = batch_target_tensor.unsqueeze(1).float()
        warped_batch_target_tensor = self.timewarp_layer(batch_target_tensor, grid, mode='nearest')  # no bilinear for label!
        warped_batch_target_tensor = warped_batch_target_tensor.squeeze(1).long()  # obtain the same shape

        return warped_batch_input_tensor, warped_batch_target_tensor

    def merge(self, bg, suffix):
        '''
        merge two batch generator. I.E
        BatchGenerator a;
        BatchGenerator b;
        a.merge(b, suffix='@1')
        :param bg:
        :param suffix: identify the video
        :return:
        '''

        self.list_of_examples += [vid + suffix for vid in bg.list_of_examples]
        self.gts += bg.gts
        self.features += bg.features

        print('Merge! Dataset length:{}'.format(len(self.list_of_examples)))


    def next_batch(self, batch_size, if_warp=False): # if_warp=True is a strong data augmentation. See grid_sampler.py for details.
        # batch = self.list_of_examples[self.index:self.index + batch_size]
        # batch_gts = self.gts[self.index:self.index + batch_size]
        batch_features = self.features[self.index:self.index + batch_size]
        batch_names = self.names[self.index:self.index + batch_size]

        self.index += batch_size

        batch_input = []
        batch_target = []

        '''
        https://github.com/happyharrycn/actionformer_release/blob/main/FAQ.md
        batch_size = 4
        feature_grid = (timestamp * FPS - 0.5 * window_size) / feature_stride
        499 = 67.11 second * 30 fps / 16 * batchsize
        '''


        for idx, vid in enumerate(batch_features):
            features = np.load(batch_features[idx]).T
            content = myargs.thumos['database'][batch_names[idx]]
            classes = np.zeros(np.shape(features)[1])
            # for i in thumos['databse'][]
            total_frames = content['fps'] * content['duration']
            classes_by_frame = np.zeros(int(total_frames))

            for anno in content['annotations']:

                id_begin = anno['segment(frames)'][0]
                id_end = anno['segment(frames)'][1]

                classes_by_frame[int(id_begin):int(id_end)] = int(anno['label_id'])

                id_begin = id_begin / total_frames * features.shape[1]
                id_end = id_end / total_frames * features.shape[1]

                # id_begin = anno['segment'][0] * content['fps'] / 16 * 4
                # id_end = anno['segment'][1] * content['fps'] / 16 * 4

                classes[int(id_begin):int(id_end)] = int(anno['label_id'])

        feature = features[:, ::self.sample_rate]
        target = classes[::self.sample_rate]
        batch_input.append(feature)
        batch_target.append(target)

        # for idx, vid in enumerate(batch):
        #     features = np.load(batch_features[idx])
        #     file_ptr = open(batch_gts[idx], 'r')
        #     content = file_ptr.read().split('\n')[:-1]
        #     classes = np.zeros(min(np.shape(features)[1], len(content)))
        #     for i in range(len(classes)):
        #         classes[i] = self.actions_dict[content[i]]
        #
        #     feature = features[:, ::self.sample_rate]
        #     target = classes[::self.sample_rate]
        #     batch_input.append(feature)
        #     batch_target.append(target)

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)  # bs, C_in, L_in
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            if if_warp:
                warped_input, warped_target = self.warp_video(torch.from_numpy(batch_input[i]).unsqueeze(0), torch.from_numpy(batch_target[i]).unsqueeze(0))
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]], batch_target_tensor[i, :np.shape(batch_target[i])[0]] = warped_input.squeeze(0), warped_target.squeeze(0)
            else:
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
                batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask, batch_names


if __name__ == '__main__':
    pass