import torch
 
from model import *
from batch_gen import BatchGenerator
from eval import func_eval

import os
import argparse
import numpy as np
import random

import json
import myargs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 19980125 # my birthday, :)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
 
parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="50salads")
parser.add_argument('--split', default='1')
parser.add_argument('--model_dir', default='models')
parser.add_argument('--result_dir', default='results')
# 修改dataset路径
# /mnt/DataDrive164/zhanghao/datasets/50salads/data/breakfast
parser.add_argument('--dataset_dir', default='/mnt/DataDrive164/zhanghao/datasets/thumos14_lite/actionformer_thumos')
parser.add_argument('--predict_folder', default='')

args = parser.parse_args()
with open(args.dataset_dir + r"/annotations/thumos14.json", 'r') as file:
    myargs.thumos = json.load(file)
 
num_epochs = 30

lr = 0.0005
num_layers = 10
num_f_maps = 64
features_dim = 2048
bz = 1

channel_mask_rate = 0.3


# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

# To prevent over-fitting for GTEA. Early stopping & large dropout rate
if args.dataset == "gtea":
    channel_mask_rate = 0.5
    
if args.dataset == 'breakfast':
    lr = 0.0001

if args.dataset == 'thumos':
    sample_rate = 2

# train split
vid_list_file = args.dataset_dir + "/data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
# test split
# vid_list_file_tst = args.dataset_dir + "/data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
# features_path = args.dataset_dir + "/data/"+args.dataset+"/features/"
features_path = args.dataset_dir + '/i3d_features' + '/'
# ground truth
# gt_path = args.dataset_dir + "/data/"+args.dataset+"/groundTruth/"

# mapping_file = args.dataset_dir + "/data/"+args.dataset+"/mapping.txt"

model_dir = "./{}/".format(args.model_dir)+args.dataset  # +"/split_"+args.split

results_dir = "./{}/".format(args.result_dir)+args.dataset  # +"/split_"+args.split
 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
 
'''
file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])
index2label = dict()
for k,v in actions_dict.items():
    index2label[v] = k
num_classes = len(actions_dict)
'''
# actions_dict = {'background': 10, 'close': 3, 'fold': 8, 'open': 1, 'pour': 2, 'put': 7, 'scoop': 5, 'shake': 4,
#                 'spread': 9, 'stir': 6, 'take': 0}
# {'CricketBowling': [5], 'CricketShot': [6], 'VolleyballSpiking': [19], 'JavelinThrow': [12], 'Shotput': [15], 'TennisSwing': [17], 'GolfSwing': [9], 'ThrowDiscus': [18], 'Billiards': [2], 'CleanAndJerk': [3], 'LongJump': [13], 'Diving': [7], 'CliffDiving': [4], 'BasketballDunk': [1], 'HighJump': [11], 'HammerThrow': [10], 'SoccerPenalty': [16], 'BaseballPitch': [0], 'FrisbeeCatch': [8], 'PoleVault': [14]}
actions_dict = {'CricketBowling': 5, 'CricketShot': 6, 'VolleyballSpiking': 19, 'JavelinThrow': 12, 'Shotput': 15, 'TennisSwing': 17, 'GolfSwing': 9, 'ThrowDiscus': 18, 'Billiards': 2, 'CleanAndJerk': 3, 'LongJump': 13, 'Diving': 7, 'CliffDiving': 4, 'BasketballDunk': 1, 'HighJump': 11, 'HammerThrow': 10, 'SoccerPenalty': 16, 'BaseballPitch': 0, 'FrisbeeCatch': 8, 'PoleVault': 14,
                'Background': 20}
num_classes = len(actions_dict)

print('\n')
print('batch_size={}'.format(bz))
print('channel_mask_rate={}'.format(channel_mask_rate))
print('features_dim={}'.format(features_dim))
print('features_path={}'.format(features_path))
print('lr={}'.format(lr))
print('classes={}'.format(num_classes))
print('epoches={}'.format(num_epochs))
print('f_maps={}'.format(num_f_maps))
print('layers={}'.format(num_layers))
print('result_dir={}'.format(results_dir))
print('sample_rate={}'.format(sample_rate))


trainer = Trainer(num_layers, 2, 2, num_f_maps, features_dim, num_classes, channel_mask_rate)
# trainer = Trainer(num_layers, 2, 2, num_f_maps, features_dim,  channel_mask_rate)
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict,  features_path, sample_rate)
    '''
    batch_gen.
    features_path    'D:\\MLdata\\ASFormer/data/gtea/features/'
    gt_path 'D:\\MLdata\\ASFormer/data/gtea/groundTruth/'
    actions_dict {'background': 10, 'close': 3, 'fold': 8, 'open': 1, 'pour': 2, 'put': 7, 'scoop': 5, 'shake': 4, 'spread': 9, 'stir': 6, 'take': 0}
    '''
    # batch_gen.read_data(vid_list_file)
    batch_gen.read_data("train")
    '''
    bath_gen.
    list_of_examples {list:21}     ['S3_Pealate_C1.txt', 'S4_CofHoney_C1.txt', 'S4_Tea_C1.txt', 'S3_Coffee_C1.txt', 'S3_CofHoney_C1.txt', 'S4_Coffee_C1.txt', 'S2_Hotdog_C1.txt', 'S3_Cheese_C1.txt', 'S2_Cheese_C1.txt', 'S2_Pealate_C1.txt', 'S4_Cheese_C1.txt', 'S2_CofHoney_C1.txt', 'S3_Tea_C1.txt', 'S3_Hotdog_C1.txt', 'S2_Coffee_C1.txt', 'S4_Hotdog_C1.txt', 'S4_Peanut_C1.txt', 'S2_Tea_C1.txt', 'S2_Peanut_C1.txt', 'S4_Pealate_C1.txt', 'S3_Peanut_C1.txt']
    gts {list:21}               ['D:\\MLdata\\ASFormer/data/gtea/groundTruth/S3_Pealate_C1.txt', 'D:\\MLdata\\ASFormer/data/gtea/groundTruth/S4_CofHoney_C1.txt', 'D:\\MLdata\\ASFormer/data/gtea/groundTruth/S4_Tea_C1.txt', 'D:\\MLdata\\ASFormer/data/gtea/groundTruth/S3_Coffee_C1.txt', 'D:\\MLdata\\ASFormer/data/gtea/groundTruth/S3_CofHoney_C1.txt', 'D:\\MLdata\\ASFormer/data/gtea/groundTruth/S4_Coffee_C1.txt', 'D:\\MLdata\\ASFormer/data/gtea/groundTruth/S2_Hotdog_C1.txt', 'D:\\MLdata\\ASFormer/data/gtea/groundTruth/S3_Cheese_C1.txt', 'D:\\MLdata\\ASFormer/data/gtea/groundTruth/S2_Cheese_C1.txt', 'D:\\MLdata\\ASFormer/data/gtea/groundTruth/S2_Pealate_C1.txt', 'D:\\MLdata\\ASFormer/data/gtea/groundTruth/S4_Cheese_C1.txt', 'D:\\MLdata\\ASFormer/data/gtea/groundTruth/S2_CofHoney_C1.txt', 'D:\\MLdata\\ASFormer/data/gtea/groundTruth/S3_Tea_C1.txt', 'D:\\MLdata\\ASFormer/data/gtea/groundTruth/S3_Hotdog_C1.txt', 'D:\\MLdata\\ASFormer/data/gtea/groundTruth/S2_Coffee_C1.txt', 'D:\\MLdata\\ASFormer/data/gtea/groundTruth/S4_Hotdog_C1.txt', 'D:\\MLdata\\ASFormer/data/gtea/groundTruth/S4_Peanut_C1.txt', 'D:\\MLdata\\ASFormer/data/gtea/groundTruth/S2_Tea_C1.txt', 'D:\\MLdata\\ASFormer/data/gtea/groundTruth/S2_Peanut_C1.txt', 'D:\\MLdata\\ASFormer/data/gtea/groundTruth/S4_Pealate_C1.txt', 'D:\\MLdata\\ASFormer/data/gtea/groundTruth/S3_Peanut_C1.txt']
    features {list:21}          ['D:\\MLdata\\ASFormer/data/gtea/features/S3_Pealate_C1.npy', 'D:\\MLdata\\ASFormer/data/gtea/features/S4_CofHoney_C1.npy', 'D:\\MLdata\\ASFormer/data/gtea/features/S4_Tea_C1.npy', 'D:\\MLdata\\ASFormer/data/gtea/features/S3_Coffee_C1.npy', 'D:\\MLdata\\ASFormer/data/gtea/features/S3_CofHoney_C1.npy', 'D:\\MLdata\\ASFormer/data/gtea/features/S4_Coffee_C1.npy', 'D:\\MLdata\\ASFormer/data/gtea/features/S2_Hotdog_C1.npy', 'D:\\MLdata\\ASFormer/data/gtea/features/S3_Cheese_C1.npy', 'D:\\MLdata\\ASFormer/data/gtea/features/S2_Cheese_C1.npy', 'D:\\MLdata\\ASFormer/data/gtea/features/S2_Pealate_C1.npy', 'D:\\MLdata\\ASFormer/data/gtea/features/S4_Cheese_C1.npy', 'D:\\MLdata\\ASFormer/data/gtea/features/S2_CofHoney_C1.npy', 'D:\\MLdata\\ASFormer/data/gtea/features/S3_Tea_C1.npy', 'D:\\MLdata\\ASFormer/data/gtea/features/S3_Hotdog_C1.npy', 'D:\\MLdata\\ASFormer/data/gtea/features/S2_Coffee_C1.npy', 'D:\\MLdata\\ASFormer/data/gtea/features/S4_Hotdog_C1.npy', 'D:\\MLdata\\ASFormer/data/gtea/features/S4_Peanut_C1.npy', 'D:\\MLdata\\ASFormer/data/gtea/features/S2_Tea_C1.npy', 'D:\\MLdata\\ASFormer/data/gtea/features/S2_Peanut_C1.npy', 'D:\\MLdata\\ASFormer/data/gtea/features/S4_Pealate_C1.npy', 'D:\\MLdata\\ASFormer/data/gtea/features/S3_Peanut_C1.npy']
    '''

    batch_gen_tst = BatchGenerator(num_classes, actions_dict,  features_path, sample_rate)
    batch_gen_tst.read_data('test')

    trainer.train(model_dir, batch_gen, num_epochs, bz, lr, batch_gen_tst)

if args.action == "predict":
    # model_dir = "./{}/".format(args.model_dir) + args.dataset  # +"/split_"+args.split
    # results_dir = "./{}/".format(args.result_dir) + args.dataset  # +"/split_"+args.split
    model_dir = model_dir + '/' + args.predict_folder
    results_dir = results_dir + '/' + args.predict_folder

    batch_gen_tst = BatchGenerator(num_classes, actions_dict,  features_path, sample_rate)
    batch_gen_tst.read_data('test')
    if not os.path.exists(results_dir):
        print('{} does not exists'.format(results_dir))
        os.mkdir(results_dir)
        print('mkdir {}'.format(results_dir))
    trainer.predict(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict, sample_rate)

