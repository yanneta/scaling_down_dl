#!/usr/bin/env python
import argparse
import torch
from utils.prepare_data import * 
from models.models import *

parser = argparse.ArgumentParser(description='PyTorch Resnet Training')
parser.add_argument('--dataset', default='', type=str,
                    help='mura, chexpert or rsna')

args = parser.parse_args()
dataset = args.dataset

batch_size = 32

train_loader, valid_loader, valid_dataset = get_dataloaders(dataset, batch_size)


lrs = {"chexpert": 0.001, "mura": 0.02, "rsna": 0.1}
num_classes_dict = {"chexpert": 5, "mura":1, "rsna": 1}
lr = lrs[dataset]
num_classes = num_classes_dict[dataset]
binary = True
if dataset == "chexpert":
    binary = False



widths = [1.0, 0.75, 0.5, 0.25]
depths = [[[[64, 2], [128, 2]], [[256, 2], [512, 1]]],
          [[[64, 2], [128, 2]], [[256, 1], [512, 1]]],
          [[[64, 2], [128, 1]], [[256, 1], [512, 1]]],
          [[[64, 2], [128, 1]], [[256, 2], [512, 1]]],
          [[[64, 1], [128, 1]], [[256, 2], [512, 1]]],
          [[[64, 1], [128, 1]], [[256, 1], [512, 1]]],
         ]


data = []
total = len(widths)*len(depths)*3
counter = 0
for w in widths:
    for d in depths:
        for seed in range(1, 4):
            d_s = sum(j[1] for i in d for j in i)
            print('width multiplier - %.3f depth multiplier - %.3f' % (w, d_s))
            model = resnet18(num_classes=num_classes, block=depthwise_block, width_mult=w, 
                         inverted_residual_setting1=d[0], 
                         inverted_residual_setting2=d[1]).cuda()
        
            p = sum(p.numel() for p in model.parameters())
            optimizer = create_optimizer(model, 0.001)
            score, t = train_triangular_policy(model, optimizer, train_loader, valid_loader, valid_dataset,
                                       loss_fn=F.binary_cross_entropy_with_logits, 
                                       dataset=dataset, binary=binary, max_lr=lr, epochs=15)
        
            counter += 1
            data.append([w, d_s, score, p, t])
            print(dataset, w, d_s, score, p, t)

filename = "logs/" + dataset + "resnet_depthwise.csv"
columns = ['width_x', 'depth_x', 'val_score', 'params', 'time_per_epoch']
df = pd.DataFrame(data=data, columns=columns)
df.to_csv(filename, index=False)


