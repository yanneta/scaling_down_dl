#!/usr/bin/env python
import torch
import argparse
from utils.prepare_data import * 
from models.models import *

parser = argparse.ArgumentParser(description='PyTorch MobileNet Training')
parser.add_argument('--dataset', default='', type=str,
                    help='mura, chexpert or rsna')

batch_size = 32
train_loader, valid_loader, valid_dataset = get_dataloaders(dataset, batch_size)

widths = [1.0, 0.75, 0.5, 0.25]
depths = [1.0, 0.7, 0.6, 0.5, 0.3, 0.2]


lrs = {"chexpert": 0.001, "mura": 0.02, "rsna": 0.1}
num_classes_dict = {"chexpert": 5, "mura":1, "rsna": 1}
lr = lrs[dataset]
num_classes = num_classes_dict[dataset]
binary = True
if dataset == "chexpert":
    binary = False

total = len(widths)*len(depths)*3
data = []
for w in widths:
    for d in depths:
        for seed in range(1,4):
            print('width multiplier - %d depth multiplier - %d percent done %.1f' % (w, d, 100*counter/total))
            model = MobileNet(num_classes=num_classes,width_mult=w, depth_mult=d).cuda()
            p = sum(p.numel() for p in model.parameters())
            optimizer = create_optimizer(model, 0.001)
            score, t = train_triangular_policy(model, optimizer, train_loader, valid_loader, valid_dataset,
                                           loss_fn=F.binary_cross_entropy_with_logits, 
                                           dataset=dataset, binary=binary, max_lr=lr, epochs=15)

            data.append([w, d, score, p, t])
            print(dataset, "mobilenet", w, d, score, p, t)

filename = "logs/" + dataset +  "mobilenet_widths_depths.csv"
columns = ['width_x', 'depth_x', 'val_score', 'params', 'time_per_epoch']
df = pd.DataFrame(data=data, columns=columns)
df.to_csv(filename, index=False)
