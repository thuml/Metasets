import torch
from torch.nn import functional as F
import sys
import os
import numpy as np
from dataloader import *
from torch.utils.data import DataLoader

model_path = sys.argv[2]
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

@torch.no_grad()
def load_model(model_path):
    model = torch.load(model_path).module.to('cuda:0')
    model.eval()
    print(model)
    return model


@torch.no_grad()
def evaluate_dataset(model):
    num_class=11
    dataset = PaddingData(pc_input_num=2048, status='test', aug=False, pc_root=sys.argv[1])
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, drop_last=False)
    correct_total = 0
    data_total = 0
    loss_total = 0
    acc_class = torch.zeros(num_class, 1)
    acc_to_class = torch.zeros(num_class, 1)
    for data, pn, label in dataloader:
        data = data.to('cuda:0')
        pn = pn.to('cuda:0')
        label = label.to('cuda:0')
        
        output = model.forward(data, pn)
        loss = F.cross_entropy(output, label)

        _, pred = torch.max(output, 1)

        acc = pred == label
 
        for j in range(0, num_class):
            label_j_list = (label == j)
            acc_class[j] += (pred[acc] == j).sum().cpu().float()
            acc_to_class[j] += label_j_list.sum().cpu().float()
        correct_total += torch.sum(pred == label).item()
        data_total += data.size(0)
        loss_total += loss.item() * data.size(0)
    avg_sum = 0
    for j in range(0, num_class):
        avg_sum += acc_class[j] / acc_to_class[j]
        print(acc_class[j] / acc_to_class[j])
    print('avg:', avg_sum / 11)
    print(correct_total / data_total)
    print(loss_total / data_total)


def main():
    model = load_model(model_path)
    evaluate_dataset(model)

if __name__ == '__main__':
    main()
