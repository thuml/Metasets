import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from model_pointnet_meta import Pointnet_cls_meta
from torchmeta.modules import DataParallel as MetaDataParallel
from torchmeta.utils.gradient_based import gradient_update_parameters
from dataloader import *
import time
import os
import argparse
import numpy as np

# Command setting
parser = argparse.ArgumentParser(description='Main')
parser.add_argument('-source', '-s', type=str, help='source dataset', default='modelnet_11')
parser.add_argument('-target', '-t', type=str, help='target dataset', default='scanobjectnn_11')
parser.add_argument('-batchsize', '-b', type=int, help='batch size', default=64)
parser.add_argument('-gpu', '-g', type=str, help='cuda id', default='0')
parser.add_argument('-epochs', '-e', type=int, help='training epoch', default=40)
parser.add_argument('-lr', type=float, help='learning rate', default=0.003)
args = parser.parse_args()


device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

BATCH_SIZE = args.batchsize * len(args.gpu.split(','))
LR = 0.001
INNER_LR = 0.0003
weight_decay = 5e-4
momentum = 0.9
max_epoch = args.epochs
num_class = 11 if args.source == 'modelnet_11' else 9
task_num = 4
swapax = args.source == 'modelnet_11'

def main():
    print ('Start Training\nInitiliazing\n')
    print('src:', args.source)
    print('tar:', args.target)
    source_train_dataset = BatchPaddingData(pc_input_num=2048, status='train', aug=True, swapax=swapax, pc_root='./data/%s' % args.source, batch_size=32, sample_num=task_num)
    source_test_dataset = PaddingData(pc_input_num=2048, status='test', aug=False, swapax=swapax, pc_root='./data/%s' % args.source)
    target_test_dataset = PaddingData(pc_input_num=2048, status='test', aug=False, pc_root='./data/%s' % args.target)

    test_datasets = [
        PaddingData(pc_input_num=2048, status='test', aug=False, drop=0.24, swapax=swapax, pc_root='./data/%s' % args.source),
        PaddingData(pc_input_num=2048, status='test', aug=False, drop=0.36, swapax=swapax, pc_root='./data/%s' % args.source),
        PaddingData(pc_input_num=2048, status='test', aug=False, drop=0.45, swapax=swapax, pc_root='./data/%s' % args.source),

        PaddingData(pc_input_num=2048, status='test', aug=False, p_scan=0.017, swapax=swapax, pc_root='./data/%s' % args.source),
        PaddingData(pc_input_num=2048, status='test', aug=False, p_scan=0.022, swapax=swapax, pc_root='./data/%s' % args.source),
        PaddingData(pc_input_num=2048, status='test', aug=False, p_scan=0.035, swapax=swapax, pc_root='./data/%s' % args.source),

        PaddingData(pc_input_num=2048, status='test', aug=False, density=1.3, swapax=swapax, pc_root='./data/%s' % args.source),
        PaddingData(pc_input_num=2048, status='test', aug=False, density=1.4, swapax=swapax, pc_root='./data/%s' % args.source),
        PaddingData(pc_input_num=2048, status='test', aug=False, density=1.6, swapax=swapax, pc_root='./data/%s' % args.source),

        PaddingData(pc_input_num=2048, status='test', aug=False, swapax=swapax, pc_root='./data/%s' % args.source),


    ]
    test_dataloaders = [DataLoader(dset, batch_size=32, shuffle=True, num_workers=8, drop_last=True) for dset in test_datasets]

    num_source_train = len(source_train_dataset)
    num_source_test = len(source_test_dataset)
    num_target_test = len(target_test_dataset)
    source_train_dataloader = DataLoader(source_train_dataset, batch_size=1, shuffle=True, num_workers=4, drop_last=True)
    source_test_dataloader = DataLoader(source_test_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4, drop_last=False)
    target_test_dataloader = DataLoader(target_test_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4, drop_last=False)

    # Model

    model = Pointnet_cls_meta(num_class=num_class)
    model = model.to(device=device)
    model = MetaDataParallel(model)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device=device)


    # Optimizer
    remain_epoch=50

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    lr_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs+remain_epoch)
   
    #Meta Learning

    best_source_test_acc = 0
    best_target_test_acc = 0
    

    for epoch in range(max_epoch):
        data_total = 0
        for batch_idx, batch in enumerate(source_train_dataloader):
            model.train()
            model.zero_grad()
            data, pn, label = batch
            label = label.to(device=device).long().reshape(task_num, -1)
            pn = pn.to(device=device).long().reshape(task_num, -1)
            data = data.to(device=device).reshape(task_num, -1, 3, 2048, 1)
    
            outer_loss = torch.tensor(0., device='cuda')
            predictions = []
            accuracy = torch.tensor(0., device='cuda')
            
            for i in range(task_num):
                logits = model(data[i], pn[i])
                inner_loss = F.cross_entropy(logits, label[i])
                model.zero_grad()
                params = gradient_update_parameters(model,
                                                    inner_loss,
                                                    step_size=INNER_LR / LR * lr_schedule.get_last_lr()[0])
                logits = model(data[i], pn[i], params=params)
                with torch.no_grad():
                    accuracy += (torch.argmax(logits, dim=1) == label[i]).float().mean()
                outer_loss += F.cross_entropy(logits, label[i])

            outer_loss.div_(task_num)
            accuracy.div_(task_num)
            outer_loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 10 == 0:
                print('Train:{} [{} /{}  loss: {:.4f} accuracy: {:.4f}\t]'.format(
                epoch, batch_idx, num_source_train, outer_loss.item(), accuracy.item()))

            if (batch_idx + 1) % 100 == 0:
                lr_schedule.step()
                print('Current lr:', lr_schedule.get_last_lr()[0])
                with torch.no_grad():
                    model.eval()
        
                    # ------------Source------------
                    loss_total = 0
                    correct_total = 0
                    data_total = 0
        
                    for batch_idx, (data, pn, label) in enumerate(source_test_dataloader):
        
                        data = data.to(device=device)
                        pn = pn.to(device=device).long()
                        label = label.to(device=device).long()
                        output = model(data, pn)
                        loss = F.cross_entropy(output, label)
                        _, pred = torch.max(output, 1)
        
        
                        loss_total += loss.item() * data.size(0)
                        correct_total += torch.sum(pred == label)
                        data_total += data.size(0)
        
                    pred_loss = loss_total / data_total
                    pred_acc = correct_total.double() / data_total
        
                    if pred_acc > best_source_test_acc:
                        best_source_test_acc = pred_acc
                    print('Source Test:{} [overall_acc: {:.4f} \t loss: {:.4f} \t Best Source Test Acc: {:.4f}]'.format(
                        epoch, pred_acc, pred_loss, best_source_test_acc
                    ))
                    # ------------Target------------
                    loss_total = 0
                    correct_total = 0
                    data_total = 0
        
                    for batch_idx, (data, pn, label) in enumerate(target_test_dataloader):
        
                        data = data.to(device=device)
                        pn = pn.to(device=device).long()
                        label = label.to(device=device).long()
                        output = model(data, pn)
                        loss = F.cross_entropy(output, label)
                        _, pred = torch.max(output, 1)
        
                        loss_total += loss.item() * data.size(0)
                        correct_total += torch.sum(pred == label)
                        data_total += data.size(0)
        
                    pred_loss = loss_total / data_total
                    pred_acc = correct_total.double() / data_total
        
                    if pred_acc > best_target_test_acc:
                        best_target_test_acc = pred_acc
                    print('Target:{} [overall_acc: {:.4f} \t loss: {:.4f} \t Best Target 2 Acc: {:.4f}]'.format(
                        epoch, pred_acc, pred_loss, best_target_test_acc
                    ))
        print('Updating task p')
        task_acc = []
        task_loss = []
        with torch.no_grad():
            model.eval()
            for loader in test_dataloaders:
                p_correct_total = 0
                p_data_total = 0
                p_loss_total = 0

                for data, pn, label in loader:
                    data = data.to(device=device)
                    pn = pn.to(device=device).long()
                    label = label.to(device=device).long()
                    output = model(data, pn)
                    loss = F.cross_entropy(output, label)
                    _, pred = torch.max(output, 1)

                    p_loss_total += loss.item() * data.size(0)
                    p_correct_total += torch.sum(pred == label)
                    p_data_total += data.size(0)

                pred_loss = p_loss_total / p_data_total
                pred_acc = p_correct_total.double() / p_data_total
                task_acc.append(pred_acc.item())
                task_loss.append(pred_loss)
        exp_loss = np.exp(task_loss)
        task_p = exp_loss / exp_loss.sum()
        print('Prev task p:', source_train_dataset.task_p)
        print('Task acc:', task_acc)
        print('task loss:', task_loss)
        print('New task p:', task_p)
        source_train_dataset.update_p(task_p)


if __name__ == '__main__':
    since = time.time()
    main()
    time_pass = since - time.time()
    print('Training complete in {:.0f}m {:.0f}s'.format(time_pass // 60, time_pass % 60))

