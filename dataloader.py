import torch
import torch.utils.data as data
import os
import sys
import h5py
import numpy as np
from multiprocessing.dummy import Pool
from torchvision import transforms
import glob
import random
import threading
import time
from data_utils import *


class PaddingData(data.Dataset):
    def __init__(self, pc_root, aug=False, status='train', pc_input_num=2048, density=0, drop=0, p_scan=0, swapax=False):
        super(PaddingData, self).__init__()

        self.status = status

        self.density = density
        self.drop = drop
        self.p_scan = p_scan

        self.aug = aug
        self.pc_list = []
        self.lbl_list = []
        self.transforms = transforms.Compose(
            [
                PointcloudToTensor(),
                PointcloudScale(),
                PointcloudRotate(),
                PointcloudRotatePerturbation(),
                PointcloudTranslate(),
                PointcloudJitter(),
            ]
        )
        self.pc_input_num = pc_input_num

        categorys = glob.glob(os.path.join(pc_root, '*'))
        categorys = [c.split(os.path.sep)[-1] for c in categorys]
        categorys = sorted(categorys)
        print(categorys)
        if self.density > 0:
            rand_points = np.random.uniform(-1, 1, 40000)
            x1 = rand_points[:20000]
            x2 = rand_points[20000:]
            power_sum = x1 ** 2 + x2 ** 2
            p_filter = power_sum < 1
            power_sum = power_sum[p_filter]
            sqrt_sum = np.sqrt(1 - power_sum)
            x1 = x1[p_filter]
            x2 = x2[p_filter]
            x = (2 * x1 * sqrt_sum).reshape(-1, 1)
            y = (2 * x2 * sqrt_sum).reshape(-1, 1)
            z = (1 - 2 * power_sum).reshape(-1, 1)
            self.density_points = np.hstack([x, y, z])
        if status == 'train':
            npy_list = glob.glob(os.path.join(pc_root, '*', 'train', '*.npy'))
        else:
            npy_list = glob.glob(os.path.join(pc_root, '*', 'test', '*.npy'))

        for idx, _dir in enumerate(npy_list):
            print("\r%d/%d" % (idx, len(npy_list)), end="")
            pc = np.load(_dir).astype(np.float32)
            if swapax:
                pc[:, 1] = pc[:, 2] + pc[:, 1]
                pc[:, 2] = pc[:, 1] - pc[:, 2]
                pc[:, 1] = pc[:, 1] - pc[:, 2]
            self.pc_list.append(pc)
            self.lbl_list.append(categorys.index(_dir.split('/')[-3]))
        print()

        print(f'{status} data num: {len(self.pc_list)}')

    def __getitem__(self, idx):
        lbl = self.lbl_list[idx]
        pc = self.pc_list[idx]
        pc = normal_pc(pc)
        if self.density > 0:
            v_point = self.density_points[np.random.choice(self.density_points.shape[0])]
            pc = density(pc, v_point, self.density)
        if self.drop > 0:
            pc = drop_hole(pc, p=self.drop)
        if self.p_scan > 0:
            pc = p_scan(pc, pixel_size=self.p_scan)
        pn = min(pc.shape[0], self.pc_input_num)
        if self.aug:
            pc = self.transforms(pc)
            pc = pc.numpy()
        if pn < self.pc_input_num:
            pc = np.append(pc, np.zeros((self.pc_input_num - pc.shape[0], 3)), axis=0)
        pc = pc[:self.pc_input_num]
        pc = np.expand_dims(pc.transpose(), axis=2)
        return torch.from_numpy(pc).type(torch.FloatTensor), pn, lbl

    def __len__(self):
        return len(self.pc_list)


class BatchPaddingData(PaddingData):
    def __init__(self, pc_root, aug=False, status='train', pc_input_num=2048, swapax=False, batch_size=32, sample_num=5):
        super(BatchPaddingData, self).__init__(pc_root, aug, status, pc_input_num, swapax=swapax)
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.batch_count = int(len(self.pc_list) / batch_size * 3)
        self.pc_list = np.array(self.pc_list)
        self.lbl_list = np.array(self.lbl_list)

        rand_points = np.random.uniform(-1, 1, 40000)
        x1 = rand_points[:20000]
        x2 = rand_points[20000:]
        power_sum = x1 ** 2 + x2 ** 2
        p_filter = power_sum < 1
        power_sum = power_sum[p_filter]
        sqrt_sum = np.sqrt(1 - power_sum)
        x1 = x1[p_filter]
        x2 = x2[p_filter]
        x = (2 * x1 * sqrt_sum).reshape(-1, 1)
        y = (2 * x2 * sqrt_sum).reshape(-1, 1)
        z = (1 - 2 * power_sum).reshape(-1, 1)
        self.density_points = np.hstack([x, y, z])
        self.fn = [
            lambda pc: drop_hole(pc, p=0.24),
            lambda pc: drop_hole(pc, p=0.36),
            lambda pc: drop_hole(pc, p=0.45),
            lambda pc: p_scan(pc, pixel_size=0.017),
            lambda pc: p_scan(pc, pixel_size=0.022),
            lambda pc: p_scan(pc, pixel_size=0.035),
            lambda pc: density(pc, self.density_points[np.random.choice(self.density_points.shape[0])], 1.3),
            lambda pc: density(pc, self.density_points[np.random.choice(self.density_points.shape[0])], 1.4),
            lambda pc: density(pc, self.density_points[np.random.choice(self.density_points.shape[0])], 1.6),
            lambda pc: pc.copy(),
        ]
        self.task_num = len(self.fn)
        self.task_p = [1 / self.task_num] * self.task_num
        self.task_index = list(range(self.task_num))

        self.create_batch(self.batch_count)

    def create_batch(self, batchsz):
        self.batch_x = []
        self.batch_y = []
        for b in range(batchsz):
            print("\rCreating Batch %d/%d" % (b, batchsz), end="")
            selected_imgs_idx = np.random.choice(self.pc_list.shape[0], self.batch_size, False)
            self.batch_x.append(self.pc_list[selected_imgs_idx])
            self.batch_y.append(self.lbl_list[selected_imgs_idx])
        print()

    def process_batch(self, pc):
        task_num = self.selected_tasks.shape[0]
        task_data = []
        task_pn = []
        pc = normal_pc(pc)
        for task_id in self.selected_tasks:
            tmp_pc = self.fn[task_id](pc)
            if self.aug:
                tmp_pc = self.transforms(tmp_pc)
                tmp_pc = tmp_pc.numpy()
            pn = min(tmp_pc.shape[0], self.pc_input_num)
            if pn < self.pc_input_num:
                tmp_pc = np.append(tmp_pc, np.zeros((self.pc_input_num - tmp_pc.shape[0], 3)), axis=0)
            tmp_pc = tmp_pc[:self.pc_input_num]
            tmp_pc = np.expand_dims(tmp_pc.transpose(), axis=2)
            task_data.append(tmp_pc)
            task_pn.append(pn)
        return np.array(task_data), np.array(task_pn)

    def update_p(self, p):
        self.task_p = p

 
    def __getitem__(self, idx):
        sample_num = min(self.sample_num, self.task_num)
        self.selected_tasks = np.random.choice(self.task_index, sample_num, p=self.task_p)
        pool = Pool(32)
        batch_data = pool.map(self.process_batch, self.batch_x[idx])
        pool.close()
        pool.join()
        batch_pc = []
        batch_pn = []
        for item in batch_data:
            batch_pc.append(item[0])
            batch_pn.append(item[1])
        batch_pc = np.swapaxes(np.array(batch_pc), 0, 1)
        batch_pn = np.swapaxes(np.array(batch_pn), 0, 1)
        batch_y = np.expand_dims(self.batch_y[idx], axis=0).repeat(sample_num, axis=0)
        return torch.from_numpy(batch_pc).type(torch.FloatTensor), batch_pn, batch_y
        

    def __len__(self):
        return self.batch_count

