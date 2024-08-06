import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.flow_util import dequantize_flow




class LDV108Dataset(data.Dataset):


    def __init__(self, opt):
        super(LDV108Dataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(
            opt['dataroot_lq'])
        self.flow_root = Path(
            opt['dataroot_flow']) if opt['dataroot_flow'] is not None else None
        assert opt['num_frame'] % 2 == 1, (
            f'num_frame should be odd number, but got {opt["num_frame"]}')
        self.num_frame = opt['num_frame']
        self.num_half_frames = opt['num_frame'] // 2

        self.keys = []

        self.frame_number = {}
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                self.keys.extend(
                    [f'{folder}/{i:03d}' for i in range(int(frame_num))])
                self.frame_number[folder] = frame_num 

        # remove the video clips used in validation
        
        self.keys = [
            v for v in self.keys if v.split('/')[0] 
        ]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            if self.flow_root is not None:
                self.io_backend_opt['db_paths'] = [
                    self.lq_root, self.gt_root, self.flow_root
                ]
                self.io_backend_opt['client_keys'] = ['lq', 'gt', 'flow']
            else:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # temporal augmentation configs
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        interval_str = ','.join(str(x) for x in opt['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000
        #print(clip_name, frame_name)
        center_frame_idx = int(frame_name)

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = center_frame_idx - self.num_half_frames * interval
        end_frame_idx = center_frame_idx + self.num_half_frames * interval
        # each clip has 100 frames starting from 0 to 99
        while (start_frame_idx < 1) or (end_frame_idx > int(self.frame_number[clip_name])):
            center_frame_idx = random.randint(0, 99)
            start_frame_idx = (
                center_frame_idx - self.num_half_frames * interval)
            end_frame_idx = center_frame_idx + self.num_half_frames * interval
        frame_name = f'{center_frame_idx:03d}'
        neighbor_list = list(
            range(center_frame_idx - self.num_half_frames * interval,
                  center_frame_idx + self.num_half_frames * interval + 1,
                  interval))
        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        assert len(neighbor_list) == self.num_frame, (
            f'Wrong length of neighbor list: {len(neighbor_list)}')

        # get the GT frame (as the center frame)
        #print(clip_name, frame_name)

        if self.is_lmdb:
            img_gt_path = f'{clip_name}/{frame_name}'
        else:
            img_gt_path = self.gt_root / clip_name / f'{frame_name}.png'
        #print(2)
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        #print(img_gt_path)
        img_gt = imfrombytes(img_bytes, float32=True)
        
        #print(3)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:03d}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:03d}.png'
            #print(3.1)
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            #print(3.2)
            #print(img_lq_path)
            img_lq = imfrombytes(img_bytes, float32=True)
            #print(3.3)
            img_lqs.append(img_lq)
        #print(4)

        # get flows
        if self.flow_root is not None:
            img_flows = []
            # read previous flows
            for i in range(self.num_half_frames, 0, -1):
                if self.is_lmdb:
                    flow_path = f'{clip_name}/{frame_name}_p{i}'
                else:
                    flow_path = (
                        self.flow_root / clip_name / f'{frame_name}_p{i}.png')
                img_bytes = self.file_client.get(flow_path, 'flow')
                cat_flow = imfrombytes(
                    img_bytes, flag='grayscale',
                    float32=False)  # uint8, [0, 255]
                dx, dy = np.split(cat_flow, 2, axis=0)
                flow = dequantize_flow(
                    dx, dy, max_val=20,
                    denorm=False)  # we use max_val 20 here.
                img_flows.append(flow)
            # read next flows
            for i in range(1, self.num_half_frames + 1):
                if self.is_lmdb:
                    flow_path = f'{clip_name}/{frame_name}_n{i}'
                else:
                    flow_path = (
                        self.flow_root / clip_name / f'{frame_name}_n{i}.png')
                img_bytes = self.file_client.get(flow_path, 'flow')
                cat_flow = imfrombytes(
                    img_bytes, flag='grayscale',
                    float32=False)  # uint8, [0, 255]
                dx, dy = np.split(cat_flow, 2, axis=0)
                flow = dequantize_flow(
                    dx, dy, max_val=20,
                    denorm=False)  # we use max_val 20 here.
                img_flows.append(flow)

            # for random crop, here, img_flows and img_lqs have the same
            # spatial size
            img_lqs.extend(img_flows)

        # randomly crop
        img_gt, img_lqs = paired_random_crop(img_gt, img_lqs, gt_size, scale,
                                             img_gt_path)
        if self.flow_root is not None:
            img_lqs, img_flows = img_lqs[:self.num_frame], img_lqs[self.
                                                                   num_frame:]

        # augmentation - flip, rotate
        img_lqs.append(img_gt)
        if self.flow_root is not None:
            img_results, img_flows = augment(img_lqs, self.opt['use_flip'],
                                             self.opt['use_rot'], img_flows)
        else:
            img_results = augment(img_lqs, self.opt['use_flip'],
                                  self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        if self.flow_root is not None:
            img_flows = img2tensor(img_flows)
            # add the zero center flow
            img_flows.insert(self.num_half_frames,
                             torch.zeros_like(img_flows[0]))
            img_flows = torch.stack(img_flows, dim=0)

        # img_lqs: (t, c, h, w)
        # img_flows: (t, 2, h, w)
        # img_gt: (c, h, w)
        # key: str
        #print(img_lqs.shape,img_gt.shape)
        if self.flow_root is not None:
            return {'lq': img_lqs, 'flow': img_flows, 'gt': img_gt, 'key': key}
        else:
            return {'lq': img_lqs, 'gt': img_gt, 'key': key}

    def __len__(self):
        return len(self.keys)
