import os
import pickle
import random
from os import path as osp
from pathlib import Path
import numpy as np

import torch
from torch.utils import data as data
import yaml

from lbasicsr.data.data_util import arbitrary_scale_downsample, generate_frame_indices
from lbasicsr.data.transforms import augment, paired_random_crop, single_random_crop, single_random_spcrop, mod_crop
from lbasicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from lbasicsr.utils.matlab_functions import imresize
from lbasicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Adobe240Dataset(data.Dataset):
    """Vimeo90K dataset for training.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_Vimeo90K_train_GT.txt

    Each line contains the following items, separated by a white space.

    1. clip name;
    2. frame number;
    3. image shape

    Examples:

    ::
        00001/0001 7 (256,448,3)
        00001/0002 7 (256,448,3)

    - Key examples: "00001/0001"
    - GT (gt): Ground-Truth;
    - LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    The neighboring frame list for different num_frame:

    ::

        num_frame | frame list
                1 | 4
                3 | 3,4,5
                5 | 2,3,4,5,6
                7 | 1,2,3,4,5,6,7

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        num_frame (int): Window size for input frames.
        gt_size (int): Cropped patched size for gt patches.
        random_reverse (bool): Random reverse input frames.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(Adobe240Dataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])

        # with open(opt['meta_info_file'], 'r') as fin:
        #     self.keys = [line.split(' ')[0] for line in fin]
        # ------ directly load image keys ------------------------------------------
        logger = get_root_logger()
        if opt['cache_keys']:
            logger.info('Using cache keys: {}'.format(opt['cache_keys']))
            cache_keys = opt['cache_keys']
        else:
            cache_keys = 'Vimeo7_train_keys.pkl'
        logger.info('Using cache keys - {}.'.format(cache_keys))
        self.paths_gt = pickle.load(open(cache_keys, 'rb'))     # list['00001_0001', 00001_0002, ..., 00096_0936], len(): 64612
        assert self.paths_gt, 'Error: GT path is empty.'

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
        
        self.half_num_frame = opt['num_frame'] // 2
        self.lr_num_frame = self.half_num_frame + 1
        assert self.lr_num_frame > 1, 'Error: Not enough LR frames to interpolate'
        # determine the LR frame list
        """
        N | frames
        1 | error
        3 | 0,2
        5 | 0,2,4
        7 | 0,2,4,6
        """
        self.lr_index_list = [i*2 for i in range(self.lr_num_frame)]
        self.lr_input = False if opt['gt_size'] == opt['lq_size'] else True
        
        with open(opt['meta_info_file']) as t:
            video_list = t.readlines()
        
        self.file_list = []
        self.gt_list = []
        for video in video_list:
            if video[-1] == '\n':
                video = video[:-1]
            index = 0
            interval = 7
            frames = (os.listdir(osp.join(self.gt_root, video)))
            frames = sorted([int(frame[:-4]) for frame in frames])
            frames = [str(frame) + '.png' for frame in frames]
            while index + interval*1 + 1 < len(frames):
                video_inputs_index = [index, index + 1 + interval]
                video_inputs = [frames[i] for i in video_inputs_index]
                video_all_gt = [frames[i] for i in range(index, index + 2 + interval * 1)]
                video_inputs = [osp.join(video, f) for f in video_inputs]
                video_gts = [osp.join(video, f) for f in video_all_gt]
                self.file_list.append(video_inputs)
                self.gt_list.append(video_gts)
                index += 1

        # temporal augmentation configs
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info(f'Interval list is {self.interval_list}.')
        logger.info(f'Random reverse is {self.random_reverse}.')
        
        logger.info(f'Length of file list: {len(self.file_list)}')
        logger.info(f'Length of gt list: {len(self.gt_list)}')
        
        self.as_down_sample = False

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        
        # scale = self.opt['scale']
        scale = 4
        gt_size = self.opt['gt_size']
        num_frame = self.opt['num_frame']
        key = self.paths_gt[0]              # '00001_0001'
        
        center_frame_idx = random.randint(2, 6)     # 2<= index <=6
        
        # determine the neighbor frames
        interval = random.choice(self.interval_list)
        if self.opt['border_mode']:
            direction = 1  # 1: forward; 0: backward
            if self.random_reverse and random.random() < 0.5:
                direction = random.choice([0, 1])
            if center_frame_idx + interval * (num_frame - 1) > 7:
                direction = 0
            elif center_frame_idx - interval * (num_frame - 1) < 1:
                direction = 1
            # get the neighbor list
            if direction == 1:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx + interval * num_frame, interval))
            else:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx - interval * num_frame, -interval))
        else:
            # ensure not exceeding the borders
            while (center_frame_idx + self.half_num_frame * interval > 7) or (center_frame_idx - self.half_num_frame * interval < 1):
                center_frame_idx = random.randint(2, 6)
            # get the neighbor list
            neighbor_list = list(
                range(center_frame_idx - self.half_num_frame * interval,
                      center_frame_idx + self.half_num_frame * interval + 1, interval))
            if self.random_reverse and random.random() < 0.5:
                neighbor_list.reverse()

        # get the GT image (as the center frame)
        img_gt_lst = []
        img_lqop_lst = [osp.join(self.lq_root, fp) for fp in self.file_list[index]]
        img_gtop_lst = np.array([osp.join(self.gt_root, fp) for fp in self.gt_list[index]])
        
        gt_sampled_idx = sorted(random.sample(range(len(img_gtop_lst)), 1))
        img_gtop_lst = img_gtop_lst[gt_sampled_idx]
        
        times = []
        for i in gt_sampled_idx:
            times.append(torch.tensor([i / 8]))
        
        img_lq_lst = [self.file_client.get(fp) for fp in img_lqop_lst]
        img_lq_lst = [imfrombytes(img_lq, float32=True) for img_lq in img_lq_lst]
        
        img_gt_lst = [self.file_client.get(fp) for fp in img_gtop_lst]
        img_gt_lst = [imfrombytes(img_gt, float32=True) for img_gt in img_gt_lst]
        
        img_lq_lst = [imresize(lq_, 1 / scale, True) for lq_ in img_lq_lst]
        
        if self.opt['phase'] == 'train':
            if self.lr_input:
                # randomly crop
                img_gt, img_lqs = paired_random_crop(img_gt_lst, img_lq_lst, gt_size, scale)
            else:
                pass
            
            # augmentation - flip, rotate
            img_lqs.append(img_gt)
            img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

            img_results = img2tensor(img_results)
            img_lqs = torch.stack(img_results[0:-1], dim=0)
            img_gt = img_results[-1]

        # img_lqs: (t, c, h, w)
        # img_gt: (c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gt, 'key': key, 'time': times, 'scale': (img_gt.shape[-2], img_gt.shape[-1])}

    def __len__(self):
        return len(self.file_list)


@DATASET_REGISTRY.register()
class ASAdobe240Dataset(Adobe240Dataset):

    def __init__(self, opt):
        super(ASAdobe240Dataset, self).__init__(opt)
        self.as_down_sample = False

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        
        # scale = self.opt['scale']
        scale = 4
        gt_size = self.opt['gt_size']
        num_frame = self.opt['num_frame']
        key = self.paths_gt[0]              # '00001_0001'
        
        center_frame_idx = random.randint(2, 6)     # 2<= index <=6
        
        # determine the neighbor frames
        interval = random.choice(self.interval_list)
        if self.opt['border_mode']:
            direction = 1  # 1: forward; 0: backward
            if self.random_reverse and random.random() < 0.5:
                direction = random.choice([0, 1])
            if center_frame_idx + interval * (num_frame - 1) > 7:
                direction = 0
            elif center_frame_idx - interval * (num_frame - 1) < 1:
                direction = 1
            # get the neighbor list
            if direction == 1:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx + interval * num_frame, interval))
            else:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx - interval * num_frame, -interval))
        else:
            # ensure not exceeding the borders
            while (center_frame_idx + self.half_num_frame * interval > 7) or (center_frame_idx - self.half_num_frame * interval < 1):
                center_frame_idx = random.randint(2, 6)
            # get the neighbor list
            neighbor_list = list(
                range(center_frame_idx - self.half_num_frame * interval,
                      center_frame_idx + self.half_num_frame * interval + 1, interval))
            if self.random_reverse and random.random() < 0.5:
                neighbor_list.reverse()

        # get the GT image (as the center frame)
        img_gt_lst = []
        img_lqop_lst = [osp.join(self.gt_root, fp) for fp in self.file_list[index]]
        img_gtop_lst = np.array([osp.join(self.gt_root, fp) for fp in self.gt_list[index]])
        
        gt_sampled_idx = sorted(random.sample(range(len(img_gtop_lst)), 3))
        img_gtop_lst = img_gtop_lst[gt_sampled_idx]
        
        times = []
        for i in gt_sampled_idx:
            times.append(torch.tensor([i / 8]))
        
        img_lqo_lst = [self.file_client.get(fp) for fp in img_lqop_lst]
        img_lqo_lst = [imfrombytes(img_lq, float32=True) for img_lq in img_lqo_lst]
        
        img_gto_lst = [self.file_client.get(fp) for fp in img_gtop_lst]
        img_gto_lst = [imfrombytes(img_gt, float32=True) for img_gt in img_gto_lst]
        
        return img_lqo_lst, img_gto_lst, times, img_lqop_lst
    
    def set_as_mode(self, iter):
        logger = get_root_logger()
        logger.info(f'From {iter}, start the arbitrary-scale downsampling mode.')
        
        self.as_down_sample = True
    
    def as_collate_fn(self, batch):
        '''
        We want to create a dataloader, which would randomly select a down-sampling scale for each batch.
        If down-sampling is performed in __getitem__ function, when num_workers > 1,
        each subprocess would have a different scale, resulting in unmatched resolutions.
        Therefore, we define a collate function for down-sampling.
        In this function, we fix resolutions of down-sampled input images to (64, 64)
        and set resolutions of GT images to (64 * d_scale, 64 * d_scale), where d_scale is the randomly sampled scale.

        For the operation of temporal sampling, see line 189 - 191 in Adobe_arbitrary.py
        '''
        
        d_scale = random.uniform(2, 4)      # randomly select down-sampling scale in [2, 4]
        lq_size = 64
        gt_size = int(np.floor(lq_size * d_scale))
        
        # img crop
        x = random.randint(0, max(0, 720 - gt_size))
        y = random.randint(0, max(0, 1280 - gt_size))
        img_lq_lst = [np.stack([img_[0][i][x:x+gt_size,y:y+gt_size] 
                                if img_[0][i].shape[0] == 720 else img_[0][i][y:y+gt_size,x:x+gt_size] for img_ in batch], axis=0) 
                      for i in range(len(batch[0][0]))]
        img_gt_lst = [np.stack([img_[1][i][x:x+gt_size,y:y+gt_size] 
                                if img_[1][i].shape[0] == 720 else img_[1][i][y:y+gt_size,x:x+gt_size] for img_ in batch], axis=0) 
                      for i in range(len(batch[0][1]))]
        
        # down-sampling
        img_lq_lst = [np.stack([imresize(img_[i], 1/(2*d_scale)) for i in range(img_.shape[0])], axis=0) for img_ in img_lq_lst]
        img_gt_lst = [np.stack([imresize(img_[i], 1/2) for i in range(img_.shape[0])], axis=0) for img_ in img_gt_lst]
        
        img_lqs = np.stack(img_lq_lst, axis=0)
        img_gts = np.stack(img_gt_lst, axis=0)
        
        # augmentation - flip, rotate
        img_lqs, img_gts = augment_a2(img_lqs, img_gts, hflip=True, rot=True)
        
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gts = img_gts[:, :, :, :, [2, 1, 0]]
        img_lqs = img_lqs[:, :, :, :, [2, 1, 0]]

        img_gts = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gts, (1, 0, 4, 2, 3)))).float()
        img_lqs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lqs, (1, 0, 4, 2, 3)))).float()

        time_t = [torch.cat([time_[2][i][None] for time_ in batch], dim=0) for i in range(len(batch[0][2]))]

        return {'lq': img_lqs, 'gt': img_gts, 'time': time_t, 'scale': [[img_gts.shape[-2]], [img_gts.shape[-1]]]}
        

def augment_a2(img_LQ, img_GT, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5
    if hflip:
        img_LQ = img_LQ[:, :, :, ::-1, :]
        img_GT = img_GT[:, :, :, ::-1, :]
    if vflip:
        img_LQ = img_LQ[:, :, ::-1, :, :]
        img_GT = img_GT[:, :, ::-1, :, :]
    if rot90:
        img_LQ = img_LQ.transpose(0, 1, 3, 2, 4)
        img_GT = img_GT.transpose(0, 1, 3, 2, 4)
    return img_LQ, img_GT


if __name__ == '__main__':
    opt_str = r"""
name: Test
type: Adobe240Dataset
dataroot_gt: datasets/Adobe240/frame/train
dataroot_lq: datasets/Adobe240/frame/train
cache_keys: lbasicsr/data/meta_info/Vimeo7_train_keys.pkl
meta_info_file: lbasicsr/data/meta_info/adobe240fps_folder_train.txt
io_backend:
    type: disk

num_frame: 7
gt_size: 128
lq_size: 32
interval_list: [1]
random_reverse: true
border_mode: false
use_hflip: true
use_rot: true
flip_sequence: false
phase: train
"""
    opt = yaml.safe_load(opt_str)
    
    dataset = Adobe240Dataset(opt)
    
    # test __gititem__
    result = dataset.__getitem__(0)
    print(result)
    
        
