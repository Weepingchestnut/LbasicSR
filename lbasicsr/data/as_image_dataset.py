import random
import numpy as np
import torch
from torch.utils import data as data
from torchvision import transforms
from torchvision.transforms.functional import normalize
from torchvision.transforms.functional import InterpolationMode
from lbasicsr.archs.arch_util import make_coord

from lbasicsr.data.data_util import paths_from_folder, paths_from_lmdb, paths_from_meta_info_file
from lbasicsr.data.transforms import augment, as_mod_crop, single_random_crop
from lbasicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from lbasicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ASINRImageDataset(data.Dataset):
    """Arbitrary-scale image dataset for INR-based arbitrary-scale image super-resolution.

    Read GT images and generate LR images, coordinate and cell.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        
        lr_patch_size (int): Cropped patched size for lr patches.
        scale_range (list): The range of scale, e.g. [min scale, max scale].
        
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(ASINRImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder = opt['dataroot_gt']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']         # ?
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paths_from_meta_info_file(self.gt_folder, self.opt['meta_info_file'])
        else:
            self.paths = paths_from_folder(self.gt_folder)
        
        self.lr_patch_size = opt.get('lr_patch_size', 48)
        scale_range = opt.get('scale_range', [1, 4])
        self.min_scale, self.max_scale = scale_range[0], scale_range[1]
        
        self.sample_quantity = opt.get('sample_quantity', None)
        self.reshape_gt = opt.get('reshape_gt', True) or opt.get('sample_quantity', None) is not None
        
        # val & test
        self.val_scale = opt.get('val_scale', (4, 4))
    
    def down_sampling(self, img, size):
        return transforms.ToTensor()(
            transforms.Resize(size, InterpolationMode.BICUBIC)(
                transforms.ToPILImage()(img)))
    
    def generate_coordinate_and_cell(self, img_gt: torch.Tensor):
        # generate hr_coord (and hr_rgb)
        self.target_size = img_gt.shape
        if self.reshape_gt:
            img_gt = img_gt.reshape(3, -1).permute(1, 0)    # [C, H, W] --> [C, H*W] --> [H*W, C]
        
        hr_coord = make_coord(self.target_size[-2:])        # [H*W, 2]
        
        if self.sample_quantity is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_quantity, replace=False)
            hr_coord = hr_coord[sample_lst]     # [2304, 2]
            img_gt = img_gt[sample_lst]         # [2304, 2]
        
        # Preparations for cell decoding
        cell = torch.ones_like(hr_coord)        # [H*W, 2]
        cell[:, 0] *= 2 / self.target_size[-2]
        cell[:, 1] *= 2 / self.target_size[-1]
        
        return img_gt, hr_coord, cell

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load gt images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        # gt_path = self.paths[index]['gt_path']
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            scale = random.uniform(self.min_scale, self.max_scale)
            gt_patch_size = round(scale * self.lr_patch_size)
            # random crop
            img_gt = single_random_crop(img_gt, (gt_patch_size, gt_patch_size))
            # flip, rotation
            img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])
            
            # BGR to RGB, HWC to CHW, numpy to tensor
            img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
            img_lq = self.down_sampling(img_gt, (self.lr_patch_size, self.lr_patch_size))

        # # color space transform
        # if 'color' in self.opt and self.opt['color'] == 'y':
        #     img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = as_mod_crop(img_gt, self.val_scale)
            # BGR to RGB, HWC to CHW, numpy to tensor
            img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
            img_lq = self.down_sampling(img_gt, (img_gt.shape[-2] // self.val_scale[0], img_gt.shape[-1] // self.val_scale[1]))
        
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        
        # Generate coordinate and cell
        img_gt, coord, cell = self.generate_coordinate_and_cell(img_gt)

        return {'lq': img_lq, 'gt': img_gt, 'gt_path': gt_path, 'coord': coord, 'cell': cell}

    def __len__(self):
        return len(self.paths)
