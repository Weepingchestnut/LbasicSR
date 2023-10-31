
import os
import math
import warnings
import cv2
import numpy as np

import torch
import torch.nn.functional as F

from os import path as osp

from lbasicsr.models import build_model
from lbasicsr.utils.lam import GaussianBlurPath, Path_gradient, attr_grad, attribution_objective, saliency_map_PG
from lbasicsr.utils.lam_util import PIL2Tensor, Tensor2PIL, cv2_to_pil, grad_abs_norm, make_pil_grid, pil_to_cv2, prepare_clips, vis_saliency, vis_saliency_kde
from lbasicsr.utils.options import parse_options


def lam_vsr(model, scale, input_path, work_dir, window_size, h, w):
    img_lr, img_hr = prepare_clips(input_path, scale=scale)              # PIL [0, 255] RGB
    tensor_lrs = PIL2Tensor(img_lr)                         # tensor [0, 1] RGB
    tensor_hrs = PIL2Tensor(img_hr)
    # cv2_lr = np.moveaxis(tensor_lrs.numpy(), 0, 2)          # numpy [0, 1] BGR
    # cv2_hr = np.moveaxis(tensor_hrs.numpy(), 0, 2)
    
    b, c, orig_h, orig_w = tensor_lrs.shape
    frame_index = math.ceil(b / 2) - 1      # get middle frame index
    
    draw_frame = pil_to_cv2(img_hr[frame_index])            # cv2 [0, 255] BGR
    cv2.rectangle(draw_frame, (w, h), (w + window_size, h + window_size), (0, 0, 255), 2)
    position_pil = cv2_to_pil(draw_frame)
    
    sigma = 1.2
    fold = 50
    l = 9
    alpha = 0.5
    
    attr_objective = attribution_objective(attr_grad, h, w, window=window_size)
    gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)
    
    interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(tensor_lrs.numpy(), model, attr_objective, 
                                                                              gaus_blur_path_func, frame_index, cuda=True)
    
    for index in range(b):
        grad_numpy, result = saliency_map_PG(interpolated_grad_numpy[index], result_numpy[index])
        abs_normed_grad_numpy = grad_abs_norm(grad_numpy)
        saliency_image_abs = vis_saliency(abs_normed_grad_numpy, zoomin=scale)
        saliency_image_kde = vis_saliency_kde(abs_normed_grad_numpy, zoomin=scale)
        blend_abs_and_input = cv2_to_pil(
            pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + pil_to_cv2(img_lr[index].resize(img_hr[index].size)) * alpha)
        blend_kde_and_input = cv2_to_pil(
            pil_to_cv2(saliency_image_kde) * (1.0 - alpha) + pil_to_cv2(img_lr[index].resize(img_hr[index].size)) * alpha)
        pil = make_pil_grid(
            [position_pil,
            saliency_image_abs,
            blend_abs_and_input,
            blend_kde_and_input,
            Tensor2PIL(torch.clamp(torch.from_numpy(result), min=0., max=1.))]
        )
        # pil.show()
        pil.save(os.path.join(work_dir, str(index) + ".png"))


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    # parse options, set distributed setting, set random seed
    opt, _ = parse_options(root_path, is_train=False)
    
    model  = build_model(opt)
    
    # test dataset patch
    video_patch = 'datasets/REDS4/lam_patch/crop_patch_000_1/car_number'
    work_dir = osp.join('work_dir', opt['network_g']['type'], video_patch.split('/')[-1])
    if not osp.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)
    
    # lam patch
    window_size = 16    # Define windoes_size of D
    win_h, win_w = 170, 100
    scale = 4
    
    lam_vsr(model.net_g.to('cpu'), scale, video_patch, work_dir, window_size, win_h, win_w)
    
    
