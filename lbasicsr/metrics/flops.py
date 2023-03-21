# Copyright (c) OpenMMLab. All rights reserved.

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


def get_flops(model, shape):

    # args = parse_args()

    if len(shape) == 1:
        input_shape = (3, shape[0], shape[0])
    elif len(shape) == 2:
        input_shape = (3, ) + tuple(shape)
    elif len(shape) in [3, 4]:  # 4 for video inputs (t, c, h, w)
        input_shape = tuple(shape)
    else:
        raise ValueError('invalid input shape')

    # cfg = Config.fromfile(args.config)

    # init_default_scope(cfg.get('default_scope', 'mmedit'))

    # model = MODELS.build(cfg.model)
    # if torch.cuda.is_available():
    #     model.cuda()
    # model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    elif hasattr(model, 'forward_tensor'):
        model.forward = model.forward_tensor
    # else:
    #     raise NotImplementedError(
    #         'FLOPs counter is currently not currently supported '
    #         f'with {model.__class__.__name__}')

    flops, params = get_model_complexity_info(model, input_shape)

    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    if len(input_shape) == 4:
        print('!!!If your network computes N frames in one forward pass, you '
              'may want to divide the FLOPs by N to get the average FLOPs '
              'for each frame.')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


# if __name__ == '__main__':
#     get_flops()
