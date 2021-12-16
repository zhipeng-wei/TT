import argparse
import os
import torch
import numpy as np
import math

import attack_methods
from dataset.ucf101 import get_dataset
from gluoncv.torch.model_zoo import get_model

from utils import CONFIG_PATHS, OPT_PATH, get_cfg_custom, MODEL_TO_CKPTS

def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N')
    parser.add_argument('--model', type=str, default='i3d_resnet101', help='i3d_resnet101 | slowfast_resnet101 | tpn_resnet101.')
    parser.add_argument('--attack_method', type=str, default='TemporalTranslation', help='TemporalTranslation | TemporalTranslation_TI')
    parser.add_argument('--step', type=int, default=10, metavar='N',
                    help='Multi-step or One-step.')

    parser.add_argument('--file_prefix', type=str, default='')

    # parameters in the paper
    parser.add_argument('--kernlen', type=int, default=15, metavar='N')
    parser.add_argument('--momentum', action='store_true', default=False, help='Use iterative momentum in MFFGSM.')
    parser.add_argument('--move_type', type=str, default='adj',help='adj | remote | random')
    parser.add_argument('--kernel_mode', type=str, default='gaussian')
    args = parser.parse_args()
    args.adv_path = os.path.join(OPT_PATH, 'UCF-{}-{}-{}-{}'.format(args.model, args.attack_method, args.step, args.file_prefix))
    if not os.path.exists(args.adv_path):
        os.makedirs(args.adv_path)
    return args

if __name__ == '__main__':
    args = arg_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print (args)

    # loading cfg
    cfg_path = CONFIG_PATHS[args.model]
    cfg = get_cfg_custom(cfg_path, args.batch_size)
    cfg.CONFIG.MODEL.PRETRAINED = False

    # loading model.
    ckpt_path = MODEL_TO_CKPTS[args.model]
    model = get_model(cfg)
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    model.cuda()
    model.eval()

    # loading dataset
    dataset_loader = get_dataset('./ucf_all_info.csv', './used_idxs.pkl', args.batch_size)


    # attack
    params = {'kernlen':args.kernlen, 
              'momentum':args.momentum,
              'move_type':args.move_type,
              'kernel_mode':args.kernel_mode}
    attack_method = getattr(attack_methods, args.attack_method)(model, params=params, steps=args.step)

    for step, data in enumerate(dataset_loader):
        if step %1 == 0:
            print ('Running {}, {}/{}'.format(args.attack_method, step+1, len(dataset_loader)))
        val_batch = data[0].cuda()
        val_label = data[1].cuda()
        adv_batches = attack_method(val_batch, val_label)
        val_batch = val_batch.detach()
        for ind,label in enumerate(val_label):
            ori = val_batch[ind].cpu().numpy()
            adv = adv_batches[ind].cpu().numpy()
            np.save(os.path.join(args.adv_path, '{}-adv'.format(label.item())), adv)
            np.save(os.path.join(args.adv_path, '{}-ori'.format(label.item())), ori)
