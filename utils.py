import os
from gluoncv.torch.engine.config import get_cfg_defaults
import torch

# config info
# refer to https://cv.gluon.ai/model_zoo/action_recognition.html
CONFIG_ROOT = './config' # config paths
CONFIG_PATHS = {
    'i3d_resnet50': os.path.join(CONFIG_ROOT, 'i3d_nl5_resnet50_v1_kinetics400.yaml'),
    'i3d_resnet101': os.path.join(CONFIG_ROOT, 'i3d_nl5_resnet101_v1_kinetics400.yaml'),
    'slowfast_resnet50': os.path.join(CONFIG_ROOT, 'slowfast_8x8_resnet50_kinetics400.yaml'),
    'slowfast_resnet101': os.path.join(CONFIG_ROOT, 'slowfast_8x8_resnet101_kinetics400.yaml'),
    'tpn_resnet50': os.path.join(CONFIG_ROOT, 'tpn_resnet50_f32s2_kinetics400.yaml'),
    'tpn_resnet101': os.path.join(CONFIG_ROOT, 'tpn_resnet101_f32s2_kinetics400.yaml')
    }

# save info
OPT_PATH = '' # output path

# ucf model infos
UCF_MODEL_ROOT = '' # ckpt file path of UCF101
MODEL_TO_CKPTS = {
    'i3d_resnet50': os.path.join(UCF_MODEL_ROOT, 'i3d_resnet50.pth'),
    'i3d_resnet101': os.path.join(UCF_MODEL_ROOT, 'i3d_resnet101.pth'),
    'slowfast_resnet50': os.path.join(UCF_MODEL_ROOT, 'slowfast_resnet50.pth'),
    'slowfast_resnet101': os.path.join(UCF_MODEL_ROOT, 'slowfast_resnet101.pth'),
    'tpn_resnet50': os.path.join(UCF_MODEL_ROOT, 'tpn_resnet50.pth'),
    'tpn_resnet101': os.path.join(UCF_MODEL_ROOT, 'tpn_resnet101.pth')
}
# ucf dataset
UCF_DATA_ROOT = '' # ucf101 dataset path
Kinetic_DATA_ROOT = '' # kinetics dataset path

def change_cfg(cfg, batch_size):
    # modify video paths and pretrain setting.
    cfg.CONFIG.DATA.VAL_DATA_PATH = Kinetic_DATA_ROOT
    cfg.CONFIG.DATA.VAL_ANNO_PATH = './kinetics400_attack_samples.csv'
    cfg.CONFIG.MODEL.PRETRAINED = True
    cfg.CONFIG.VAL.BATCH_SIZE = batch_size
    return cfg

def get_cfg_custom(cfg_path, batch_size=16):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg = change_cfg(cfg, batch_size)
    return cfg

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count