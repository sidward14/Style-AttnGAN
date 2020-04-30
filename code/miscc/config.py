from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# Dataset name: flowers, birds
__C.DATASET_NAME = 'birds'
__C.CONFIG_NAME = ''
__C.DATA_DIR = ''
__C.GPU_ID = 0
__C.CUDA = True
__C.WORKERS = 6

__C.RNN_TYPE = 'LSTM'   # 'GRU'  # 'TRANSFORMER'  # 'LSTM'
__C.B_VALIDATION = False

__C.TREE = edict()
__C.TREE.BRANCH_NUM = 3
__C.TREE.BASE_SIZE = 64


# Training options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.SNAPSHOT_INTERVAL = 2000
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.ENCODER_LR = 2e-4
__C.TRAIN.RNN_GRAD_CLIP = 0.25
__C.TRAIN.FLAG = True
__C.TRAIN.NET_E = ''
__C.TRAIN.NET_G = ''
__C.TRAIN.B_NET_D = True

__C.TRAIN.SMOOTH = edict()
__C.TRAIN.SMOOTH.GAMMA1 = 5.0
__C.TRAIN.SMOOTH.GAMMA3 = 10.0
__C.TRAIN.SMOOTH.GAMMA2 = 5.0
__C.TRAIN.SMOOTH.LAMBDA = 1.0


# Modal options
__C.GAN = edict()
__C.GAN.DF_DIM = 32
__C.GAN.GF_DIM = 32
__C.GAN.Z_DIM = 100              # start with the StyleGAN/ProGAN default of 512, rather than the AttnGAN default of 100
__C.GAN.W_DIM = 100              # probably just leave it as matched with Z_DIM (StyleGAN only)
__C.GAN.CONDITION_DIM = 100
__C.GAN.EMBED_COND_VARS = False  # whether you want to pass the c_code through an embedding layer before concatenating to z (StyleGAN only)
__C.GAN.LRMUL = .01              # lrmul for mapping network (StyleGAN only)
__C.GAN.F_NUM = 8                # Number of fcs in the mapping network (StyleGAN only)
__C.GAN.R_NUM = 2                # (Original GAN Generator only)
__C.GAN.PCT_MIXING_REG = .9      # percent of batches to use mixing regularization on (StyleGAN only)
__C.GAN.BETA = .995              # EWMA parameter for calculating average w for the truncation trick (StyleGAN only)
__C.GAN.PSI = .7                 # multiplicative factor by which to truncate the average w for the truncation trick (StyleGAN only)
__C.GAN.CUTOFF_STAGE = 4         # cutoff stage to stop using the truncation trick (StyleGAN only)
__C.GAN.MBSTD_GROUP_SIZE = 4     # group size for minibatch standard deviation to improve variation (StyleGAN only)
__C.GAN.BLUR_TYPE = 'binomial'   # type of low-pass filter to use for better upssampling/downsampling (StyleGAN only)
# TODO: __C.GAN.NUM_CLASSES = 0
__C.GAN.B_ATTENTION = True
__C.GAN.B_DCGAN = False
__C.GAN.B_STYLEGEN = True        # whether to use StyleGAN Generator (StyleGAN only)
__C.GAN.B_STYLEDISC = True       # whether to use StyleGAN Discriminator (StyleGAN only)


__C.TEXT = edict()
__C.TEXT.CAPTIONS_PER_IMAGE = 10
__C.TEXT.EMBEDDING_DIM = 256  # 768 # 256 for the default RNN_ENCODER bi-directional LSTM
__C.TEXT.WORDS_NUM = 18


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
