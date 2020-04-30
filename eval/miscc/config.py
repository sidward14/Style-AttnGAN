from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# Dataset name: flowers, birds
__C.DATASET_NAME = 'birds'
__C.CONFIG_NAME = 'attn2'
__C.DATA_DIR = ''
__C.GPU_ID = 0
__C.CUDA = False
__C.WORKERS = 1

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
__C.TRAIN.FLAG = False
__C.TRAIN.NET_E = 'data/text_encoder200.pth'
__C.TRAIN.NET_G = 'data/bird_AttnGAN2.pth'
__C.TRAIN.B_NET_D = False

__C.TRAIN.SMOOTH = edict()
__C.TRAIN.SMOOTH.GAMMA1 = 5.0
__C.TRAIN.SMOOTH.GAMMA3 = 10.0
__C.TRAIN.SMOOTH.GAMMA2 = 5.0
__C.TRAIN.SMOOTH.LAMBDA = 1.0


# Modal options
__C.GAN = edict()
__C.GAN.DF_DIM = 32              # default is 64
__C.GAN.GF_DIM = 32              # start with 128 for StyleGAN, default is 32
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
__C.TEXT.WORDS_NUM = 25
