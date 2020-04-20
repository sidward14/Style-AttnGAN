from __future__ import print_function

from miscc.utils import mkdir_p
from miscc.utils import build_super_images
from miscc.losses import sent_loss, words_loss
from miscc.config import cfg, cfg_from_file

from datasets import TextDataset
from datasets import prepare_data

from model import TRANSFORMER_ENCODER, RNN_ENCODER, CNN_ENCODER

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
# from torch.nn.utils.rnn import pad_packed_sequence

from transformers import GPT2Model


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


UPDATE_INTERVAL = 200

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/DAMSM/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--text_encoder_type', type=str.casefold, default = 'rnn' )
    args = parser.parse_args()
    return args


def train( dataloader, cnn_model, nlp_model, text_encoder_type, batch_size,
           labels, optimizer, epoch, ixtoword, image_dir ):
    cnn_model.train()
    nlp_model.train()
    text_encoder_type = text_encoder_type.casefold()
    assert text_encoder_type in ( 'rnn', 'transformer', )
    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0
    count = (epoch + 1) * len(dataloader)
    start_time = time.time()
    for step, data in enumerate(dataloader, 0):
        # print('step', step)
        nlp_model.zero_grad()
        cnn_model.zero_grad()

        imgs, captions, cap_lens, \
            class_ids, keys = prepare_data( data )


        # words_features: batch_size x nef x 17 x 17
        # sent_code: batch_size x nef
        words_features, sent_code = cnn_model(imgs[-1])
        # print( words_features.shape, sent_code.shape )
        # --> batch_size x nef x 17*17
        nef, att_sze = words_features.size(1), words_features.size(2)
        # words_features = words_features.view(batch_size, nef, -1)

        # Forward Prop:
        # inputs:
        #   captions: torch.LongTensor of ids of size batch x n_steps
        # outputs:
        #   words_emb: batch_size x nef x seq_len
        #   sent_emb: batch_size x nef
        if text_encoder_type == 'rnn':
            hidden = nlp_model.init_hidden( batch_size )
            words_emb, sent_emb = nlp_model( captions, cap_lens, hidden )
        elif text_encoder_type == 'transformer':
            words_emb = nlp_model( captions )[0].transpose(1, 2).contiguous()
            sent_emb = words_emb[ :, :, -1 ].contiguous()
            # sent_emb = sent_emb.view(batch_size, -1)
        # print( words_emb.shape, sent_emb.shape )

        # Compute Loss:
        # NOTE: the ideal loss for Transformer may be different than that for bi-directional LSTM
        w_loss0, w_loss1, attn_maps = words_loss( words_features, words_emb, labels,
                                                  cap_lens, class_ids, batch_size )
        w_total_loss0 += w_loss0.data
        w_total_loss1 += w_loss1.data
        loss = w_loss0 + w_loss1

        s_loss0, s_loss1 = \
            sent_loss( sent_code, sent_emb, labels, class_ids, batch_size )
        loss += s_loss0 + s_loss1
        s_total_loss0 += s_loss0.data
        s_total_loss1 += s_loss1.data
        #
        # Backprop:
        loss.backward()
        #
        # `clip_grad_norm` helps prevent
        # the exploding gradient problem in RNNs / LSTMs.
        if text_encoder_type == 'rnn':
            torch.nn.utils.clip_grad_norm(nlp_model.parameters(),
                                          cfg.TRAIN.RNN_GRAD_CLIP)
        optimizer.step()

        if step % UPDATE_INTERVAL == 0:
            count = epoch * len(dataloader) + step

            # print(  s_total_loss0, s_total_loss1 )
            s_cur_loss0 = s_total_loss0.item() / UPDATE_INTERVAL
            s_cur_loss1 = s_total_loss1.item() / UPDATE_INTERVAL

            # print(  w_total_loss0, w_total_loss1 )
            w_cur_loss0 = w_total_loss0.item() / UPDATE_INTERVAL
            w_cur_loss1 = w_total_loss1.item() / UPDATE_INTERVAL

            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  's_loss {:5.5f} {:5.5f} | '
                  'w_loss {:5.5f} {:5.5f}'
                  .format(epoch, step, len(dataloader),
                          elapsed * 1000. / UPDATE_INTERVAL,
                          s_cur_loss0, s_cur_loss1,
                          w_cur_loss0, w_cur_loss1))
            s_total_loss0 = 0
            s_total_loss1 = 0
            w_total_loss0 = 0
            w_total_loss1 = 0
            start_time = time.time()

            # Attention Maps
            img_set, _ = \
                build_super_images(imgs[-1].cpu(), captions,
                                   ixtoword, attn_maps, att_sze)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/attention_maps%d.png' % (image_dir, step)
                im.save(fullpath)
    return count


def evaluate(dataloader, cnn_model, nlp_model, text_encoder_type, batch_size):
    cnn_model.eval()
    nlp_model.eval()
    text_encoder_type = text_encoder_type.casefold()
    assert text_encoder_type in ( 'rnn', 'transformer', )
    s_total_loss = 0
    w_total_loss = 0
    for step, data in enumerate(dataloader, 0):
        real_imgs, captions, cap_lens, \
                class_ids, keys = prepare_data( data )

        words_features, sent_code = cnn_model(real_imgs[-1])
        # nef = words_features.size(1)
        # words_features = words_features.view(batch_size, nef, -1)

        if text_encoder_type == 'rnn':
            hidden = nlp_model.init_hidden( batch_size )
            words_emb, sent_emb = nlp_model( captions, cap_lens, hidden )
        elif text_encoder_type == 'transformer':
            words_emb = nlp_model( captions )[0].transpose(1, 2).contiguous()
            sent_emb = words_emb[ :, :, -1 ].contiguous()
            # sent_emb = sent_emb.view(batch_size, -1)

        w_loss0, w_loss1, attn = words_loss( words_features, words_emb, labels,
                                             cap_lens, class_ids, batch_size )
        w_total_loss += ( w_loss0 + w_loss1 ).data

        s_loss0, s_loss1 = \
            sent_loss( sent_code, sent_emb, labels, class_ids, batch_size )
        s_total_loss += ( s_loss0 + s_loss1 ).data

        if step == 50:
            break

    s_cur_loss = s_total_loss.item() / step
    w_cur_loss = w_total_loss.item() / step

    return s_cur_loss, w_cur_loss


def build_models( text_encoder_type ):
    # build model ############################################################
    text_encoder_type = text_encoder_type.casefold()
    if text_encoder_type not in ( 'rnn', 'transformer' ):
      raise ValueError( 'Unsupported text_encoder_type' )

    if text_encoder_type == 'rnn':
        text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    elif text_encoder_type == 'transformer':
        # don't initialize the weights of these huge models from scratch...
        text_encoder = GPT2Model.from_pretrained( TRANSFORMER_ENCODER )
                                                  # output_hidden_states = True )
    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)

    labels = Variable(torch.LongTensor(range(batch_size)))
    start_epoch = 0
    if cfg.TRAIN.NET_E:
        if text_encoder_type == 'rnn':
          state_dict = torch.load(cfg.TRAIN.NET_E)
          text_encoder.load_state_dict(state_dict)
        elif  text_encoder_type == 'transformer':
          text_encoder = GPT2Model.from_pretrained( cfg.TRAIN.NET_E )
                                                    # output_hidden_states = True )
        print('Load ', cfg.TRAIN.NET_E)
          #
        name = cfg.TRAIN.NET_E.replace( 'text_encoder', 'image_encoder' )
        state_dict = torch.load(name)
        image_encoder.load_state_dict(state_dict)
        print('Load ', name)

        istart = cfg.TRAIN.NET_E.rfind('_') + 8
        iend = cfg.TRAIN.NET_E.rfind('.')
        start_epoch = cfg.TRAIN.NET_E[istart:iend]
        start_epoch = int(start_epoch) + 1
    else:
      if text_encoder_type == 'rnn':
        print( 'Training RNN from scratch' )
      elif text_encoder_type == 'transformer':
        print( 'Training Transformer starting from pretrained model' )
      print( 'Training CNN starting from ImageNet pretrained Inception-v3' )

    print('start_epoch', start_epoch)

    if cfg.CUDA:
        text_encoder = text_encoder.cuda()
        image_encoder = image_encoder.cuda()
        labels = labels.cuda()

    return text_encoder, image_encoder, labels, start_epoch


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    mkdir_p(model_dir)
    mkdir_p(image_dir)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(cfg.DATA_DIR, args.text_encoder_type, 'train',
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)

    print(dataset.n_words, dataset.embeddings_num)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    # # validation data #
    print( args.text_encoder_type )
    dataset_val = TextDataset(cfg.DATA_DIR, args.text_encoder_type, 'test',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    # Train ##############################################################
    text_encoder, image_encoder, labels, start_epoch = build_models( args.text_encoder_type )
    para = list(text_encoder.parameters())
    for v in image_encoder.parameters():
        if v.requires_grad:
            para.append(v)
    # optimizer = optim.Adam(para, lr=cfg.TRAIN.ENCODER_LR, betas=(0.5, 0.999))
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        lr = cfg.TRAIN.ENCODER_LR
        for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
            optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))
            epoch_start_time = time.time()
            count = train(dataloader, image_encoder, text_encoder,
                          args.text_encoder_type,
                          batch_size, labels, optimizer, epoch,
                          dataset.ixtoword, image_dir)
            print('-' * 89)
            if len(dataloader_val) > 0:
                s_loss, w_loss = evaluate(dataloader_val, image_encoder,
                                          text_encoder, args.text_encoder_type, batch_size)
                print('| end epoch {:3d} | valid loss '
                      '{:5.5f} {:5.5f} | lr {:.8f}|'
                      .format(epoch, s_loss, w_loss, lr))
            print('-' * 89)
            if lr > cfg.TRAIN.ENCODER_LR/10.:
                lr *= 0.98

            if (epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or
                epoch == cfg.TRAIN.MAX_EPOCH):
                torch.save(image_encoder.state_dict(),
                           '%s/image_encoder%d.pth' % (model_dir, epoch))
                torch.save(text_encoder.state_dict(),
                           '%s/text_encoder%d.pth' % (model_dir, epoch))
                print('Save G/Ds models.')
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
