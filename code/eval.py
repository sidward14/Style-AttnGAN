from six.moves import range

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from PIL import Image

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2
from datasets import prepare_data
from miscc.utils import weights_init
from model import G_DCGAN, G_NET, G_NET_STYLED
from model import TRANSFORMER_ENCODER, RNN_ENCODER

import numpy as np

import os

from transformers import GPT2Model

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class Evaluator(object):
    def __init__(self, data_loader, n_words, ixtoword, text_encoder_type):
        torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE

        self.n_words = n_words
        self.ixtoword = ixtoword

        self.data_loader = data_loader

        self.text_encoder_type = text_encoder_type.casefold()
        if self.text_encoder_type not in ( 'rnn', 'transformer' ):
          raise ValueError( 'Unsupported text_encoder_type' )

    def build_models_eval(self, init_func = None):
        # #######################generator########################### #
        if cfg.GAN.B_DCGAN:
            netG = G_DCGAN()
        elif cfg.GAN.B_STYLEGEN:
            netG = G_NET_STYLED()
        else:
            netG = G_NET()
            if init_func is not None:
                netG.apply(init_func)
        # print( netG.__class__ )
        model_dir = cfg.TRAIN.NET_G  # the path to save generated images
        try:
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            state_dict = torch.load(model_dir, map_location = lambda storage, loc: storage)
        except:
            msg = f'The path for the models cfg.TRAIN.NET_G = {cfg.TRAIN.NET_G} is not found'
            raise ValueError( msg )
        if cfg.GAN.B_STYLEGEN:
            # netG.load_state_dict( state_dict )
            netG.w_ewma = state_dict[ 'w_ewma' ]
            if cfg.CUDA:
                netG.w_ewma = netG.w_ewma.to( 'cuda:' + str( cfg.GPU_ID ) )
            netG.load_state_dict( state_dict[ 'netG_state_dict' ] )
        else:
            netG.load_state_dict( state_dict )
        print('Load G from: ', model_dir)
        netG.cuda()
        netG.eval()

        # ###################text encoder########################### #
        if self.text_encoder_type == 'rnn':
            text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        elif self.text_encoder_type == 'transformer':
            text_encoder = GPT2Model.from_pretrained( TRANSFORMER_ENCODER )
        state_dict = \
            torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder = text_encoder.cuda()
        text_encoder.eval()

        return text_encoder, netG

    def sampling(self, split_dir):
        if split_dir == 'test':
            split_dir = 'valid'
        model_dir = cfg.TRAIN.NET_G  # the path to save generated images

        # Build and load the generator and text encoder
        text_encoder, netG = self.build_models_eval(init_func = weights_init)

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        with torch.no_grad():
            noise = Variable(torch.FloatTensor(batch_size, nz))
            noise = noise.cuda()

        # the path to save generated images
        s_tmp = model_dir[:model_dir.rfind('.pth')]
        save_dir = '%s/%s' % (s_tmp, split_dir)
        mkdir_p(save_dir)

        cnt = 0

        for _ in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
            for step, data in enumerate(self.data_loader, 0):
                cnt += batch_size
                if step % 100 == 0:
                    print('step: ', step)
                # if step > 50:
                #     break

                imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

                #######################################################
                # (1) Extract text embeddings
                ######################################################
                if self.text_encoder_type == 'rnn':
                    hidden = text_encoder.init_hidden(batch_size)
                    words_embs, sent_emb = text_encoder( captions, cap_lens, hidden )
                elif self.text_encoder_type == 'transformer':
                    words_embs = text_encoder( captions )[0].transpose(1, 2).contiguous()
                    sent_emb = words_embs[ :, :, -1 ].contiguous()
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)
                for j in range(batch_size):
                    s_tmp = '%s/single/%s' % (save_dir, keys[j])
                    folder = s_tmp[:s_tmp.rfind('/')]
                    if not os.path.isdir(folder):
                        print('Make a new folder: ', folder)
                        mkdir_p(folder)
                    k = -1
                    # for k in range(len(fake_imgs)):
                    im = fake_imgs[k][j].data.cpu().numpy()
                    # [-1, 1] --> [0, 255]
                    im = (im + 1.0) * 127.5
                    im = im.astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))
                    im = Image.fromarray(im)
                    fullpath = '%s_s%d.png' % (s_tmp, k)
                    im.save(fullpath)
        return save_dir

    def gen_example(self, data_dic):
        model_dir = cfg.TRAIN.NET_G  # the path to save generated images

        # Build and load the generator and text encoder
        text_encoder, netG = self.build_models_eval()

        # the path to save generated images
        s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
        # print( data_dic.keys() )
        save_dirs = []

        for key in data_dic:
            save_dir = '%s/%s' % (s_tmp, key)
            save_dirs.append( save_dir )
            mkdir_p(save_dir)
            captions, cap_lens, sorted_indices = data_dic[key]

            batch_size = captions.shape[0]
            nz = cfg.GAN.Z_DIM

            with torch.no_grad():
                captions = Variable(torch.from_numpy(captions))
                cap_lens = Variable(torch.from_numpy(cap_lens))

                captions = captions.cuda()
                cap_lens = cap_lens.cuda()

            for i in range(1):  # 16
                with torch.no_grad():
                    noise = Variable(torch.FloatTensor(batch_size, nz))
                    # noise = Variable(torch.FloatTensor(1, nz))
                    noise = noise.cuda()

                #######################################################
                # (1) Extract text embeddings
                ######################################################
                if self.text_encoder_type == 'rnn':
                    hidden = text_encoder.init_hidden(batch_size)
                    words_embs, sent_emb = text_encoder( captions, cap_lens, hidden )
                elif self.text_encoder_type == 'transformer':
                    words_embs = text_encoder( captions )[0].transpose(1, 2).contiguous()
                    sent_emb = words_embs[ :, :, -1 ].contiguous()                    
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                mask = (captions == 0)

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                # noise = noise.repeat( batch_size, 1 )
                fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
                # G attention
                cap_lens_np = cap_lens.cpu().data.numpy()
                for j in range(batch_size):
                    save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                    for k in range(len(fake_imgs)):
                        im = fake_imgs[k][j].data.cpu().numpy()
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        # print('im', im.shape)
                        im = np.transpose(im, (1, 2, 0))
                        # print('im', im.shape)
                        im = Image.fromarray(im)
                        fullpath = '%s_g%d.png' % (save_name, k)
                        im.save(fullpath)

                    for k in range(len(attention_maps)):
                        if len(fake_imgs) > 1:
                            im = fake_imgs[k + 1].detach().cpu()
                        else:
                            im = fake_imgs[0].detach().cpu()
                        attn_maps = attention_maps[k]
                        att_sze = attn_maps.size(2)
                        img_set, sentences = \
                            build_super_images2(im[j].unsqueeze(0),
                                                captions[j].unsqueeze(0),
                                                [cap_lens_np[j]], self.ixtoword,
                                                [attn_maps[j]], att_sze)
                        if img_set is not None:
                            im = Image.fromarray(img_set)
                            fullpath = '%s_a%d.png' % (save_name, k)
                            im.save(fullpath)
        return save_dirs