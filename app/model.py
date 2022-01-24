import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np

from miscc.config import cfg
from GlobalAttention import GlobalAttentionGeneral as ATT_NET

from gan_lab.stylegan.architectures import StyleMappingNetwork, StyleConditionedMappingNetwork, \
                                           StyleAddNoise
from gan_lab.utils.latent_utils import gen_rand_latent_vars
from gan_lab.utils.custom_layers import LinearEx, Conv2dEx, Conv2dBias, Lambda, \
                                        NormalizeLayer, get_blur_op, concat_mbstd_layer

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

# ############## Text2Image Encoder-Decoder #######
TRANSFORMER_ENCODER = 'gpt2'

class RNN_ENCODER(nn.Module):
    def __init__(self, ntoken, ninput=300, drop_prob=0.5,
                 nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()

        self.n_steps = cfg.TEXT.WORDS_NUM
        self.rnn_type = cfg.RNN_TYPE

        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # Do not need to initialize RNN parameters, which have been initialized
        # http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.nlayers * self.num_directions,
                                       bsz, self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden, mask=None):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))
        #
        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb


# ############## G networks ###################
class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()

        self.t_dim = cfg.TEXT.EMBEDDING_DIM
        self.c_dim = cfg.GAN.CONDITION_DIM

        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar



class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.functional.interpolate(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block


# Keep the spatial size
def Block3x3_relu( in_planes, out_planes, ex = False, norm_type = 'batchnorm' ):
    if not ex:
        conv = conv3x3(in_planes, out_planes * 2)
    else:
        conv = Conv2dEx( ni = in_planes, nf = out_planes * 2, ks = 3, stride = 1, padding = 1,
                         init = 'He', init_type = 'StyleGAN', gain_sq_base = 2.,
                         equalized_lr = True, include_bias = False )
    if norm_type == 'batchnorm':
        norm = nn.BatchNorm2d(out_planes * 2)
    elif norm_type == 'instancenorm':
        norm = NormalizeLayer( 'InstanceNorm' )
    block = nn.Sequential(
        conv,
        norm,
        GLU())
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num, ex = False, norm_type = 'batchnorm', use_glu = True, bottle = False, use_bias = False ):
        super(ResBlock, self).__init__()
        b = 2 if bottle else 1
        if use_glu:
            relu = GLU()
            nf = channel_num * 2
        else:
            relu = nn.LeakyReLU( negative_slope = .2 )
            nf = channel_num
        if not ex:
            conv1 = conv3x3(channel_num, b*nf, use_bias = use_bias )
            conv2 = conv3x3(b*channel_num, channel_num, use_bias = use_bias )
        else:
            conv1 = Conv2dEx( ni = channel_num, nf = b*nf, ks = 3, stride = 1, padding = 1,
                              init = 'He', init_type = 'StyleGAN', gain_sq_base = 2.,
                              equalized_lr = True, include_bias = use_bias )
            conv2 = Conv2dEx( ni = b*channel_num, nf = channel_num, ks = 3, stride = 1, padding = 1,
                              init = 'He', init_type = 'StyleGAN', gain_sq_base = 2.,
                              equalized_lr = True, include_bias = use_bias )
        norm1 = []
        norm2 = []
        if norm_type == 'batchnorm':
            norm1 = [ nn.BatchNorm2d( b*nf ) ]
            norm2 = [ nn.BatchNorm2d(channel_num ) ]
        elif norm_type == 'instancenorm':
            norm1 = [ NormalizeLayer( 'InstanceNorm' ) ]
            norm2 = [ NormalizeLayer( 'InstanceNorm' ) ]
        self.block = nn.Sequential(
            conv1,
            *norm1,
            relu,
            conv2,
            *norm2
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out



class CNN_ENCODER(nn.Module):
    def __init__(self, nef):
        super(CNN_ENCODER, self).__init__()
        if cfg.TRAIN.FLAG:
            self.nef = nef
        else:
            self.nef = 256  # define a uniform ranker

        model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)
        # print(model)

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.functional.interpolate(x,size=(299, 299), mode='bilinear')
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global image features
        cnn_code = self.emb_cnn_code(x)
        # 512
        if features is not None:
            features = self.emb_features(features)

        return features, cnn_code


class INIT_STAGE_G_STYLED( nn.Module ):
    def __init__( self, ngf, ncf ):
        super( INIT_STAGE_G_STYLED, self ).__init__()
        self.gf_dim = ngf  # ngf = LEN_LATENT*FMAP_G_INIT_FCTR
        # self.in_dim = cfg.GAN.Z_DIM + ncf

        self._use_noise = True
        self._use_mixing_reg = True if cfg.GAN.PCT_MIXING_REG else False

        self.define_module()

    def define_module( self ):
        ngf = self.gf_dim
        # nz, ngf = self.in_dim, self.gf_dim

        self.gen_layers = nn.ModuleList( )

        # self.fc = nn.Sequential(
        #     nn.Linear(nz, ngf * 4 * 4 * 2, bias=False),
        #     nn.BatchNorm1d(ngf * 4 * 4 * 2),  # maybe use PixelNorm imstead
        #     GLU()
        # )

        self.upsampler = nn.Upsample( scale_factor = 2, mode = 'nearest' )

        # initializing the input to 1 has about the same effect as applying PixelNorm to the input:
        self.init_layer = nn.Parameter(
            torch.FloatTensor( 1, ngf, 4, 4 ).fill_( 1 )
        )
        bias_nl_norm = nn.Sequential( Conv2dBias( nf = ngf ),
                                        nn.LeakyReLU( negative_slope = .2 ),
                                        NormalizeLayer( 'InstanceNorm', ni = ngf ) )

        conv = Conv2dEx( ni = ngf, nf = ngf, ks = 3,
                         stride = 1, padding = 1, init = 'He', init_type = 'StyleGAN',
                         gain_sq_base = 2., equalized_lr = True, include_bias = False )
        self.nl = nn.LeakyReLU( negative_slope = .2 )
        self.norm = NormalizeLayer( 'InstanceNorm' )
        w_to_styles = (
            LinearEx( nin_feat = cfg.GAN.W_DIM, nout_feat = 2 * ngf,
                      init = 'He', init_type = 'StyleGAN', gain_sq_base = 1., equalized_lr = True ),
            LinearEx( nin_feat = cfg.GAN.W_DIM, nout_feat = 2 * ngf,
                      init = 'He', init_type = 'StyleGAN', gain_sq_base = 1., equalized_lr = True ),
        )

        self.gen_layers.append(
            nn.ModuleList( [
                None,
                StyleAddNoise( nf = ngf ),
                bias_nl_norm,
                w_to_styles[0]
            ] )
        )
        self.gen_layers.append(
            nn.ModuleList( [
                conv,
                StyleAddNoise( nf = ngf ),
                nn.Sequential( Conv2dBias( nf = ngf ), self.nl, self.norm ),
                w_to_styles[1]
            ] )
        )

        # resolutions 8 through 64:
        self._increase_scale( ngf, ngf )
        self._increase_scale( ngf, ngf )
        self._increase_scale( ngf, ngf )
        self._increase_scale( ngf, ngf // 2 )

    def _increase_scale( self, in_planes, out_planes ):
        blur_op = get_blur_op( blur_type = cfg.GAN.BLUR_TYPE, num_channels = out_planes ) if \
                  cfg.GAN.BLUR_TYPE is not None else None

        self.gen_layers.append(
            self._get_conv_layer( ni = in_planes, nf = out_planes, upsample = True, blur_op = blur_op )
        )
        self.gen_layers.append(
            self._get_conv_layer( ni = out_planes, nf = out_planes )
        )

    def _get_conv_layer( self, ni, nf, upsample = False, blur_op = None ):
        upsampler = []
        if upsample:
            upsampler.append( self.upsampler )
        conv = Conv2dEx( ni = ni, nf = nf, ks = 3, stride = 1, padding = 1,
                         init = 'He', init_type = 'StyleGAN', gain_sq_base = 2.,
                         equalized_lr = True, include_bias = False )
        blur = []
        if blur_op is not None:
            assert isinstance( blur_op, nn.Module )
            blur.append( blur_op )
        w_to_style = LinearEx( nin_feat = cfg.GAN.W_DIM, nout_feat = 2 * nf,
                               init = 'He', init_type = 'StyleGAN', gain_sq_base = 1.,
                               equalized_lr = True )

        return nn.ModuleList( [ nn.Sequential( *upsampler, conv, *blur ),
                                StyleAddNoise( nf = nf ),
                                nn.Sequential( Conv2dBias( nf = nf ), self.nl, self.norm ),
                                w_to_style ] )

    def train( self, mode = True ):
        """Overwritten to turn on mixing regularization (if > 0%) during training mode."""
        super( INIT_STAGE_G_STYLED, self ).train( mode = mode )

        self._use_noise = True
        self._use_mixing_reg = True if cfg.GAN.PCT_MIXING_REG else False

    def eval( self ):
        """Overwritten to turn off mixing regularization during evaluation mode."""
        super( INIT_STAGE_G_STYLED, self ).eval( )

        self._use_mixing_reg = False

    @property
    def use_noise( self ):
        return self._use_noise

    @use_noise.setter
    def use_noise( self, mode ):
        """Allows for optionally evaluating without noise inputs."""
        if self.training:
            raise Exception( 'Once use_noise argument is set, it cannot be changed' + \
                             ' for training purposes. It can, however, be changed in eval mode.' )
        else:
            self._use_noise = mode

    def forward( self, x, x2 = None, cutoff_idx = None, style_mixing_stage = None, noise = None,
                 use_truncation_trick = False, trunc_cutoff_stage = None, w_ewma = None, w_eval_psi = 1. ):
        """
            input: x (disentangled latent variable)
            :param z_code: batch x cfg.GAN.Z_DIM
            :param c_code: batch x cfg.TEXT.EMBEDDING_DIM
            :return: batch x ngf/16 x 64 x 64
        """
        # x = self.z_to_w( x )
        bs = x.shape[0]

        out = self.init_layer.expand( bs, -1, -1, -1 )

        for n, layer in enumerate( self.gen_layers ):
            # Training Mode Only:
            if n == cutoff_idx and self.training:
                x = x2.detach().clone()
                x.requires_grad_( True )

            if n:
                out = layer[ 0 ]( out )
            if self.use_noise:
                out = layer[ 1 ]( out, noise = noise[ n ] if noise is not None else None )
            out = layer[ 2 ]( out )

            # Evaluation Mode Only:
            if n == style_mixing_stage:
                assert ( style_mixing_stage and not self.training and isinstance( x2, torch.Tensor ) )
                x = x2.detach().clone()
                # the new z that is sampled for style-mixing is already de-truncated
                if use_truncation_trick and trunc_cutoff_stage is not None and n < 2*trunc_cutoff_stage:
                    x = w_ewma.expand_as( x ) + w_eval_psi * ( x - w_ewma.expand_as( x ) )
            elif use_truncation_trick and not self.training and trunc_cutoff_stage is not None and n == 2*trunc_cutoff_stage:
                # de-truncate w for higher resolutions; more memory-efficient than defining 2 w's
                x = ( x - w_ewma.expand_as( x ) ).div( w_eval_psi ) + w_ewma.expand_as( x )

            y = layer[ 3 ]( x ).view( -1, 2, layer[ 3 ].nout_feat // 2, 1, 1 )
            out = out * ( y[ :, 0 ].contiguous().add( 1 ) ) + \
                y[ :, 1 ].contiguous()  # add 1 for skip-connection effect

        return out

class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, ncf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = cfg.GAN.Z_DIM + ncf

        self.define_module()

    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 4 * 4 * 2, bias=False),
            # removing for single instance caption
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU()
        )

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z_code, c_code):
        """
        :param z_code: batch x cfg.GAN.Z_DIM
        :param c_code: batch x cfg.TEXT.EMBEDDING_DIM
        :return: batch x ngf/16 x 64 x 64
        """
        c_z_code = torch.cat((c_code, z_code), 1)
        # state size ngf x 4 x 4
        out_code = self.fc(c_z_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size ngf/3 x 8 x 8
        out_code = self.upsample1(out_code)
        # state size ngf/4 x 16 x 16
        out_code = self.upsample2(out_code)
        # state size ngf/8 x 32 x 32
        out_code32 = self.upsample3(out_code)
        # state size ngf/16 x 64 x 64
        out_code64 = self.upsample4(out_code32)

        return out_code64


class NEXT_STAGE_G_STYLED( nn.Module ):
    def __init__( self, ngf, nef, ncf, res ):
        super( NEXT_STAGE_G_STYLED, self ).__init__()
        self.res = res
        self.gf_dim = ngf
        self.ef_dim = nef
        self.cf_dim = ncf
        self.num_residual = cfg.GAN.R_NUM

        self._use_noise = True
        self._use_mixing_reg = True if cfg.GAN.PCT_MIXING_REG else False

        self.define_module()

    # def _make_layer(self, block, channel_num):
    #     layers = []
    #     for i in range(cfg.GAN.R_NUM):
    #         layers.append(block(channel_num))
    #     return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        # self.block3x3 = Block3x3_relu( ngf, ngf, ex = True, norm_type = 'batchnorm' )
        self.att = ATT_NET( ngf, self.ef_dim )
        # self.residual = self._make_layer(ResBlock, ngf * 2)
        self.residual = ResBlock( ngf * 2, ex = True, norm_type = 'batchnorm' )  # instancenorm
        # self.upsample = upBlock(ngf * 2, ngf)

        self.gen_layers = nn.ModuleList( )

        self.upsampler = nn.Upsample( scale_factor = 2, mode = 'nearest' )

        self.nl = nn.LeakyReLU( negative_slope = .2 )
        self.norm = NormalizeLayer( 'InstanceNorm' )

        self.init_stage = int( np.log2( self.res ) ) - 2

        # going from resolution 64 to 128 or from 128 to 256:
        self._increase_scale( ngf * 2 , ngf )

    def _increase_scale( self, in_planes, out_planes ):
        blur_op = get_blur_op( blur_type = cfg.GAN.BLUR_TYPE, num_channels = out_planes ) if \
                  cfg.GAN.BLUR_TYPE is not None else None

        self.gen_layers.append(
            self._get_conv_layer( ni = in_planes, nf = out_planes, upsample = True, blur_op = blur_op )
        )
        self.gen_layers.append(
            self._get_conv_layer( ni = out_planes, nf = out_planes )
        )

    def _get_conv_layer( self, ni, nf, upsample = False, blur_op = None ):
        upsampler = []
        if upsample:
            upsampler.append( self.upsampler )
        conv = Conv2dEx( ni = ni, nf = nf, ks = 3, stride = 1, padding = 1,
                         init = 'He', init_type = 'StyleGAN', gain_sq_base = 2.,
                         equalized_lr = True, include_bias = False )
        blur = []
        if blur_op is not None:
            assert isinstance( blur_op, nn.Module )
            blur.append( blur_op )
        w_to_style = LinearEx( nin_feat = cfg.GAN.W_DIM, nout_feat = 2 * nf,
                               init = 'He', init_type = 'StyleGAN', gain_sq_base = 1.,
                               equalized_lr = True )

        return nn.ModuleList( [ nn.Sequential( *upsampler, conv, *blur ),
                                StyleAddNoise( nf = nf ),
                                nn.Sequential( Conv2dBias( nf = nf ), self.nl, self.norm ),
                                w_to_style ] )

    def train( self, mode = True ):
        """Overwritten to turn on mixing regularization (if > 0%) during training mode."""
        super( NEXT_STAGE_G_STYLED, self ).train( mode = mode )

        self._use_noise = True
        self._use_mixing_reg = True if cfg.GAN.PCT_MIXING_REG else False

    def eval( self ):
        """Overwritten to turn off mixing regularization during evaluation mode."""
        super( NEXT_STAGE_G_STYLED, self ).eval( )

        self._use_mixing_reg = False

    @property
    def use_noise( self ):
        return self._use_noise

    @use_noise.setter
    def use_noise( self, mode ):
        """Allows for optionally evaluating without noise inputs."""
        if self.training:
            raise Exception( 'Once use_noise argument is set, it cannot be changed' + \
                             ' for training purposes. It can, however, be changed in eval mode.' )
        else:
            self._use_noise = mode

    def forward( self, out, x, word_embs, mask, x2 = None, cutoff_idx = None, style_mixing_stage = None, noise = None,
                 use_truncation_trick = False, trunc_cutoff_stage = None, w_ewma = None, w_eval_psi = 1. ):
        """
            input: x (disentangled latent variable)
            h_code1(query):  batch x idf x ih x iw (queryL=ihxiw)
            word_embs(context): batch x cdf x sourceL (sourceL=seq_len)
            c_code1: batch x idf x queryL
            att1: batch x sourceL x queryL
        """
        # x = self.z_to_w( x )

        # out = self.block3x3( out )

        self.att.applyMask( mask )
        c_code, att = self.att( out, word_embs )
        out = torch.cat(( out, c_code ), 1 )
        out = self.residual( out )

        for n, layer in enumerate( self.gen_layers ):
            out = layer[ 0 ]( out )
            if self.use_noise:
                out = layer[ 1 ]( out, noise = noise[ n ] if noise is not None else None )
            out = layer[ 2 ]( out )

            # Training Mode Only:
            if ( n + 2*self.init_stage ) == cutoff_idx and self.training:
                x = x2.detach().clone()
                x.requires_grad_( True )

            # Evaluation Mode Only:
            if ( n + 2*self.init_stage ) == style_mixing_stage:
                assert ( style_mixing_stage and not self.training and isinstance( x2, torch.Tensor ) )
                x = x2.detach().clone()
            #     the new z that is sampled for style-mixing is already de-truncated
                if use_truncation_trick and trunc_cutoff_stage is not None and ( n + 2*self.init_stage ) < 2*trunc_cutoff_stage:
                    x = w_ewma.expand_as( x ) + w_eval_psi * ( x - w_ewma.expand_as( x ) )
            elif use_truncation_trick and not self.training and trunc_cutoff_stage is not None and ( n + 2*self.init_stage ) == 2*trunc_cutoff_stage:
                # de-truncate w for higher resolutions; more memory-efficient than defining 2 w's
                x = ( x - w_ewma.expand_as( x ) ).div( w_eval_psi ) + w_ewma.expand_as( x )

            y = layer[ 3 ]( x ).view( -1, 2, layer[ 3 ].nout_feat // 2, 1, 1 )
            out = out * ( y[ :, 0 ].contiguous().add( 1 ) ) + \
                  y[ :, 1 ].contiguous()  # add 1 for skip-connection effect

        return out, att

class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, nef, ncf):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.cf_dim = ncf
        self.num_residual = cfg.GAN.R_NUM
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        self.att = ATT_NET(ngf, self.ef_dim)
        self.residual = self._make_layer(ResBlock, ngf * 2)
        self.upsample = upBlock(ngf * 2, ngf)

    def forward(self, h_code, c_code, word_embs, mask):
        """
            h_code1(query):  batch x idf x ih x iw (queryL=ihxiw)
            word_embs(context): batch x cdf x sourceL (sourceL=seq_len)
            c_code1: batch x idf x queryL
            att1: batch x sourceL x queryL
        """
        self.att.applyMask(mask)
        c_code, att = self.att(h_code, word_embs)
        h_c_code = torch.cat((h_code, c_code), 1)
        out_code = self.residual(h_c_code)

        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)

        return out_code, att

class GET_IMAGE_G_STYLED( nn.Module ):
    def __init__( self, ngf ):
        super( GET_IMAGE_G_STYLED, self ).__init__()
        conv = Conv2dEx( ni = ngf, nf = 3, ks = 1, stride = 1,
                         padding = 0, init = 'He', init_type = 'StyleGAN',
                         gain_sq_base = 1., equalized_lr = True )
        self.torgb = nn.Sequential( conv ) if not cfg.GAN.B_TANH else nn.Sequential( conv, nn.Tanh() )

    def forward( self, h_code ):
        return self.torgb( h_code )

class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET_STYLED( nn.Module ):
    def __init__(self):
        super(G_NET_STYLED, self).__init__()
        ngf = cfg.GAN.GF_DIM * 2      # 256  # 32 default
        nef = cfg.TEXT.EMBEDDING_DIM  # 768
        ncf = cfg.GAN.CONDITION_DIM   # 100

        self.ca_net = CA_NET()

        self.map_net = StyleConditionedMappingNetwork(
            ncf,
            len_latent = cfg.GAN.Z_DIM,
            len_dlatent = cfg.GAN.W_DIM,
            num_fcs = cfg.GAN.F_NUM,
            lrmul = cfg.GAN.LRMUL,
            nl = nn.LeakyReLU( negative_slope = .2 ),
            equalized_lr = True,
            normalize_z = True,
            embed_cond_vars = cfg.GAN.EMBED_COND_VARS
        )

        if cfg.TREE.BRANCH_NUM > 0:
            self.final_stage = int( np.log2( 64 ) ) - 1
            self.h_net1 = INIT_STAGE_G_STYLED( ngf * 2, ncf )
            self.img_net1 = GET_IMAGE_G_STYLED( ngf )
        if cfg.TREE.BRANCH_NUM > 1:
            self.final_stage += 1
            self.h_net2 = NEXT_STAGE_G_STYLED( ngf, nef, ncf, res = 128 )
            self.img_net2 = GET_IMAGE_G_STYLED( ngf )
        if cfg.TREE.BRANCH_NUM > 2:
            self.final_stage += 1
            self.h_net3 = NEXT_STAGE_G_STYLED( ngf, nef, ncf, res = 256 )
            self.img_net3 = GET_IMAGE_G_STYLED( ngf )

        # truncation trick  # TODO: Move this to the G_NET_STYLED class
        assert 0. <= cfg.GAN.BETA <= 1.
        self.w_ewma_beta = cfg.GAN.BETA
        self._w_eval_psi = cfg.GAN.PSI  # allow psi to be any number you want, perhaps worthy of experimentation
        # 0 < cfg.GAN.CUTOFF_STAGE <= int( np.log2( 64 ) ) - 2 ) or \
        assert ( ( isinstance( cfg.GAN.CUTOFF_STAGE, int ) and \
                    0 < cfg.GAN.CUTOFF_STAGE <= self.final_stage - 1 ) or \
                    cfg.GAN.CUTOFF_STAGE is None )
        self._trunc_cutoff_stage = cfg.GAN.CUTOFF_STAGE
        # set the below to `False` if you want to turn off during evaluation mode
        self.use_truncation_trick = True if self._trunc_cutoff_stage else False
        self.w_ewma = None

    def to( self, *args, **kwargs ):
        """Overwritten to allow for non-Parameter objects' Tensors to be sent to the appropriate device."""
        super( G_NET_STYLED, self ).to( *args, **kwargs )

        for arg in args:
            if arg in ( 'cpu', 'cuda', ) or isinstance( arg, torch.device ):
                if self.w_ewma is not None:
                    self.w_ewma = self.w_ewma.to( arg )
                    break

    @property
    def w_eval_psi( self ):
        return self._w_eval_psi

    @w_eval_psi.setter
    def w_eval_psi( self, new_w_eval_psi ):
        """Change this to your choosing (but only in evaluation mode), optionally allowing for |psi| to be > 1."""
        if not self.training:
            self._w_eval_psi = new_w_eval_psi
        else:
            raise Exception( 'Can only alter psi value for truncation trick on w during evaluation mode.' )

    @property
    def trunc_cutoff_stage( self ):
        return self._trunc_cutoff_stage

    @trunc_cutoff_stage.setter
    def trunc_cutoff_stage( self, new_trunc_cutoff_stage ):
        """Change this to your choosing (but only in evaluation mode)."""
        if not self.training:
            if ( isinstance( new_trunc_cutoff_stage, int ) and \
                0 < new_trunc_cutoff_stage <= self.final_stage ) or new_trunc_cutoff_stage is None:
                self._trunc_cutoff_stage = new_trunc_cutoff_stage
            else:
                message = f'Input cutoff stage for truncation trick on w must be of type `int` in range (0,{self.final_stage}] or `None`.'
                raise ValueError( message )
        else:
            raise Exception( 'Can only alter cutoff stage for truncation trick on w during evaluation mode.' )

    def forward( self, z_code, sent_emb, word_embs, mask, z2_code = None, cutoff_idx = None, style_mixing_stage = None, is_dlatent = False ):
        """
            :param z_code: batch x cfg.GAN.Z_DIM
            :param sent_emb: batch x cfg.TEXT.EMBEDDING_DIM
            :param word_embs: batch x cdf x seq_len
            :param mask: batch x seq_len
            :return:
        """
        fake_imgs = []
        att_maps = []
        c_code, mu, logvar = self.ca_net( sent_emb )

        w2_code = None
        if z2_code is not None:
            if style_mixing_stage is None or style_mixing_stage >= 2*self.final_stage:
                raise ValueError( 'Please specify a valid style_mixing_stage if specifying a 2nd latent/dlatent variable' )
        else:
            style_mixing_stage = None

        if not is_dlatent:
            w_code = self.map_net( z_code, c_code )
            if z2_code is not None:
                w2_code = self.map_net( z2_code, c_code )
        else:
            w_code = z_code.clone()
            if z2_code is not None:
                w2_code = z2_code.clone()

        if z2_code is None:
            if cfg.GAN.PCT_MIXING_REG:
                if np.random.rand() < cfg.GAN.PCT_MIXING_REG:
                    if z2_code is None:
                        z2_code = gen_rand_latent_vars( num_samples = z_code.shape[0], length = z_code.shape[1],
                                                        distribution = 'normal', device = z_code.device )
                    w2_code = self.map_net( z2_code, c_code )
                    cutoff_idx = torch.randint( 1, 2*self.final_stage, ( 1, ) ).item()

        if self.use_truncation_trick:
            # Training Mode Only:
            if self.training:
                if self.w_ewma is None:
                    self.w_ewma = w_code.detach().clone().mean( dim = 0 )
                else:
                    with torch.no_grad():
                        # TODO: Implement a memory-efficient method to compute this for the ewma generator
                        #       (currently just using the same average w for the generator and the ewma generator)
                        self.w_ewma = w_code.mean( dim = 0 ) * ( 1. - self.w_ewma_beta ) + \
                                      self.w_ewma * ( self.w_ewma_beta )
            # Evaluation Mode Only:
            elif self.trunc_cutoff_stage is not None:
                w_code = self.w_ewma.expand_as( w_code ) + self.w_eval_psi * ( w_code - self.w_ewma.expand_as( w_code ) )

        if cfg.TREE.BRANCH_NUM > 0:
            h_code1 = \
                self.h_net1( w_code, x2 = w2_code, cutoff_idx = cutoff_idx, style_mixing_stage = style_mixing_stage,
                             use_truncation_trick = self.use_truncation_trick, trunc_cutoff_stage = self.trunc_cutoff_stage, w_ewma = self.w_ewma, w_eval_psi = self.w_eval_psi )
            fake_img1 = self.img_net1( h_code1 )
            fake_imgs.append( fake_img1 )
        if cfg.TREE.BRANCH_NUM > 1:
            h_code2, att1 = \
                self.h_net2( h_code1, w_code, word_embs, mask, x2 = w2_code, cutoff_idx = cutoff_idx, style_mixing_stage = style_mixing_stage,
                             use_truncation_trick = self.use_truncation_trick, trunc_cutoff_stage = self.trunc_cutoff_stage, w_ewma = self.w_ewma, w_eval_psi = self.w_eval_psi )
            fake_img2 = self.img_net2( h_code2 )
            fake_imgs.append( fake_img2 )
            if att1 is not None:
                att_maps.append( att1 )
        if cfg.TREE.BRANCH_NUM > 2:
            h_code3, att2 = \
                self.h_net3( h_code2, w_code, word_embs, mask, x2 = w2_code, cutoff_idx = cutoff_idx, style_mixing_stage = style_mixing_stage,
                             use_truncation_trick = self.use_truncation_trick, trunc_cutoff_stage = self.trunc_cutoff_stage, w_ewma = self.w_ewma, w_eval_psi = self.w_eval_psi )
            fake_img3 = self.img_net3( h_code3 )
            fake_imgs.append( fake_img3 )
            if att2 is not None:
                att_maps.append( att2 )

        return fake_imgs, att_maps, mu, logvar

class G_NET(nn.Module):
    def __init__(self):
        super(G_NET, self).__init__()

        ngf = cfg.GAN.GF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ncf = cfg.GAN.CONDITION_DIM


        self.ca_net = CA_NET()

        if cfg.TREE.BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G(ngf * 16, ncf)
            self.img_net1 = GET_IMAGE_G(ngf)
        # gf x 64 x 64
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(ngf, nef, ncf)
            self.img_net2 = GET_IMAGE_G(ngf)
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(ngf, nef, ncf)
            self.img_net3 = GET_IMAGE_G(ngf)

    def forward(self, z_code, sent_emb, word_embs, mask):
        """
            :param z_code: batch x cfg.GAN.Z_DIM
            :param sent_emb: batch x cfg.TEXT.EMBEDDING_DIM
            :param word_embs: batch x cdf x seq_len
            :param mask: batch x seq_len
            :return:
        """
        fake_imgs = []
        att_maps = []
        c_code, mu, logvar = self.ca_net(sent_emb)

        if cfg.TREE.BRANCH_NUM > 0:
            h_code1 = self.h_net1(z_code, c_code)
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)
        if cfg.TREE.BRANCH_NUM > 1:
            h_code2, att1 = \
                self.h_net2(h_code1, c_code, word_embs, mask)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
            if att1 is not None:
                att_maps.append(att1)
        if cfg.TREE.BRANCH_NUM > 2:
            h_code3, att2 = \
                self.h_net3(h_code2, c_code, word_embs, mask)
            fake_img3 = self.img_net3(h_code3)
            fake_imgs.append(fake_img3)
            if att2 is not None:
                att_maps.append(att2)

        return fake_imgs, att_maps, mu, logvar


class G_DCGAN(nn.Module):
    def __init__(self):
        super(G_DCGAN, self).__init__()
        ngf = cfg.GAN.GF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ncf = cfg.GAN.CONDITION_DIM
        self.ca_net = CA_NET()

        # 16gf x 64 x 64 --> gf x 64 x 64 --> 3 x 64 x 64
        if cfg.TREE.BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G(ngf * 16, ncf)
        # gf x 64 x 64
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(ngf, nef, ncf)
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(ngf, nef, ncf)
        self.img_net = GET_IMAGE_G(ngf)

    def forward(self, z_code, sent_emb, word_embs, mask):
        """
            :param z_code: batch x cfg.GAN.Z_DIM
            :param sent_emb: batch x cfg.TEXT.EMBEDDING_DIM
            :param word_embs: batch x cdf x seq_len
            :param mask: batch x seq_len
            :return:
        """
        att_maps = []
        c_code, mu, logvar = self.ca_net(sent_emb)
        if cfg.TREE.BRANCH_NUM > 0:
            h_code = self.h_net1(z_code, c_code)
        if cfg.TREE.BRANCH_NUM > 1:
            h_code, att1 = self.h_net2(h_code, c_code, word_embs, mask)
            if att1 is not None:
                att_maps.append(att1)
        if cfg.TREE.BRANCH_NUM > 2:
            h_code, att2 = self.h_net3(h_code, c_code, word_embs, mask)
            if att2 is not None:
                att_maps.append(att2)

        fake_imgs = self.img_net(h_code)
        return [fake_imgs], att_maps, mu, logvar


# ############## D networks ##########################
def Block3x3_leakRelu(in_planes, out_planes, ex = False, norm_type = 'batchnorm' ):
    if not ex:
        conv = conv3x3(in_planes, out_planes)
    else:
        conv = Conv2dEx( ni = in_planes, nf = out_planes, ks = 3, stride = 1, padding = 1,
                         init = 'He', init_type = 'StyleGAN', gain_sq_base = 2.,
                         equalized_lr = True, include_bias = False )
    if norm_type == 'batchnorm':
        norm = nn.BatchNorm2d(out_planes)
    elif norm_type == 'instancenorm':
        norm = NormalizeLayer( 'InstanceNorm' )
    block = nn.Sequential(
        conv,
        norm,
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 16
def encode_image_by_16times(ndf):
    encode_img = nn.Sequential(
        # --> state size. ndf x in_size/2 x in_size/2
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 2ndf x x in_size/4 x in_size/4
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 4ndf x in_size/8 x in_size/8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 8ndf x in_size/16 x in_size/16
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition = False, ex = False, norm_type = 'batchnorm' ):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if self.bcondition:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + nef, ndf * 8, ex = ex, norm_type = norm_type )

        if not ex:
            conv = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4)
        else:
            _ni = ndf * 8
            conv = Conv2dEx( ni = _ni, nf = 1, ks = 4,
                             stride = 1, padding = 0, init = 'He', init_type =  'StyleGAN',
                             gain_sq_base = 2., equalized_lr = True )
        self.outlogits = nn.Sequential(
            conv,
            nn.Sigmoid()
        )

    def forward(self, h_code, c_code=None):
        if self.bcondition and c_code is not None:
            # conditioning output
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((h_code, c_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)
        return output.view(-1)


# For 64 x 64 images
class D_NET_STYLED64( nn.Module ):
    def __init__( self, b_jcu = True ):
        super( D_NET_STYLED64, self ).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS( ndf, nef, bcondition = False, ex = True, norm_type = 'batchnorm' )  # instancenorm
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS( ndf, nef, bcondition = True, ex = True, norm_type = 'batchnorm' )  # instancenorm


        self.disc_blocks = nn.ModuleList( )

        self.pooler = nn.AvgPool2d( kernel_size = 2, stride = 2 )

        self.nl = nn.LeakyReLU( negative_slope = .2 )

        self.mbstd_group_size = cfg.GAN.MBSTD_GROUP_SIZE
        mbstd_layer = self.get_mbstd_layer( )

        _ndff = ndf * 8
        self.disc_blocks.insert( 0,
            nn.Sequential(
                *mbstd_layer,
                Conv2dEx( ni = _ndff + ( 1 if mbstd_layer else 0 ), nf = _ndff, ks = 3, stride = 1,
                         padding = 1, init = 'He', init_type =  'StyleGAN', gain_sq_base = 2.,
                         equalized_lr = True ),  # this can be done with a linear layer as well (efficiency)
                self.nl,
                # Conv2dEx( ni = ndf, nf = _fmap_end, ks = 4,
                #         stride = 1, padding = 0, init = 'He', init_type =  'StyleGAN',
                #         gain_sq_base = 2., equalized_lr = True ),
                # self.nl,
                # Lambda( lambda x: x.view( -1, _fmap_end ) ),
                # LinearEx( nin_feat = _fmap_end, nout_feat = 1, init = 'He',
                #          init_type =  'StyleGAN', gain_sq_base = 1., equalized_lr = True )
            )
        )

        # resolutions 8 through 64:
        self._increase_scale( ndf * 8, ndf * 8 )
        self._increase_scale( ndf * 8, ndf * 8 )
        self._increase_scale( ndf * 8, ndf * 8 )
        self._increase_scale( ndf * 8, ndf * 4 )

        self.preprocess_x = Lambda( lambda x: x.view( -1, 3, 64, 64 ) )
        self._update_fromrgb( nf = ndf * 4 )

    def _increase_scale( self, in_planes, out_planes ):
        blur_op = get_blur_op( blur_type = cfg.GAN.BLUR_TYPE, num_channels = out_planes ) if \
                  cfg.GAN.BLUR_TYPE is not None else None

        self.disc_blocks.insert( 0,
            nn.Sequential(
                self._get_conv_layer( ni = out_planes, nf = out_planes ),
                self._get_conv_layer( ni = out_planes, nf = in_planes, downsample = True, blur_op = blur_op )
            )
        )

    def _get_conv_layer( self, ni, nf, downsample = False, blur_op = None ):
        blur = []
        if blur_op is not None:
            assert isinstance( blur_op, nn.Module )
            blur.append( blur_op )

        if downsample:
            conv = Conv2dEx( ni = ni, nf = nf, ks = 3, stride = 1, padding = 1,
                             init = 'He', init_type = 'StyleGAN', gain_sq_base = 2.,
                             equalized_lr = True, include_bias = False )
            pooler = [ self.pooler ]
            bias = [ Conv2dBias( nf = nf ) ]
        else:
            conv = Conv2dEx( ni = ni, nf = nf, ks = 3, stride = 1, padding = 1,
                             init = 'He', init_type = 'StyleGAN', gain_sq_base = 2.,
                             equalized_lr = True, include_bias = True )
            pooler = []
            bias = []

        return nn.Sequential( *blur, conv, *( pooler + bias ), self.nl )

    def _update_fromrgb( self, nf ):
        self.fromrgb = nn.Sequential(
            Conv2dEx( ni = 3, nf = nf, ks = 1, stride = 1,
                      padding = 0, init = 'He', init_type = 'StyleGAN',
                      gain_sq_base = 2., equalized_lr = True ),
            self.nl
        )

    # def get_fmap( self, scale_stage ):
    #   return min( int( FMAP_BASE / ( 2**scale_stage ) ), FMAP_MAX )

    def get_mbstd_layer( self ):
        if self.mbstd_group_size == -1:
            mbstd_layer = []
        else:
            mbstd_layer = [ Lambda(
                lambda x, group_size: concat_mbstd_layer( x, group_size ),
                group_size = self.mbstd_group_size
            ) ]

        return mbstd_layer

    # NOTE: Perhaps remove for loop and ideally just make it all one function (i.e. `return self.func( x )`)
    def forward( self, x ):
        x = self.fromrgb( self.preprocess_x( x ) )
        for disc_block in self.disc_blocks:
            x = disc_block( x )
        return x
            
class D_NET64(nn.Module):
    def __init__(self, b_jcu=True):
        super(D_NET64, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        self.img_code_s16 = encode_image_by_16times(ndf)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, x_var):
        print( x_var.shape )
        x_code4 = self.img_code_s16(x_var)  # 4 x 4 x 8df
        return x_code4


# For 128 x 128 images
class D_NET_STYLED128( D_NET_STYLED64 ):
    def __init__( self, b_jcu = True ):
        super( D_NET_STYLED128, self ).__init__( b_jcu = b_jcu )

        ndf = cfg.GAN.DF_DIM

        # blur_op = get_blur_op( blur_type = cfg.GAN.BLUR_TYPE, num_channels = ndf * 4 ) if \
        #           cfg.GAN.BLUR_TYPE is not None else None

        # self.disc_blocks.insert( 0, self._get_conv_layer( ni = ndf * 4, nf = ndf * 4, downsample = True, blur_op = blur_op ) )
        # self.disc_blocks.insert( 0, self.nl )
        # self.disc_blocks.insert( 0, ResBlock( ndf * 4, ex = True, norm_type = None, use_glu = False, use_bias = True ) )  # batchnorm  # instancenorm
        # self.disc_blocks.insert( 0, ResBlock( ndf * 4, ex = True, norm_type = 'batchnorm', use_glu = False ) )  # batchnorm  # instancenorm
        # self.disc_blocks.insert( 0, self._get_conv_layer( ni = ndf * 4, nf = ndf * 4 ) )

        # going from resolution 64 to 128:
        self._increase_scale( ndf * 4, ndf * 4 )  # self._increase_scale( ndf * 4, ndf * 2 )
        # self.disc_blocks.insert( 0, self._get_conv_layer( ni = ndf * 4, nf = ndf * 4 ) )

        self.preprocess_x = Lambda( lambda x: x.view( -1, 3, 128, 128 ) )
        self._update_fromrgb( nf = ndf * 4 )  # self._update_fromrgb( nf = ndf * 2 )

class D_NET128(nn.Module):
    def __init__(self, b_jcu=True):
        super(D_NET128, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)
        #
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, x_var):
        x_code8 = self.img_code_s16(x_var)   # 8 x 8 x 8df
        x_code4 = self.img_code_s32(x_code8)   # 4 x 4 x 16df
        x_code4 = self.img_code_s32_1(x_code4)  # 4 x 4 x 8df
        return x_code4


# For 256 x 256 images
class D_NET_STYLED256( D_NET_STYLED128 ):
    def __init__( self, b_jcu = True ):
        super( D_NET_STYLED256, self ).__init__( b_jcu = b_jcu )

        ndf = cfg.GAN.DF_DIM

        # blur_op = get_blur_op( blur_type = cfg.GAN.BLUR_TYPE, num_channels = ndf * 4 ) if \
        #           cfg.GAN.BLUR_TYPE is not None else None

        # self.disc_blocks.insert( 0, self._get_conv_layer( ni = ndf * 4, nf = ndf * 4, downsample = True, blur_op = blur_op ) )
        # self.disc_blocks.insert( 0, self.nl )
        # self.disc_blocks.insert( 0, ResBlock( ndf * 4, ex = True, norm_type = None, use_glu = False ) )  # batchnorm  # instancenorm
        # self.disc_blocks.insert( 0, ResBlock( ndf * 4, ex = True, norm_type = 'batchnorm', use_glu = False ) )  # batchnorm  # instancenorm
        # self.disc_blocks.insert( 0, self._get_conv_layer( ni = ndf * 4, nf = ndf * 4 ) )

        # going from resolution 128 to 256:
        self._increase_scale( ndf * 4, ndf * 4 )  # self._increase_scale( ndf * 2, ndf * 1 )

        self.preprocess_x = Lambda( lambda x: x.view( -1, 3, 256, 256 ) )
        self._update_fromrgb( nf = ndf * 4 )  # self._update_fromrgb( nf = ndf * 1 )

class D_NET256(nn.Module):
    def __init__(self, b_jcu=True):
        super(D_NET256, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s64_1 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s64_2 = Block3x3_leakRelu(ndf * 16, ndf * 8)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, x_var):
        x_code16 = self.img_code_s16(x_var)
        x_code8 = self.img_code_s32(x_code16)
        x_code4 = self.img_code_s64(x_code8)
        x_code4 = self.img_code_s64_1(x_code4)
        x_code4 = self.img_code_s64_2(x_code4)
        return x_code4
