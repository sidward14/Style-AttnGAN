from __future__ import print_function
from six.moves import range

from tqdm import tqdm
import numpy as np

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from miscc.config import cfg
from miscc.utils import weights_init
from datasets import prepare_data

import lpips

#----------------------------------------------------------------------------

# Normalize batch of vectors.
def normalize(v):
    return v / torch.sqrt( torch.sum(torch.square(v), dim = -1, keepdim = True) )

# Spherical interpolation of a batch of vectors.
def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = torch.sum( a * b, dim = -1, keepdim = True )
    p = t * torch.acos( d )
    c = normalize(b - d * a)
    d = a * torch.cos( p ) + c * torch.sin( p )
    return normalize(d)

# Linear interpolation of a batch of vectors.
def lerp(a, b, t):
    return a + (b - a) * t

#----------------------------------------------------------------------------

def compute_ppl( evaluator, space = 'smart', num_samples = 100000, eps = 1e-4, net = 'vgg' ):
    """Perceptual Path Length: PyTorch implementation of the `PPL` class in
       https://github.com/NVlabs/stylegan/blob/master/metrics/perceptual_path_length.py
    """
    assert space in ['z', 'w', 'smart']
    assert net in ['vgg', 'alex']

    ppl_loss_fn = lpips.LPIPS( net = net, lpips = True )
    ppl_loss_fn.cuda()

    text_encoder, netG = evaluator.build_models_eval(init_func = weights_init)
    if space == 'smart':
        space = 'w' if cfg.GAN.B_STYLEGEN else 'z'
    if space == 'w':
        init_res = ( netG.h_net1.init_layer.shape[-2], netG.h_net1.init_layer.shape[-1], )
        upscale_fctr = netG.h_net1.upsampler.scale_factor
        res_init_layers = [ int( np.rint( r*upscale_fctr**((n-n%2)//2) ) ) for n,r in enumerate( init_res*5 ) ]
        res_2G = int( np.rint( netG.h_net2.res ) )
        res_3G = int( np.rint( netG.h_net3.res ) )

    batch_size = evaluator.batch_size
    nz = cfg.GAN.Z_DIM
    with torch.no_grad():
        z_code01 = Variable(torch.FloatTensor(batch_size * 2, nz))
        z_code01 = z_code01.cuda()
        t = Variable( torch.FloatTensor( batch_size, 1 ) )
        t = t.cuda()

    ppls = []
    dl_itr = iter( evaluator.data_loader )
    # for step, data in enumerate( evaluator.data_loader, 0 ):
    pbar = tqdm( range( num_samples // batch_size ), dynamic_ncols = True )
    for step in pbar:
        try:
            data = next( dl_itr )
        except StopIteration:
            dl_itr = iter( evaluator.data_loader )
            data = next( dl_itr )

        if step % 100 == 0:
            pbar.set_description( 'step: {}'.format( step ) )

        imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

        #######################################################
        # (1) Extract text embeddings
        ######################################################
        if evaluator.text_encoder_type == 'rnn':
            hidden = text_encoder.init_hidden( batch_size )
            words_embs, sent_emb = text_encoder( captions, cap_lens, hidden )
        elif evaluator.text_encoder_type == 'transformer':
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
        t.data.uniform_( 0, 1 )
        z_code01.data.normal_( 0, 1 )
        c_code, _, _ = netG.ca_net( sent_emb )
        sent_emb = torch.cat( ( sent_emb, sent_emb, ), 0 ).detach()
        words_embs = torch.cat( ( words_embs, words_embs, ), 0 ).detach()
        mask = torch.cat( ( mask, mask, ), 0 ).detach()
        if space == 'w':
            # Control out the StyleGAN noise, as we are trying to measure feature interpolability only in PPL
            netG.noise_net1 = [ torch.randn( batch_size, 1, res, res, dtype = torch.float32, device = z_code01.device ) for res in res_init_layers ]
            netG.noise_net1 = [ torch.cat( ( noise, noise, ), 0 ).detach() for noise in netG.noise_net1 ]
            netG.noise_net2 = [ torch.randn( batch_size, 1, res_2G, res_2G, dtype = torch.float32, device = z_code01.device ) for _ in range( 2 ) ]
            netG.noise_net2 = [ torch.cat( ( noise, noise, ), 0 ).detach() for noise in netG.noise_net2 ]
            netG.noise_net3 = [ torch.randn( batch_size, 1, res_3G, res_3G, dtype = torch.float32, device = z_code01.device ) for _ in range( 2 ) ]
            netG.noise_net3 = [ torch.cat( ( noise, noise, ), 0 ).detach() for noise in netG.noise_net3 ]
            w_code01 = netG.map_net( z_code01, torch.cat( ( c_code, c_code, ), 0 ) )
            w_code0, w_code1 = w_code01[0::2], w_code01[1::2]
            w_code0_lerp = lerp( w_code0, w_code1, t )
            w_code1_lerp = lerp( w_code0, w_code1, t + eps )
            w_code_lerp = torch.cat( ( w_code0_lerp, w_code1_lerp, ), 0 ).detach()
            fake_imgs01, _, _, _ = netG( w_code_lerp, sent_emb, words_embs, mask, is_dlatent = True )
        else:
            z_code0, z_code1 = z_code01[0::2], z_code01[1::2]
            z_code0_slerp = slerp( z_code0, z_code1, t )
            z_code1_slerp = slerp( z_code0, z_code1, t + eps )
            z_code_slerp = torch.cat( ( z_code0_slerp, z_code1_slerp, ), 0 ).detach()
            fake_imgs01, _, _, _ = netG( z_code_slerp, sent_emb, words_embs, mask )

        fake_imgs01 = fake_imgs01[-1]

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        if fake_imgs01.shape[2] > 256:
            factor = fake_imgs01.shape[2] // 256
            fake_imgs01 = torch.reshape( fake_imgs01, [-1, fake_imgs01.shape[1],
                                         fake_imgs01.shape[2] // factor, factor,
                                         fake_imgs01.shape[3] // factor, factor] )
            fake_imgs01 = torch.mean( fake_imgs01, dim = ( 3, 5, ), keepdim = False )

        # # Scale dynamic range to [-1,1] for the lpips VGG.
        # fake_imgs01 = (fake_imgs01 + 1) * (255 / 2)
        # fake_imgs01.clamp_( 0, 255 )
        fake_imgs01.clamp_( -1., 1. )

        fake_imgs0, fake_imgs1 = fake_imgs01[:batch_size], fake_imgs01[batch_size:]
        # fake_imgs0, fake_imgs1 = fake_imgs01[0::2], fake_imgs01[1::2]

        # Evaluate perceptual distances.
        ppls.append( ppl_loss_fn.forward( fake_imgs0, fake_imgs1 ).squeeze().detach().cpu().numpy() * (1 / 1e-4**2) )

    ppls = np.concatenate( ppls, axis = 0 )

    # Reject outliers.
    lo = np.percentile( ppls, 1, interpolation = 'lower' )
    hi = np.percentile( ppls, 99, interpolation = 'higher' )
    ppls = np.extract( np.logical_and( lo <= ppls, ppls <= hi ), ppls )

    return np.mean( ppls )
