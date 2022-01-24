

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

def sampling( text_encoder_type, split_dir ):
    if split_dir == 'test':
        split_dir = 'valid'
    model_dir = cfg.TRAIN.NET_G  # the path to save generated images

    # Build and load the generator and text encoder
    text_encoder, netG = self.build_models_eval(init_func = weights_init)  # TODO:

    batch_size = self.batch_size  # TODO:
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
        for step, data in enumerate(self.data_loader, 0):  # TODO:
            cnt += batch_size
            if step % 100 == 0:
                print('step: ', step)
            # if step > 50:
            #     break

            imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

            #######################################################
            # (1) Extract text embeddings
            ######################################################
            if text_encoder_type == 'rnn':
                hidden = text_encoder.init_hidden(batch_size)
                words_embs, sent_emb = text_encoder( captions, cap_lens, hidden )
            elif text_encoder_type == 'transformer':
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

def gen_example( ixtoword, data_dic, text_encoder_type ):
    model_dir = cfg.TRAIN.NET_G  # the path to save generated images

    # Build and load the generator and text encoder
    text_encoder, netG = self.build_models_eval()  # TODO:

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
            if text_encoder_type == 'rnn':
                hidden = text_encoder.init_hidden(batch_size)
                words_embs, sent_emb = text_encoder( captions, cap_lens, hidden )
            elif text_encoder_type == 'transformer':
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
                                            [cap_lens_np[j]], ixtoword,
                                            [attn_maps[j]], att_sze)
                    if img_set is not None:
                        im = Image.fromarray(img_set)
                        fullpath = '%s_a%d.png' % (save_name, k)
                        im.save(fullpath)
    return save_dirs