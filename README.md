# Style-AttnGAN

Pytorch implementation of a modified (styled) [AttnGAN](https://arxiv.org/abs/1711.10485) [[code]](https://github.com/taoxugit/AttnGAN) architecture that incorporates the strong latent space control provided by [StyleGAN](https://arxiv.org/abs/1812.04948)*. This architecture enables one to not only synthesize an image from an input text description, but also move that image in a desired disentangled dimension to alter its structure at different scales (from high-level coarse styles such as pose to fine-grained styles such as background lighting).

<p align="center"><b><i>GIF OF LATENT SPACE CONTROL EXAMPLE COMING SOON</i></b></p>

This implementation also provides one with the option to utilize state-of-the-art transformer architectures from huggingface's [transformers](https://github.com/huggingface/transformers) library as the text encoder for Style-AttnGAN (currently only supports GPT-2).

Original AttnGAN paper: [AttnGAN: Fine-Grained Text to Image Generation
with Attentional Generative Adversarial Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf) by Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, Xiaodong He. (This work was performed when Tao was an intern with Microsoft Research). Thank you for your brilliant work.

<img src="framework.png" width="900px" height="350px"/>

### Copied from LICENSE file (MIT License) for visibility:
*Copyright for portions of project Style-AttnGAN are held by Tao Xu, 2018 as part of project AttnGAN. All other copyright for project Style-AttnGAN are held by Sidhartha Parhi, 2020. __All non-data files that have not been modified by Sidhartha Parhi will include the copyright notice "Copyright (c) 2018 Tao Xu" at the top of the file.__*

--------------------------------------------------------------------------------

### Dependencies
python 3.6+

Pytorch 1.0+

In addition, please add the project folder to PYTHONPATH and `pip install` the following packages:
- `python-dateutil`
- `easydict`
- `pandas`
- `torchfile`
- `nltk`
- `scikit-image`
- `transformers`
- `gan-lab`



**Data**

1. Download preprocessed metadata from taoxugit for [birds](https://drive.google.com/open?id=1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ) [coco](https://drive.google.com/open?id=1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9) and save them to `data/`
2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to `data/birds/`
3. Download [coco](http://cocodataset.org/#download) dataset and extract the images to `data/coco/`



**Training**
- Pre-train DAMSM models:
  - For bird dataset: `python pretrain_DAMSM.py --cfg cfg/DAMSM/bird.yml --gpu 0`
  - For coco dataset: `python pretrain_DAMSM.py --cfg cfg/DAMSM/coco.yml --gpu 0`
 
- Train Style-AttnGAN models:
  - For bird dataset: `python main.py --cfg cfg/bird_attn2_style.yml --gpu 0`
  - For coco dataset: `python main.py --cfg cfg/coco_attn2_style.yml --gpu 0`

- Train original AttnGAN models:
  - For bird dataset: `python main.py --cfg cfg/bird_attn2.yml --gpu 0`
  - For coco dataset: `python main.py --cfg cfg/coco_attn2.yml --gpu 0`

- `*.yml` files are example configuration files for training/evaluation our models.
- User can also optionally set the `--text_encoder_type` flag to `transformer` to use GPT-2 as the text encoder (more text encoder options coming soon).



**Pretrained Model**
- [DAMSM for bird](https://drive.google.com/open?id=1GNUKjVeyWYBJ8hEU-yrfYQpDOkxEyP3V). Download and save it to `DAMSMencoders/`
- [DAMSM for coco](https://drive.google.com/open?id=1zIrXCE9F6yfbEJIbNP5-YrEe2pZcPSGJ). Download and save it to `DAMSMencoders/`
- [Style-AttnGAN for bird](). Download and save it to `models/`
  - _COMING SOON_
- [Style-AttnGAN for coco](). Download and save it to `models/`
  - _COMING SOON_
- [Original AttnGAN for bird](https://drive.google.com/open?id=1lqNG75suOuR_8gjoEPYNp8VyT_ufPPig). Download and save it to `models/`
- [Original AttnGAN for coco](https://drive.google.com/open?id=1i9Xkg9nU74RAvkcqKE-rJYhjvzKAMnCi). Download and save it to `models/`

- [AttnDCGAN for bird](https://drive.google.com/open?id=19TG0JUoXurxsmZLaJ82Yo6O0UJ6aDBpg). Download and save it to `models/`
  - This is an variant of AttnGAN which applies the propsoed attention mechanisms to DCGAN framework. 

**Sampling**
- Run `python main.py --cfg cfg/eval_bird.yml --gpu 0` to generate examples from captions in files listed in "./data/birds/example_filenames.txt". Results are saved to `DAMSMencoders/`. 
- Change the `eval_*.yml` files to generate images from other pre-trained models. 
- Input your own sentence in "./data/birds/example_captions.txt" if you wannt to generate images from customized sentences. 

**Validation**
- To generate images for all captions in the validation dataset, change B_VALIDATION to True in the eval_*.yml. and then run `python main.py --cfg cfg/eval_bird.yml --gpu 0`
- We compute inception score for models trained on birds using [StackGAN-inception-model](https://github.com/hanzhanggit/StackGAN-inception-model). [Coming soon for Style-AttnGAN]
- We compute inception score for models trained on coco using [improved-gan/inception_score](https://github.com/openai/improved-gan/tree/master/inception_score). [Coming soon for Style-AttnGAN]

--------------------------------------------------------------------------------

**Examples of latent space control in Style-AttnGAN**
- _GIF COMING SOON_

**Examples of style-mixing in Style-AttnGAN**
- _COMING SOON_

**Examples generated by Style-AttnGAN**

- _COMING SOON_

**Examples generated by original AttnGAN [[Blog]](https://blogs.microsoft.com/ai/drawing-ai/)**

 bird example              |  coco example
:-------------------------:|:-------------------------:
![](https://github.com/taoxugit/AttnGAN/blob/master/example_bird.png)  |  ![](https://github.com/taoxugit/AttnGAN/blob/master/example_coco.png)

### Inception Score, R-precision, and qualitative comparisons with original AttnGAN and MirrorGAN
- ***COMING SOON***

--------------------------------------------------------------------------------

### Creating an API
[Evaluation code](eval) embedded into a callable containerized API is included in the `eval\` folder.

### Citing Original AttnGAN
If you find the original AttnGAN useful in your research, please consider citing:

```
@article{Tao18attngan,
  author    = {Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, Xiaodong He},
  title     = {AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks},
  Year = {2018},
  booktitle = {{CVPR}}
}
```

**Reference**

- [AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks](https://arxiv.org/abs/1711.10485) [[code]](https://github.com/taoxugit/AttnGAN)
- [StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/abs/1710.10916) [[code]](https://github.com/hanzhanggit/StackGAN-v2)
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) [[code]](https://github.com/carpedm20/DCGAN-tensorflow)

--------------------------------------------------------------------------------

### TODO:
- [ ] Add link to downloading pretrained Style-AttnGAN models
- [ ] Provide examples for latent space control (GIF) and style-mixing
- [ ] __Inception Score and R-precision comparisons with original AttnGAN and MirrorGAN__
- [ ] __Qualitative comparisons with original AttnGAN and MirrorGAN__
- [ ] Implement easier API functionality for latent space control
- [ ] Implement improvements from StyleGAN2
- [ ] Implement more options for SOTA transformer architectures as the text encoder (currently only supports GPT-2)

*Improvements from StyleGAN2 coming soon