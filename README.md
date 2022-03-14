# Style-AttnGAN

PyTorch implementation of a modified (style-based) [AttnGAN](https://arxiv.org/abs/1711.10485) [[code]](https://github.com/taoxugit/AttnGAN) architecture that incorporates the strong latent space control provided by [StyleGAN](https://arxiv.org/abs/1812.04948)*. This architecture enables one to not only synthesize an image from an input text description, but also move that image in a desired disentangled dimension to alter its structure at different scales (from high-level coarse styles such as pose to fine-grained styles such as background lighting).

<p align="center"><b><i>GIF OF LATENT SPACE INTERPOLATION EXAMPLE COMING SOON</i></b></p>

| "this is a black bird with gray and white wings and a bright yellow belly and chest." | "tiny bird with long thighs, and a long pointed brown bill." <img width=240/>| "a small bird has a royal blue crown, black eyes, and a baby blue colored bill." |
|:--:|:--:|:--:|
<img src="https://github.com/sidward14/Style-AttnGAN/raw/master/examples/for_readme/Baltimore_Oriole/Baltimore_Oriole_Style-AttnGAN.png" width="240" height="240"/> | <img src="https://github.com/sidward14/Style-AttnGAN/raw/master/examples/for_readme/Mockingbird/Mockingbird_Style-AttnGAN.png" width="240" height="240"/> | <img src="https://github.com/sidward14/Style-AttnGAN/raw/master/examples/for_readme/Indigo_Bunting/Indigo_Bunting_Style-AttnGAN.png" width="240" height="240"/> |

This implementation also provides one with the option to utilize state-of-the-art transformer-based architectures from huggingface's [transformers](https://github.com/huggingface/transformers) library as the text encoder for Style-AttnGAN (currently only supports GPT-2). Among other things, utilization of these transformer-based encoders significantly improves image synthesis when the length of the input text sequence is large.

Original AttnGAN paper: [AttnGAN: Fine-Grained Text to Image Generation
with Attentional Generative Adversarial Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf) by Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, Xiaodong He. (This work was performed when Tao was an intern with Microsoft Research). Thank you for your brilliant work.

<p align="middle">
  <img src="https://github.com/sidward14/Style-AttnGAN/raw/master/examples/for_readme/framework.png" width="590" height="229"/>
  <img src="https://github.com/sidward14/Style-AttnGAN/raw/master/examples/for_readme/stylegan-generator.png" width="200" height="229"/>
</p>

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;*AttnGAN architecture from [arxiv.org/abs/1711.10485](https://arxiv.org/abs/1711.10485)* &emsp;&emsp;&emsp;&nbsp; *StyleGAN generator architecture*
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp; *from [arxiv.org/abs/1812.04948](https://arxiv.org/abs/1812.04948)* 

### Copied from LICENSE file (MIT License) for visibility:
*Copyright for portions of project Style-AttnGAN are held by Tao Xu, 2018 as part of project AttnGAN. All other copyright for project Style-AttnGAN are held by Sidhartha Parhi, 2020. __All non-data files that have not been modified by Sidhartha Parhi include the copyright notice "Copyright (c) 2018 Tao Xu" at the top of the file.__*

--------------------------------------------------------------------------------
### Instructions:

**Dependencies**

python 3.7+

PyTorch 1.0+

In addition, please add the project folder to PYTHONPATH and `pip install` the following packages (or go `pip install -r requirements.txt`):
- `python-dateutil`
- `easydict`
- `pandas`
- `torchfile`
- `nltk`
- `scikit-image`
- `pyyaml`
- `tqdm`
- `pytorch-fid`
- `lpips`
- `transformers`
- `gan-lab`



**Data**

1. Download preprocessed data from taoxugit for [birds](https://drive.google.com/open?id=1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ) [coco](https://drive.google.com/open?id=1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9) and save them to `data/
2. Unzip `data/birds/text.zip` and/or `data/coco/train2014-text.zip` & `data/coco/val2014-text.zip`
3. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to `data/birds/`
4. Download [coco](http://cocodataset.org/#download) dataset and extract the images to `data/coco/`



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

- `*.yml` files are example configuration files for training/evaluation models.
- After pretraining DAMSM, save best text & image encoders into `DAMSMencoders/` and specify the text encoder path in NET_E in the appropriate Style-AttnGAN yaml or in code/miscc/config.py. Alternatrively, skip this step by downloading appropriate DAMSM in the "Pretrained Model" section below.
- User can also optionally set the `--text_encoder_type` flag to `transformer` to use GPT-2 as the text encoder (more text encoder options coming soon). If doing so, user must pre-train the DAMSM (see instruction above).



**Pretrained Model**
- [DAMSM for bird](https://drive.google.com/open?id=1GNUKjVeyWYBJ8hEU-yrfYQpDOkxEyP3V). Download and save it to `DAMSMencoders/`. Text encoder here is BiLSTM.
- [DAMSM for coco](https://drive.google.com/open?id=1zIrXCE9F6yfbEJIbNP5-YrEe2pZcPSGJ). Download and save it to `DAMSMencoders/`. Text encoder here is BiLSTM.
- [Style-AttnGAN for bird](https://drive.google.com/file/d/11Fo003VQJbXK9OBT18PESlP6wqtbaQAo/view?usp=sharing). Download and save it to `models/`
- [Style-AttnGAN for coco](). Download and save it to `models/`
  - _COMING SOON_
- [Original AttnGAN for bird](https://drive.google.com/open?id=1lqNG75suOuR_8gjoEPYNp8VyT_ufPPig). Download and save it to `models/`
- [Original AttnGAN for coco](https://drive.google.com/open?id=1i9Xkg9nU74RAvkcqKE-rJYhjvzKAMnCi). Download and save it to `models/`

- [AttnDCGAN for bird](https://drive.google.com/open?id=19TG0JUoXurxsmZLaJ82Yo6O0UJ6aDBpg). Download and save it to `models/`
  - This is an variant of AttnGAN which applies the propsoed attention mechanisms to DCGAN framework. 

**Sampling**
- Run `python main.py --cfg cfg/eval_bird_style.yml --gpu 0` to generate examples from captions in files listed in "./data/birds/example_filenames.txt". Results are saved in `models/`. 
- Change the `eval_*.yml` files to generate images from other pre-trained models. 
- Input your own sentence in "./data/birds/example_captions.txt" if you wannt to generate images from customized sentences. 

**Validation**
- To generate images for all captions in the validation dataset, <i>change B_VALIDATION to True</i> in the eval_*.yml file, and then run `python main.py --cfg cfg/eval_bird_style.yml --gpu 0`
- Metrics can be computed based on the validation set. Set the corresponding booleans to <i>True</i> in the eval_*.yml file in order to do so.
  - The following metrics can be computed with this Style-AttnGAN repo directly:    
  &nbsp;&nbsp;I. &nbsp;&nbsp;&nbsp;PPL:  
  &emsp;&emsp;&emsp;Perceptual Path Length. A measure of linear interpolability of the latent space.  
  &emsp;&emsp;&emsp;Derived from [LPIPS](https://arxiv.org/abs/1801.03924). The lower the better. A lower score is an indication that one  
  &emsp;&emsp;&emsp;can more easily tune the latent code to generate the exact image he/she desires.  
  &nbsp;&nbsp;II. &nbsp;&nbsp;FID:  
  &emsp;&emsp;&emsp;Fréchet Inception Distance. A measure of generated image quality.  
  &emsp;&emsp;&emsp;Proposed as an improvement over the Inception Score (IS). The lower the better.  
  &nbsp;&nbsp;III. &nbsp;R-precision:  
  &emsp;&emsp;&emsp;A measure of visual-semantic similarity between generated images and  
  &emsp;&emsp;&emsp;their corresponding text description. Expressed as a percentage.  
  &emsp;&emsp;&emsp;_CLEANED UP IMPLEMENTATION COMING SOON_
  - To compute Inception Score (IS), use [this repo](https://github.com/hanzhanggit/StackGAN-inception-model) for the birds dataset and [this repo](https://github.com/openai/improved-gan/tree/master/inception_score) for the COCO dataset.

--------------------------------------------------------------------------------
### Custom Dataset:
Below are instructions for setting up and training & testing with your own dataset. I use the dataset barebones_birds as an example custom dataset.

**Data**
- I've included a bare-bones version of the birds dataset above (download link: [barebones_birds](https://drive.google.com/drive/folders/1ez0cviLIvSZV8xspwJ0NYJJvyY5F9lRN?usp=sharing)). Extract it into `data/`. Then, go into `data/barebones_birds` and extract `text.zip` into `text/` and `images.zip` into `images/`.

  - This dataset consists of only images and captions for those images. Use this as an example on how to set up your own custom single-class dataset for training and inference.
  - Note that this dataset uses jpg. If you want to use some other extension (e.g. png) instead, you need to specify the extension in EXT_IN and EXT_OUT in the 3 YAML files discussed below.

**Training**
- Pre-train DAMSM:
  - For your own custom dataset, you will have to pre-train the DAMSM for it. To do so, first you will have to set up the YAML file for it. See `code/cfg/DAMSM/barebones_bird.yml` for an example on how to set that up.
  - Run `python pretrain_DAMSM.py --cfg cfg/DAMSM/barebones_bird.yml --gpu 0`
  - After pretraining DAMSM, save best text & image encoders into `DAMSMencoders/`

- Train Style-AttnGAN
  - Set up the YAML file for Style-AttnGAN training. See `code/cfg/barebones_bird_attn2_style.yml` for an example on how to set that up. Make sure to specify the path to the best text encoder you put in `DAMSMencoders/` for NET_E.
  - Run `python main.py --cfg cfg/barebones_bird_attn2_style.yml --gpu 0`
  - The Style-AttnGAN training above will periodicially save your models

**Inference**
- Set up the YAML file for Style-AttnGAN inference. See `code/cfg/eval_barebones_bird_style.yml` for an example on how to set that up. Make sure to se that same path for NET_E that you used above for barebones_bird_attn2_style.yml, and specify the path to the best Style-AttnGAN for NET_G.
- After that, follow the instructions in the section above for Sampling and / or Validation, except instead of using evals_bird_style.yml, use eval_barebones_bird_style.yml.

--------------------------------------------------------------------------------
### Qualitative performance comparisons on birds dataset (CUB-200-2011):

**Examples generated by Style-AttnGAN vs AttnGAN for input text:**  
|  | "this is a black bird with gray and white wings and a bright yellow belly and chest." | "tiny bird with long thighs, and a long pointed brown bill." <img width=210/>| "a small bird has a royal blue crown, black eyes, and a baby blue colored bill." |
|:--:|:--:|:--:|:--:|
| Style-AttnGAN Generated Images | <img src="https://github.com/sidward14/Style-AttnGAN/raw/master/examples/for_readme/Baltimore_Oriole/Baltimore_Oriole_Style-AttnGAN.png" width="210" height="210"/> | <img src="https://github.com/sidward14/Style-AttnGAN/raw/master/examples/for_readme/Mockingbird/Mockingbird_Style-AttnGAN.png" width="210" height="210"/> | <img src="https://github.com/sidward14/Style-AttnGAN/raw/master/examples/for_readme/Indigo_Bunting/Indigo_Bunting_Style-AttnGAN.png" width="210" height="210"/> |
| AttnGAN Generated Images |  <img src="https://github.com/sidward14/Style-AttnGAN/raw/master/examples/for_readme/Baltimore_Oriole/Baltimore_Oriole_AttnGAN.png" width="210" height="210"/> | <img src="https://github.com/sidward14/Style-AttnGAN/raw/master/examples/for_readme/Mockingbird/Mockingbird_AttnGAN.png" width="210" height="210"/> | <img src="https://github.com/sidward14/Style-AttnGAN/raw/master/examples/for_readme/Indigo_Bunting/Indigo_Bunting_AttnGAN.png" width="210" height="210"/>
| Real Images | <img src="https://github.com/sidward14/Style-AttnGAN/raw/master/examples/for_readme/Baltimore_Oriole/Baltimore_Oriole_real.jpg" width="210" height="210"/> | <img src="https://github.com/sidward14/Style-AttnGAN/raw/master/examples/for_readme/Mockingbird/Mockingbird_real.png" width="210" height="210"/> | <img src="https://github.com/sidward14/Style-AttnGAN/raw/master/examples/for_readme/Indigo_Bunting/Indigo_Bunting_real.png" width="210" height="210"/> |

_I have visually inspected a few hundred generated images from both Style-AttnGAN and AttnGAN. My human judgement is that the Style-AttnGAN images are more photo-realistic and are much more consistent than the AttnGAN images. I plan to test out Style-AttnGAN with other datasets and compare to the SOTA in the near future._

**Examples of latent space interpolation with Style-AttnGAN**
- _GIF COMING SOON_

**Examples of style-mixing with Style-AttnGAN**
- _COMING SOON_

**Comparisons with MirrorGAN and DM-GAN**
- _COMING SOON_

--------------------------------------------------------------------------------
### Quantitative performance comparisons on birds dataset (CUB-200-2011):
**Notes about the table in general:** The arrows next to each metric title indicate whether lower is better or higher is better. Official reports of scores are used wherever possible; otherwise scores are computed using this repo. "(-)" indicates that the value hasn't been computed/recorded yet; _these will be updated soon_.

**Note about FID**: [It has been mentioned](https://github.com/MinfengZhu/DM-GAN) that the [PyTorch implementation of FID](https://github.com/mseitzer/pytorch-fid) produces different scores than the [Tensorflow implementation of FID](https://github.com/bioinf-jku/TTUR). _**Also, even though the quantitative quality metrics in the table below (i.e. FID and IS) for Style-AttnGAN seem to be on the lower end relative to the SOTA, visual inspection of these generated images tells a different story. Upon visual inspection of a few hundred generated images, the Style-AttnGAN generated images look more photo-realistic and are much more consistent than the AttnGAN generated images. See the qualitative comparison above for some examples of this.**_

**Note about PPL**:  The PPL (Perceptual Path Length) implementation in Style-AttnGAN aims to replicate the PPL implementation in the [official StyleGAN repo](https://github.com/NVlabs/stylegan). To get a sense of good vs bad scores, please look at the README for this official StyleGAN repo and/or going through the [StyleGAN paper](https://arxiv.org/abs/1812.04948). _Note that for some of these text-to-image generation architectures (e.g. AttnGAN), it is difficult (and probably not possible) to obtain a measure of linear interpolability of the latent space using the PPL metric due to how the text embedding and the noise are mixed together and learned. Nevertheless, my qualitative experiments clearly show the advantage of Style-AttnGAN over AttnGAN in terms of linear interpolation of the latent space. I will put examples of these qualitative experiments in this README soon._

|Model| PPL↓ | R-precision↑ | IS↑ | PyTorch FID↓ | TF FID↓ |
|:--:|:--:|:--:|:--:|:--:|:--:|
| Style-AttnGAN | 493.7656 | (-) | 4.33 ± 0.09 | 25.60 | (-) |
| AttnGAN | (-) | 67.82% ± 4.43% | 4.36 ± 0.03 | 23.98 | 14.01 |
| MirrorGAN | (-) | 60.42% ± (-) | 4.56 ± 0.05 | (-) | (-) |
| DM-GAN | (-) | 72.31% ± 0.91% | 4.75 ± 0.07 | 16.09 | (-) |

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
- [x] Add link to downloading pretrained Style-AttnGAN models
- [ ] Provide examples for latent space control (GIF) and style-mixing
- [ ] Complete the Table of quantitative performance comparisons
- [ ] Qualitative comparisons with MirrorGAN and DM-GAN
- [ ] Implement easier API functionality for latent space control
- [ ] Implement improvements from StyleGAN2
- [ ] Analysis with COCO dataset
- [ ] Implement more options for SOTA transformer architectures as the text encoder (currently only supports GPT-2)
- [ ] Deploy as an interactive web app that makes it easy to control the specific image one wants to generate

*Improvements from StyleGAN2 coming soon