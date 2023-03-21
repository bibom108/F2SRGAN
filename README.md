# F2SRGAN
This is an official implementation of paper "F2SRGAN: A Lightweight Approach Boosting Perceptual Quality in Single Image Super-Resolution via A Revised Fast Fourier Convolution".

# Data prepare
You should organize the images layout like this:

```shell
SR_training_datasets
├── DIV2K_train_HR
├── Flickr2K
└── DIV2K_valid_HR

SR_tesing_datasets
├── Set5
├── Set14
├── BSDS100
└── Urban100
```

# Train
There are 5 mode of training:
- **pre**: L1 loss only.
- **per**: Perceptual loss only.
- **gan**: Perceptual loss and gan loss.
- **rgan**: Perceptual loss and rgan loss.
- **full**: The proposed loss in paper.  

To replicate the paper's result, first train model with `pre` mode, followed by `full` mode.
# Test
Pretrained weight for `x2` and `x4` upscale factor are provided in `pretrain_weight`.
# Evaluate
- First run `inference.py` to produce output images.
- To measure the Perceptual Index, please refer to this [Repository](https://github.com/roimehrez/PIRM2018) for more information

# References
- [SwiftSRGAN](https://github.com/Koushik0901/Swift-SRGAN)
- [LaMa](https://github.com/advimman/lama)
