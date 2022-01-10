# Collections of GANs

Pytorch implementation of unsupervised GANs.

For more defails about calculating Inception Score and FID using pytorch can be found here [pytorch_gan_metrics](https://github.com/w86763777/pytorch-gan-metrics) 

## Models
- [x] DCGAN
- [x] WGAN
- [x] WGAN-GP
- [x] SN-GAN 

## Requirements
- Install python packages
    ```bash
    pip install -U pip setuptools
    pip install -r requirements.txt
    ```

## Results

|Model          |Dataset|Inception Score|FID  |
|--------------|:-----:|:--------------:|:---:|
|DCGAN          |CIFAR10|5.91(0.15)     |47.46|
|WGAN(CNN)      |CIFAR10|6.46(0.24)     |38.98|
|WGAN-GP(CNN)   |CIFAR10|7.69(0.19)     |22.81|
|WGAN-GP(ResNet)|CIFAR10|7.80(0.20)     |21.48|
|SNGAN(CNN)     |CIFAR10|7.64(0.21)     |21.86|
|SNGAN(ResNet)  |CIFAR10|8.21(0.17)     |19.11|

## Examples
- DCGAN

    ![dcgan_gif](https://drive.google.com/uc?export=view&id=1KSBk0Va_4bJXgM7TruZgfHly9If_74-n) ![dcgan_png](https://drive.google.com/uc?export=view&id=1GGa6xd28dHds5lJTRCqnJks-ie_3luH8)

- WGAN(CNN)

    ![wgan_gif](https://drive.google.com/uc?export=view&id=1Yrq-peVwSTUQmK_CF0dcf86mXYp2Qd4D) ![wgan_png](https://drive.google.com/uc?export=view&id=1vER25JI0U9awv25x1Muz5-b7sY-Wyi7d)

- WGAN-GP(CNN)

    ![wgangp_cnn_gif](https://drive.google.com/uc?export=view&id=1wUqaKPo4BhCcByHyTEuNhDeb2CfUJB6f) ![wgangp_cnn_png](https://drive.google.com/uc?export=view&id=1w-9N5c7s-f7Ocb6DGVtLloOe5Xkl4CRd)

- WGAN-GP(ResNet)

    ![wgangp_res_gif](https://drive.google.com/uc?export=view&id=16gadJh0K4ZWelmTqIjkPLlk2P423EpCR) ![wgangp_res_png](https://drive.google.com/uc?export=view&id=1ZRYJo7rtbN99hK71OT2dvj94uxzGG063)

- SNGAN(CNN)

    ![sngan_cnn_gif](https://drive.google.com/uc?export=view&id=1zWtmiwsYJqSqxY7LISBzaBfZHrzbbhXi) ![sngan_cnn_png](https://drive.google.com/uc?export=view&id=1Uq387vzBWptqDWk1c5s1jRYxcuN-IzGZ)

- SNGAN(ResNet)

    ![sngan_res_gif](https://drive.google.com/uc?export=view&id=1et3V7NbLEqH6aOWzkOQceNcnfY3WBOGz) ![sngan_res_png](https://drive.google.com/uc?export=view&id=1neYWCexP8kY2eixMpztNL50TKFLXZcBL)

## Reproduce
- Download [cifar10.test.npz](https://drive.google.com/drive/folders/1UBdzl6GtNMwNQ5U-4ESlIer43tNjiGJC?usp=sharing) for calculating FID. Then, create folder `stats` for the npz files
    ```
    stats
    └── cifar10.test.npz
    ```

- Train from scratch
    ```bash
    # DCGAN
    python dcgan.py --flagfile ./configs/DCGAN_CIFAR10.txt
    # WGAN(CNN)
    python wgan.py --flagfile ./configs/WGAN_CIFAR10_CNN.txt
    # WGAN-GP(CNN)
    python wgangp.py --flagfile ./configs/WGANGP_CIFAR10_CNN.txt
    # WGAN-GP(ResNet)
    python wgangp.py --flagfile ./configs/WGANGP_CIFAR10_RES.txt
    # SNGAN(CNN)
    python sngan.py --flagfile ./configs/SNGAN_CIFAR10_CNN.txt
    # SNGAN(ResNet)
    python sngan.py --flagfile ./configs/SNGAN_CIFAR10_RES.txt
    ```
    Though the training procedures of different GANs are almost identical, I still separate different methods into different files for clear reading.

## Learning curve
![inception_score_curve](https://drive.google.com/uc?export=view&id=12JTJS5--2dDjFyVhHJ-b264Qp3S-v8xS)
![fid_curve](https://drive.google.com/uc?export=view&id=1P4e_DEyW4wvFubPSu5t_i2gVRoecGqs5)

## Change Log
- 2021-01-10
    - Update pytorch to 1.10.1 and CUDA 11.3
    - Calculate FID and Inception by `pytorch_gan_metrics`
    - Use 50k generated images and CIFAR10 train set to calculate FID
    - Fix default parameters

- 2021-04-16
    - Update pytorch to 1.8.1
    - Move metrics to submodule.
    - Evaluate FID on CIFAR10 test set instead of training set.
    - Fix `cifar10.test.npz` download link and sample images.