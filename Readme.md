# Collections of GANs

Pytorch implementation of unsupervised GANs.

For more defails about calculating Inception Score and FID using pytorch can be found in [pytorch-inception-score-fid](https://github.com/w86763777/pytorch-inception-score-fid) 

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
|DCGAN          |CIFAR10|6.04(0.05)     |47.90|
|WGAN(CNN)      |CIFAR10|6.64(0.6)      |33.27|
|WGAN-GP(CNN)   |CIFAR10|7.47(0.06)     |24.00|
|WGAN-GP(ResNet)|CIFAR10|7.74(0.10)     |21.89|
|SNGAN(CNN)     |CIFAR10|7.44(0.11)     |24.94|
|SNGAN(ResNet)  |CIFAR10|8.22(0.13)     |16.24|

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
- Download [cifar10.test.npz](https://drive.google.com/drive/folders/1UBdzl6GtNMwNQ5U-4ESlIer43tNjiGJC?usp=sharing) for calculating FID score. Then, create folder `stats` for the npz files
    ```
    stats
    ├── cifar10.test.npz
    ├── cifar10.train.npz
    └── stl10.unlabeled.48.npz
    ```

- Train from scratch
    ```bash
    # DCGAN
    python dcgan.py --flagfile ./config/DCGAN_CIFAR10.txt
    # WGAN(CNN)
    python wgan.py --flagfile ./config/WGAN_CIFAR10_CNN.txt
    # WGAN-GP(CNN)
    python wgangp.py --flagfile ./config/WGANGP_CIFAR10_CNN.txt
    # WGAN-GP(ResNet)
    python wgangp.py --flagfile ./config/WGANGP_CIFAR10_RES.txt
    # SNGAN(CNN)
    python sngan.py --flagfile ./config/SNGAN_CIFAR10_CNN.txt
    # SNGAN(ResNet)
    python sngan.py --flagfile ./config/SNGAN_CIFAR10_RES.txt
    ```
    Though the training procedures of different GANs are almost identical, I still separate different methods into different files for clear reading.

## Learning curve
![inception_score_curve](https://drive.google.com/uc?export=view&id=12JTJS5--2dDjFyVhHJ-b264Qp3S-v8xS)
![fid_curve](https://drive.google.com/uc?export=view&id=1P4e_DEyW4wvFubPSu5t_i2gVRoecGqs5)

## Change Log
- 2021-04-16
    - Update pytorch to 1.8.1
    - Move metrics to submodule.
    - Evaluate FID on CIFAR10 test set instead of training set.
    - Fix `cifar10.test.npz` download link and sample images.