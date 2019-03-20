# Collections of GANs

Pytorch implementation of GANs

## Models
- [ ] GAN
- [x] DCGAN
- [ ] WGAN
- [x] WGAN-GP
- [x] SN-GAN 

## Requirements
- python 3.6
- Install python packages
    ```bash
    $ pip install -r requirements.txt
    ```

## How To Run
- Train model
    ```bash
    $ python dcgan.py   # DCGAN
    $ python wgangp.py  # WGAN-GP
    $ python sngan.py   # SN-GAN
    ```
- [Optional] See tensorboard for more training details.
    ```bash
    $ tensorboard --logdir=log/
    ```
    open [0.0.0.0:6006](0.0.0.0:6006)

## Generate GIF
```bash
$ python sample2gif.py --name <model name>
```
All available model name is list in [Models](#Models)

## Results on CIFAR-10
- DCGAN
![](https://drive.google.com/uc?export=view&id=1QMmH3PeXXeOq6f-1kZlKk2hcQI3CSGE5) ![](https://drive.google.com/uc?export=view&id=1Br655M_Y4ghaola9iox4Ik9phVl47QBr)

- WGAN-GP
![](https://drive.google.com/uc?export=view&id=1VYjMLPulK_1iaNy4LntzsfNvHZOCVFz2) ![](https://drive.google.com/uc?export=view&id=1ZeioKXL2C9bgmQPQzXoMmQBHS_0wdHt_)

- SN-GAN
![](https://drive.google.com/uc?export=view&id=1niXJACfN97UntCtxTBDxN-A01ZaSWrAy) ![](https://drive.google.com/uc?export=view&id=1emYfU-84ef5pJCrqplxifXUvm29839R1)