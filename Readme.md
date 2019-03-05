# Spectral Normalization GAN

- Pytorch implementation of [SN-GAN](https://arxiv.org/abs/1802.05957)
- The architecture of generator and discriminator are based on [DCGAN](https://arxiv.org/abs/1511.06434)
- [Wasserstein Loss](https://arxiv.org/abs/1701.07875)

## Requirements
- python 3.6
- Install python packages
    ```bash
    $ pip install -r requirements.txt
    ```

## How To Run
- Train model
    ```bash
    $ python sngan.py
    ```
- [Optional] See tensorboard for more training details.
    ```bash
    $ tensorboard --logdir=log/
    ```
    open [0.0.0.0:6006](0.0.0.0:6006)