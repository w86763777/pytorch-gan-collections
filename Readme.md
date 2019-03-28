# Collections of GANs

Pytorch implementation of GANs

## Models
- [ ] GAN
- [x] DCGAN
- [x] WGAN
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

    ![](https://drive.google.com/uc?export=view&id=1iGE9cwVDmiB2sCpT92Sg3PmlEu5dlf6r) ![](https://drive.google.com/uc?export=view&id=1iKKBEF7pXoq1v4xglco6Isypl33QI1zj)

- WGAN

    ![](https://drive.google.com/uc?export=view&id=1v3v_j8zPDY01RWRBc-ClO97L_zsRxPli) ![](https://drive.google.com/uc?export=view&id=1qmnnTrs3RF71WiQ5tx4SwJIyPmmlbVS5)

- WGAN-GP

    ![](https://drive.google.com/uc?export=view&id=172Nhzr8E_usITv_Z_0JAOKCM2ishyplW) ![](https://drive.google.com/uc?export=view&id=1GRvbxoN-dubX53NtjIg2KXR0i3Ow1WaI)

- SN-GAN

    ![](https://drive.google.com/uc?export=view&id=1itXkiwjemOT2uOjYUIKPr_t7myBjeFz5) ![](https://drive.google.com/uc?export=view&id=1OVvr9xs5pV-BEQ5JlVK_3r4stUcbzox8)