# Collections of GANs

Pytorch implementation of unsupervised GANs.

For more defails about calculating Inception Score and FID Score using pytorch can be found in [Pytorch-Unified-Inception-FID-Score](https://github.com/w86763777/Pytorch-Unified-Inception-FID-Score) 

## Models
- [x] DCGAN
- [x] WGAN
- [x] WGAN-GP
- [x] SN-GAN 

## Requirements
- python 3.6
- Install python packages
    ```bash
    pip install -U pip setuptools
    pip install -r requirements.txt
    ```

## Results

|Model          |Dataset|Inception Score|FID Score|
|---------------|-------|---------------|---------|
|DCGAN          |CIFAR10|6.111(0.088)   |41.75    |
|WGAN(CNN)      |CIFAR10|6.605(0.071)   |30.73    |
|WGAN-GP(CNN)   |CIFAR10|7.415(0.065)   |21.89    |
|WGAN-GP(ResNet)|CIFAR10|7.829(0.076)   |15.57    |
|SNGAN(CNN)     |CIFAR10|7.521(0.111)   |20.41    |
|SNGAN(ResNet)  |CIFAR10|8.214(0.094)   |14.41    |

## Examples
- DCGAN

    ![image](https://drive.google.com/uc?export=view&id=14vz9JTxi4A8p5x2kiS7STnAMMGJb8_U0) ![image](./images/dcgan_cifar10.png)

- WGAN(CNN)

    ![image](https://drive.google.com/uc?export=view&id=12Y5E-Vf-U-fzPLO_bmhhLNK1-q0Lo_OS) ![image](./images/wgan_cifar10_cnn.png)

- WGAN-GP(CNN)

    ![image](https://drive.google.com/uc?export=view&id=1i7B2i_nDZrTyvhOefmEHRs_mGXU7mv4Q) ![image](./images/wgangp_cifar10_cnn.png)

- WGAN-GP(ResNet)

    ![image](https://drive.google.com/uc?export=view&id=1WbMPMUwd2ltDkqowBMcIwUWP7dF87LH0) ![image](./images/wgangp_cifar10_res.png)

- SNGAN(CNN)

    ![image](https://drive.google.com/uc?export=view&id=1tQyWeyjNNOlWWBPo2XwhwZQ9t1q5a1v5) ![image](./images/sngan_cifar10_cnn.png)

- SNGAN(ResNet)

    ![image](https://drive.google.com/uc?export=view&id=1CN6vgPqodAQBtp9OElPvCaNakomKKP4E) ![image](./images/sngan_cifar10_res.png)

## Reproduce

### Training
- DCGAN
	```
    python gans/dcgan.py --flagfile ./config/DCGAN_CIFAR10.txt
    ```
- WGAN(CNN)
	```
    python gans/wgan.py --flagfile ./config/WGAN_CIFAR10_CNN.txt
    ```
- WGAN-GP(CNN)
	```
    python gans/wgangp.py --flagfile ./config/WGANGP_CIFAR10_CNN.txt
    ```
- WGAN-GP(ResNet)
	```
    python gans/wgangp.py --flagfile ./config/WGANGP_CIFAR10_RES.txt
    ```
- SNGAN(CNN)
	```
    python gans/sngan.py --flagfile ./config/SNGAN_CIFAR10_CNN.txt
    ```
- SNGAN(ResNet)
	```
    python gans/sngan.py --flagfile ./config/SNGAN_CIFAR10_RES.txt
    ```

### Generate GIF
```bash
python tools/sample2gif.py --logdir path/to/logdir
```
e.g.
```bash
python tools/sample2gif.py --logdir ./logs/DCGAN_CIFAR10
```
output GIF is `./logs/DCGAN_CIFAR10/progress.gif`