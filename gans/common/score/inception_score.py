import argparse

import numpy as np
import torch
from tqdm import trange

from inception import InceptionV3


def get_inception_score(images, device, splits=10, batch_size=32,
                        verbose=False):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM['prob']
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    preds = []
    if verbose:
        iterator = trange(0, len(images), batch_size, dynamic_ncols=True)
    else:
        iterator = range(0, len(images), batch_size)
    for start in iterator:
        end = start + batch_size
        batch_images = images[start: end]
        batch_images = torch.from_numpy(batch_images).type(torch.FloatTensor)
        batch_images = batch_images.to(device)
        pred = model(batch_images)[0]
        preds.append(pred.cpu().numpy())

    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
        part = preds[
            (i * preds.shape[0] // splits):
            ((i + 1) * preds.shape[0] // splits), :]
        kl = part * (
            np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def main(args):
    from torchvision import datasets, transforms
    device = torch.device('cuda:0')

    dataloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            args.inception_dir, train=True, download=True,
            transform=transforms.ToTensor()),
        batch_size=64, shuffle=False, num_workers=6, drop_last=False)

    images = []
    for batch_images, _ in dataloader:
        with torch.no_grad():
            images.append(batch_images.cpu().numpy())
    images = np.concatenate(images, axis=0)

    m, s = get_inception_score(
        images, device, args.splits, args.batch_size, verbose=True)
    print(m, s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Calculate Inception score of CIFAR10")
    parser.add_argument("--inception_dir", type=str, default='./data',
                        help='path to inception model dir')
    parser.add_argument("--splits", type=int, default=10,
                        help="bins (original)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    main(parser.parse_args())
