import numpy as np
import torch
import torch.utils.data
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models.inception import inception_v3
from scipy.stats import entropy


def inception_score(imgs, batch_size=32, cuda=False, bins=1, scale=True):
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    if cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Set up dataloader
    dataloader = DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model = inception_model.type(torch.FloatTensor).to(device)
    inception_model.eval()

    def get_pred(x):
        with torch.no_grad():
            x = F.interpolate(x, size=(299, 299), mode='bilinear',
                              align_corners=True)
            x = inception_model(x)
            x = F.softmax(x, dim=1)
            x = x.data.cpu().numpy()
        return x

    # Get predictions
    preds = []
    for batch in dataloader:
        batch = batch.type(torch.FloatTensor).to(device)
        with torch.no_grad():
            if scale:
                batch = F.interpolate(batch, size=(299, 299), mode='bilinear',
                                      align_corners=True)
            x = inception_model(batch)
            x = F.softmax(x, dim=1)
            x = x.data.cpu().numpy()
        preds.append(x)
    preds = np.concatenate(preds)

    # compute the mean kl-div
    split_scores = []
    for k in range(bins):
        if k + 1 != bins:
            part = preds[k * (N // bins): (k + 1) * (N // bins), :]
        else:
            part = preds[k * (N // bins):, :]
        # marginal prob
        p_y = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            # label prob
            p_yx = part[i, :]
            scores.append(entropy(p_yx, p_y))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


if __name__ == '__main__':
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    cifar = dset.CIFAR10(
        root='./data', download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    print(inception_score(
        IgnoreLabelDataset(cifar),
        batch_size=32,
        cuda=True,
        bins=10))
