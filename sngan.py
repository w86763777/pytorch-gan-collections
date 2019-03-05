import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import spectral_norm
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from tqdm import tqdm


class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim

        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=1, padding=1),
            nn.Sigmoid())

    def forward(self, z):
        return self.model(z.view(-1, self.z_dim, 1, 1))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, 4, 2, padding=1, bias=False)),
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, padding=1, bias=False)),
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, padding=1, bias=False)),
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Conv2d(256, 512, 4, 2, padding=1, bias=False)),
            nn.LeakyReLU(0.1))
        self.linear = spectral_norm(nn.Linear(2 * 2 * 512, 1))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.model(x).view(batch_size, -1)
        x = self.linear(x)
        return x


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--name', type=str, default='SN-WGAN')
parser.add_argument('--log-dir', type=str, default='log')
parser.add_argument('--z-dim', type=int, default=128)
parser.add_argument('--D-iter', type=int, default=5)
parser.add_argument('--sample-iter', type=int, default=500)
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataloader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        './data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, num_workers=4)


net_G = Generator(args.z_dim).to(device)
net_D = Discriminator().to(device)

optim_G = optim.Adam(net_G.parameters(), lr=args.lr, betas=(0.0, 0.9))
optim_D = optim.Adam(net_D.parameters(), lr=args.lr, betas=(0.0, 0.9))

# use an exponentially decaying learning rate
scheduler_G = optim.lr_scheduler.ExponentialLR(optim_G, gamma=0.99)
scheduler_D = optim.lr_scheduler.ExponentialLR(optim_D, gamma=0.99)

train_writer = SummaryWriter(os.path.join(args.log_dir, 'train'))
valid_writer = SummaryWriter(os.path.join(args.log_dir, 'valid'))

sample_z = torch.randn(args.batch_size, args.z_dim).to(device)

iter_num = 0
for epoch in range(args.epochs):
    with tqdm(dataloader) as t:
        t.set_description('Epoch %2d/%2d' % (epoch + 1, args.epochs))
        for data, _ in t:
            data = data.to(device)
            if iter_num == 0:
                train_writer.add_image('real sample', make_grid(data))

            # update discriminator
            optim_D.zero_grad()
            z = torch.randn(args.batch_size, args.z_dim).to(device)
            loss_D = -net_D(data).mean() + net_D(net_G(z)).mean()
            loss_D.backward()
            optim_D.step()
            train_writer.add_scalar('loss', -loss_D.item(), iter_num)
            t.set_postfix(loss='%.4f' % -loss_D.item())

            if iter_num % args.D_iter == 0:
                # update generator
                optim_G.zero_grad()
                z = torch.randn(args.batch_size, args.z_dim).to(device)
                loss_G = -net_D(net_G(z)).mean()
                loss_G.backward()
                optim_G.step()

            if iter_num % args.sample_iter == 0:
                fake = net_G(sample_z).cpu()
                valid_writer.add_image('sample', make_grid(fake), iter_num)

            iter_num += 1
        scheduler_G.step()
        scheduler_D.step()
