import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
from tqdm import trange

from gans.scores.fid_score import fid_score
from gans.scores.inception_score import inception_score
from gans.scores.utils import IgnoreLabelDataset, GenerativeDataset


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(128, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh())

    def forward(self, z):
        return self.main(z.view(-1, 128, 1, 1))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 2, 1, bias=False),
            nn.AdaptiveAvgPool2d((1, 1)))

    def forward(self, x):
        x = self.main(x)
        return x.view(-1, 1)


def loop(dataloader):
    while True:
        for x in iter(dataloader):
            yield x


parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=100000)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--name', type=str, default='DCGAN')
parser.add_argument('--log-dir', type=str, default='log')
parser.add_argument('--z-dim', type=int, default=128)
parser.add_argument('--iter-G', type=int, default=3)
parser.add_argument('--sample-iter', type=int, default=1000)
parser.add_argument('--sample-size', type=int, default=64)
args = parser.parse_args()
log_dir = os.path.join(args.log_dir, args.name)

device = torch.device('cuda')
cifar10 = datasets.CIFAR10(
    './data', train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))
dataloader = torch.utils.data.DataLoader(
    cifar10, batch_size=args.batch_size, shuffle=True, num_workers=4,
    drop_last=True)


net_G = Generator().to(device)
net_D = Discriminator().to(device)

optim_G = optim.Adam(net_G.parameters(), lr=args.lr, betas=(0.5, 0.999))
optim_D = optim.Adam(net_D.parameters(), lr=args.lr, betas=(0.5, 0.999))

train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
valid_writer = SummaryWriter(os.path.join(log_dir, 'valid'))

real_label = torch.full((args.batch_size, 1), 1).to(device)
fake_label = torch.full((args.batch_size, 1), 0).to(device)
label = torch.cat([real_label, fake_label], dim=0)
criteria = nn.BCEWithLogitsLoss()

os.makedirs(os.path.join(log_dir, 'sample'), exist_ok=True)
sample_z = torch.randn(args.sample_size, args.z_dim).to(device)

valid_dataset = GenerativeDataset(net_G, args.z_dim, 10000, device)
looper = loop(dataloader)
with trange(args.iterations, dynamic_ncols=True) as pbar:
    for step in pbar:
        real, _ = next(looper)
        real = real.to(device)

        # update discriminator
        z = torch.randn(args.batch_size, args.z_dim).to(device)
        pred_D = torch.cat([net_D(real), net_D(net_G(z).detach())])
        loss_D = criteria(pred_D, label)
        optim_D.zero_grad()
        loss_D.backward()
        optim_D.step()
        train_writer.add_scalar('loss', loss_D.item(), step)

        if step % args.iter_G == 0:
            # update generator
            z = torch.randn(args.batch_size, args.z_dim).to(device)
            pred_G = net_D(net_G(z))
            loss_G = criteria(pred_G, real_label)
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()
            train_writer.add_scalar('loss/G', loss_G.item(), step)
            pbar.set_postfix(loss_D='%.4f' % loss_D.item(),
                             loss_G='%.4f' % loss_G.item())

        if step == 0:
            grid = (make_grid(real[:args.sample_size]) + 1) / 2
            train_writer.add_image('real sample', grid)

        if step == 0 or (step + 1) % args.sample_iter == 0:
            fake = net_G(sample_z).cpu()
            grid = (make_grid(fake) + 1) / 2
            valid_writer.add_image('sample', grid, step)
            save_image(grid, os.path.join(log_dir, 'sample', '%d.png' % step))

        if step == 0 or (step + 1) % 10000 == 0:
            torch.save(
                net_G.state_dict(), os.path.join(log_dir, 'G_%d.pt' % step))
            score, _ = inception_score(valid_dataset, batch_size=64, cuda=True)
            valid_writer.add_scalar('Inception Score', score, step)
            score = fid_score(IgnoreLabelDataset(cifar10), valid_dataset,
                              batch_size=64, cuda=True, normalize=True,
                              r_cache='./.fid_cache/cifar10')
            valid_writer.add_scalar('FID Score', score, step)
