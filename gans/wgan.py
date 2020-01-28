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
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.linear = nn.Linear(z_dim, 2 * 2 * 512)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh())

    def forward(self, z):
        outputs = self.linear(z)
        outputs = outputs.view(-1, 512, 2, 2)
        outputs = self.model(outputs)
        return outputs


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
            nn.LeakyReLU(0.2))
        self.linear = nn.Linear(2 * 2 * 512, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.model(x).view(batch_size, -1)
        x = self.linear(x)
        return x


def loop(dataloader):
    while True:
        for x in iter(dataloader):
            yield x


parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=200000)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--clip', type=float, default=1e-2)
parser.add_argument('--name', type=str, default='WGAN')
parser.add_argument('--log-dir', type=str, default='log')
parser.add_argument('--z-dim', type=int, default=128)
parser.add_argument('--iter-D', type=int, default=5)
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

net_G = Generator(args.z_dim).to(device)
net_D = Discriminator().to(device)

optim_G = optim.RMSprop(net_G.parameters(), lr=args.lr)
optim_D = optim.RMSprop(net_D.parameters(), lr=args.lr)

train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
valid_writer = SummaryWriter(os.path.join(log_dir, 'valid'))

os.makedirs(os.path.join(log_dir, 'sample'), exist_ok=True)
sample_z = torch.randn(args.sample_size, args.z_dim).to(device)

valid_dataset = GenerativeDataset(net_G, args.z_dim, 10000, device)
looper = loop(dataloader)
with trange(args.iterations, dynamic_ncols=True) as pbar:
    for step in pbar:
        real, _ = next(looper)
        real = real.to(device)

        z = torch.randn(args.batch_size, args.z_dim).to(device)
        with torch.no_grad():
            fake = net_G(z).detach()
        loss = -net_D(real).mean() + net_D(fake).mean()
        optim_D.zero_grad()
        loss.backward()
        optim_D.step()
        train_writer.add_scalar('loss', -loss.item(), step)
        pbar.set_postfix(loss='%.4f' % -loss.item())

        for param in net_D.parameters():
            param.data.clamp_(-args.clip, args.clip)

        if step % args.iter_D == 0:
            z = torch.randn(args.batch_size, args.z_dim).to(device)
            loss_G = -net_D(net_G(z)).mean()
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

        if step == 0:
            grid = (make_grid(real) + 1) / 2
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