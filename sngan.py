import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import spectral_norm
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
from tqdm import tqdm


class GenResBlock(nn.Module):
    def __init__(self, channels):
        class Upsample(nn.Module):
            def forward(self, x):
                return torch.nn.functional.interpolate(x, scale_factor=2)

        super().__init__()
        self.blocks = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, padding=1),
            Upsample(),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, padding=1),
        )
        self.residual = Upsample()

    def forward(self, x):
        return self.residual(x) + self.blocks(x)


class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.linear = nn.Linear(z_dim, 4 * 4 * 256)

        self.blocks = nn.Sequential(
            GenResBlock(256),
            GenResBlock(256),
            GenResBlock(256),
            nn.ReLU(),
            nn.Conv2d(256, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        inputs = self.linear(z)
        inputs = inputs.view(-1, 256, 4, 4)
        return self.blocks(inputs)


class DisResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=False):
        super().__init__()
        self.down = False
        if down or in_channels != out_channels:
            self.down = True
            self.residual = spectral_norm(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1))
        stride = 2 if down else 1
        self.blocks = nn.Sequential(
            nn.ReLU(),
            spectral_norm(nn.Conv2d(in_channels, in_channels, 3, stride, 1)),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        )

    def forward(self, x):
        if self.down:
            return self.residual(x) + self.blocks(x)
        else:
            return x + self.blocks(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            DisResBlock(3, 128, down=True),
            DisResBlock(128, 128, down=True),
            DisResBlock(128, 128),
            DisResBlock(128, 128),
            nn.ReLU(),
            nn.AvgPool2d((8, 8)))
        self.linear = spectral_norm(nn.Linear(128, 1))

    def forward(self, x):
        x = self.model(x).view(-1, 128)
        x = self.linear(x)
        return x


def loop(dataloader):
    while True:
        for x in iter(dataloader):
            yield x


parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=100000)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--name', type=str, default='SN-GAN-CIFAR10')
parser.add_argument('--log-dir', type=str, default='log')
parser.add_argument('--z-dim', type=int, default=128)
parser.add_argument('--D-iter', type=int, default=5)
parser.add_argument('--sample-iter', type=int, default=500)
parser.add_argument('--sample-size', type=int, default=64)
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataloader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        './data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])),
    batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)


net_G = Generator(args.z_dim).to(device)
net_D = Discriminator().to(device)

optim_G = optim.Adam(net_G.parameters(), lr=args.lr, betas=(0.5, 0.9))
optim_D = optim.Adam(net_D.parameters(), lr=args.lr, betas=(0.5, 0.9))

train_writer = SummaryWriter(os.path.join(args.log_dir, args.name, 'train'))
valid_writer = SummaryWriter(os.path.join(args.log_dir, args.name, 'valid'))

os.makedirs(os.path.join(args.log_dir, args.name, 'sample'), exist_ok=True)
sample_z = torch.randn(args.sample_size, args.z_dim).to(device)

with tqdm(total=args.iterations) as t:
    for iter_num, (real, _) in enumerate(loop(dataloader)):
        if iter_num == args.iterations:
            break
        real = real.to(device)
        if iter_num == 0:
            grid = (make_grid(real) + 1) / 2
            train_writer.add_image('real sample', grid)

        optim_D.zero_grad()
        z = torch.randn(args.batch_size, args.z_dim).to(device)
        with torch.no_grad():
            fake = net_G(z).detach()
        loss_real = torch.nn.functional.relu(1 - net_D(real)).mean()
        loss_fake = torch.nn.functional.relu(1 + net_D(fake)).mean()
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optim_D.step()
        train_writer.add_scalar('loss', loss_D.item(), iter_num)
        t.set_postfix(loss='%.4f' % loss_D.item())

        if iter_num % args.D_iter == 0:
            optim_G.zero_grad()
            z = torch.randn(args.batch_size, args.z_dim).to(device)
            loss_G = -net_D(net_G(z)).mean()
            loss_G.backward()
            optim_G.step()

        if iter_num % args.sample_iter == 0:
            fake = net_G(sample_z).cpu()
            grid = (make_grid(fake) + 1) / 2
            valid_writer.add_image('sample', grid, iter_num)
            save_image(grid, os.path.join(
                args.log_dir, args.name, 'sample', '%d.png' % iter_num))

        t.update(1)
        if (iter_num + 1) % 10000 == 0:
            torch.save(
                net_G.state_dict(),
                os.path.join(args.log_dir, args.name, 'G_%d.pt' % iter_num))
