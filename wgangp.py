import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
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
            nn.Conv2d(32, 3, 3, stride=1, padding=1),
            nn.Tanh())

    def forward(self, z):
        return self.model(z.view(-1, self.z_dim, 1, 1))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
            nn.LeakyReLU(0.1))
        self.linear = nn.Linear(2 * 2 * 512, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.model(x).view(batch_size, -1)
        x = self.linear(x)
        return x


def calc_gradient_penalty(net_D, real, fake):
    alpha = torch.rand(real.size(0), 1, 1, 1).to(device)
    alpha = alpha.expand(real.size())

    interpolates = alpha * real.detach() + (1 - alpha) * fake.detach()
    interpolates.requires_grad = True
    disc_interpolates = net_D(interpolates)
    ones = torch.ones(disc_interpolates.size()).to(device)
    gradients = autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=ones,
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=[1, 2]))
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()
    return gradient_penalty


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--name', type=str, default='WGAN-GP')
parser.add_argument('--log-dir', type=str, default='log')
parser.add_argument('--z-dim', type=int, default=128)
parser.add_argument('--alpha', type=float, default=10)
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
    batch_size=args.batch_size, shuffle=True, num_workers=4)

net_G = Generator(args.z_dim).to(device)
net_D = Discriminator().to(device)

one = torch.ones(1).to(device)
mone = -torch.ones(1).to(device)

optim_G = optim.Adam(net_G.parameters(), lr=args.lr, betas=(0.5, 0.9))
optim_D = optim.Adam(net_D.parameters(), lr=args.lr, betas=(0.5, 0.9))

scheduler_G = optim.lr_scheduler.ExponentialLR(optim_G, gamma=0.995)
scheduler_D = optim.lr_scheduler.ExponentialLR(optim_D, gamma=0.995)

train_writer = SummaryWriter(os.path.join(args.log_dir, args.name, 'train'))
valid_writer = SummaryWriter(os.path.join(args.log_dir, args.name, 'valid'))

os.makedirs(os.path.join(args.log_dir, args.name, 'sample'))
sample_z = torch.randn(args.sample_size, args.z_dim).to(device)

iter_num = 0
for epoch in range(args.epochs):
    with tqdm(dataloader) as t:
        t.set_description('Epoch %2d/%2d' % (epoch + 1, args.epochs))
        for real, _ in t:
            real = real.to(device)
            if iter_num == 0:
                grid = (make_grid(real) + 1) / 2
                train_writer.add_image('real sample', grid)

            # update discriminator
            optim_D.zero_grad()
            z = torch.randn(real.size(0), args.z_dim).to(device)
            fake = net_G(z)
            loss_gp = calc_gradient_penalty(net_D, real, fake)
            loss_w = -net_D(real).mean() + net_D(fake.detach()).mean()
            loss_D = loss_w + args.alpha * loss_gp
            loss_D.backward()
            optim_D.step()
            train_writer.add_scalar('loss', -loss_w.item(), iter_num)
            train_writer.add_scalar('loss_gp', loss_gp.item(), iter_num)
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
                grid = (make_grid(fake) + 1) / 2
                valid_writer.add_image('sample', grid, iter_num)
                save_image(grid, os.path.join(
                    args.log_dir, args.name, 'sample', '%d.png' % iter_num))
            iter_num += 1
        scheduler_G.step()
        scheduler_D.step()
    if (epoch + 1) % 100 == 0:
        torch.save(net_G.state_dict(),
                   os.path.join(args.log_dir, args.name, 'net_G.pt'))
