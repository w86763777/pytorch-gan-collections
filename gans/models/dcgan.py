import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim, M):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.z_dim, 1024, M, 1, 0, bias=False),  # 4, 4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        weights_init(self)

    def forward(self, z):
        return self.main(z.view(-1, self.z_dim, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, M):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # 64
            nn.Conv2d(3, 64, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32
            nn.Conv2d(64, 128, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            # 16
            nn.Conv2d(128, 256, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),
            # 8
            nn.Conv2d(256, 512, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(512)
            # 4
        )
        self.linear = nn.Linear(M // 16 * M // 16 * 512, 1)
        weights_init(self)

    def forward(self, x):
        x = self.main(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


class Generator32(Generator):
    def __init__(self, z_dim):
        super().__init__(z_dim, M=2)


class Generator48(Generator):
    def __init__(self, z_dim):
        super().__init__(z_dim, M=4)


class Discriminator32(Discriminator):
    def __init__(self):
        super().__init__(M=32)


class Discriminator48(Discriminator):
    def __init__(self):
        super().__init__(M=48)


def weights_init(m):
    modules = (torch.nn.Conv2d, torch.nn.ConvTranspose2d)
    for param in m.modules():
        if isinstance(param, modules):
            torch.nn.init.xavier_normal_(param.weight.data)
            if param.bias is not None:
                torch.nn.init.zeros_(param.bias.data)
