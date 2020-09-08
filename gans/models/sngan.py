import math
from functools import partial

import torch
import torch.nn as nn

from torch.nn.utils.spectral_norm import spectral_norm


class LeakySoftplus(nn.Softplus):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * x + (1 - self.alpha) * super().forward(x)


G_activation_maps = {
    'relu': nn.ReLU,
    'selu': nn.SELU,
    'elu': nn.ELU,
    'softplus': nn.Softplus,
    'softplus20': partial(nn.Softplus, beta=20),
}

D_activation_maps = {
    'relu': nn.ReLU,
    'selu': nn.SELU,
    'elu': nn.ELU,
    'softplus': nn.Softplus,
    'softplus20': partial(nn.Softplus, beta=20),
    'leakyrelu': partial(nn.LeakyReLU, 0.1),
    'leakysoftplus': LeakySoftplus,
}


class Generator(nn.Module):
    def __init__(self, z_dim, activation='relu', M=4):
        super().__init__()
        self.z_dim = z_dim
        self.M = M
        self.linear = nn.Linear(self.z_dim, M * M * 512)
        self.main = nn.Sequential(
            nn.BatchNorm2d(512),
            G_activation_maps[activation](),
            nn.ConvTranspose2d(
                512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            G_activation_maps[activation](),
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            G_activation_maps[activation](),
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            G_activation_maps[activation](),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh())
        dcgan_weights_init(self)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(x.size(0), -1, self.M, self.M)
        x = self.main(x)
        return x


# In CR-SNGAN, the channel sizes are    [64, 128, 128, 256, 256, 512, 512]
# In SNGAN, the channel sizes are       [64, 64, 128, 128, 256, 256, 512]
class Discriminator(nn.Module):
    def __init__(self, activation, M=32):
        super().__init__()
        self.M = M

        self.main = nn.Sequential(
            # M
            spectral_norm(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)),
            D_activation_maps[activation](),
            spectral_norm(
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            D_activation_maps[activation](),
            # M / 2
            spectral_norm(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
            D_activation_maps[activation](),
            spectral_norm(
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            D_activation_maps[activation](),
            # M / 4
            spectral_norm(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            D_activation_maps[activation](),
            spectral_norm(
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            D_activation_maps[activation](),
            # M / 8
            spectral_norm(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
            D_activation_maps[activation]()
        )

        self.linear = spectral_norm(nn.Linear(M // 8 * M // 8 * 512, 1))
        dcgan_weights_init(self)

    def forward(self, x):
        x = self.main(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


class Generator32(Generator):
    def __init__(self, z_dim, activation='relu'):
        super().__init__(z_dim, activation, M=4)


class Generator48(Generator):
    def __init__(self, z_dim, activation='relu'):
        super().__init__(z_dim, activation, M=6)


class Discriminator32(Discriminator):
    def __init__(self, activation='leakysoftplus'):
        super().__init__(activation, M=32)


class Discriminator48(Discriminator):
    def __init__(self, activation='leakysoftplus'):
        super().__init__(activation, M=48)


class ResGenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            G_activation_maps[activation](),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            G_activation_maps[activation](),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ResGenerator32(nn.Module):
    def __init__(self, z_dim, activation='relu'):
        super().__init__()
        self.z_dim = z_dim
        self.linear = nn.Linear(z_dim, 4 * 4 * 256)

        self.blocks = nn.Sequential(
            ResGenBlock(256, 256, activation),
            ResGenBlock(256, 256, activation),
            ResGenBlock(256, 256, activation),
            nn.BatchNorm2d(256),
            G_activation_maps[activation](),
            nn.Conv2d(256, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        weights_init(self)

    def forward(self, z):
        inputs = self.linear(z)
        inputs = inputs.view(-1, 256, 4, 4)
        return self.blocks(inputs)


class ResGenerator48(nn.Module):
    def __init__(self, z_dim, activation='relu'):
        super().__init__()
        self.z_dim = z_dim
        self.linear = nn.Linear(z_dim, 6 * 6 * 512)

        self.blocks = nn.Sequential(
            ResGenBlock(512, 256, activation),
            ResGenBlock(256, 128, activation),
            ResGenBlock(128, 64, activation),
            nn.BatchNorm2d(64),
            G_activation_maps[activation](),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        weights_init(self)

    def forward(self, z):
        inputs = self.linear(z)
        inputs = inputs.view(-1, 512, 6, 6)
        return self.blocks(inputs)


class OptimizedResDisblock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0)))
        self.residual = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1)),
            D_activation_maps[activation](),
            spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1)),
            nn.AvgPool2d(2))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ResDisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=False,
                 activation='relu'):
        super().__init__()
        shortcut = []
        if in_channels != out_channels or down:
            shortcut.append(spectral_norm(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0)))
        if down:
            shortcut.append(nn.AvgPool2d(2))
        self.shortcut = nn.Sequential(*shortcut)

        residual = [
            D_activation_maps[activation](),
            spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1)),
            D_activation_maps[activation](),
            spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1)),
        ]
        if down:
            residual.append(nn.AvgPool2d(2))
        self.residual = nn.Sequential(*residual)

    def forward(self, x):
        return (self.residual(x) + self.shortcut(x))


class ResDiscriminator32(nn.Module):
    def __init__(self, activation='relu'):
        super().__init__()
        self.model = nn.Sequential(
            OptimizedResDisblock(3, 128, activation),
            ResDisBlock(128, 128, down=True, activation=activation),
            ResDisBlock(128, 128, activation),
            ResDisBlock(128, 128, activation),
            D_activation_maps[activation]())
        self.linear = spectral_norm(nn.Linear(128, 1, bias=False))
        weights_init(self)

    def forward(self, x):
        x = self.model(x).sum(dim=[2, 3])
        x = self.linear(x)
        return x


class ResDiscriminator48(nn.Module):
    def __init__(self, activation='relu'):
        super().__init__()
        self.model = nn.Sequential(
            OptimizedResDisblock(3, 64, activation),
            ResDisBlock(64, 128, down=True, activation=activation),
            ResDisBlock(128, 256, down=True, activation=activation),
            ResDisBlock(256, 512, down=True, activation=activation),
            D_activation_maps[activation]())
        self.linear = spectral_norm(nn.Linear(512, 1, bias=False))
        weights_init(self)

    def forward(self, x):
        x = self.model(x).sum(dim=[2, 3])
        x = self.linear(x)
        return x


def weights_init(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            if 'residual' in name:
                torch.nn.init.xavier_uniform_(module.weight, gain=math.sqrt(2))
            else:
                torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


def dcgan_weights_init(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
