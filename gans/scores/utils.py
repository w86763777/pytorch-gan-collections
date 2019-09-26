import torch


class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)


class GenerativeDataset(torch.utils.data.Dataset):
    """Only support DataLoader with num_worker=1"""

    def __init__(self, generator, z_dim, dataset_size, device):
        self.generator = generator
        self.z_dim = z_dim
        self.dataset_size = dataset_size
        self.device = device

    def __getitem__(self, index):
        z = torch.randn(1, self.z_dim).to(self.device)
        self.generator.eval()
        with torch.no_grad():
            image = self.generator(z)
        self.generator.train()
        return image[0]

    def __len__(self):
        return self.dataset_size