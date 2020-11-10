import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataloader, Dataset


class FakeData(Dataset):
    def __init__(self, dims=2, n_classes=3):
        super().__init__()
        self.dims = dims
        self.n_points = 4
        self.n_classes = n_classes

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        y = torch.rand_int(self.n_classes, (1,))
        mean = y.float()
        std = y.float().sqrt()
        x = torch.randn(self.dims, self.n_points) * std + mean

        return x, y


class Model(nn.Module):
    def __init__(self, in_dim=2, latent_dim=3):
        super().__init__()

        self.in_dim = in_dim
        self.concat_block0 = nn.Sequential(nn.Conv1d(in_channels=in_dim,
                                                    out_channels=latent_dim,
                                                    kernel_size=1),
                                          nn.Batchnorm1d(latent_dim),
                                          nn.ReLU(),
                                          nn.Conv1d(in_channels=latent_dim,
                                                    out_channels=latent_dim,
                                                    kernel_size=1),
                                          nn.Batchnorm1d(latent_dim),
                                          nn.ReLU(),
                                          nn.Conv1d(in_channels=latent_dim,
                                                    out_channels=latent_dim
                                                                 - in_dim,
                                                    kernel_size=1),
                                          )

        self.sum_block1 = nn.Sequential(nn.Conv1d(in_channels=latent_dim,
                                                    out_channels=latent_dim,
                                                    kernel_size=1),
                                          nn.Batchnorm1d(latent_dim),
                                          nn.ReLU(),
                                          nn.Conv1d(in_channels=latent_dim,
                                                    out_channels=latent_dim,
                                                    kernel_size=1),
                                          nn.Batchnorm1d(latent_dim),
                                          nn.ReLU(),
                                          nn.Conv1d(in_channels=latent_dim,
                                                    out_channels=latent_dim,
                                                    kernel_size=1),
                                          )

        self.sum_block2 = nn.Sequential(nn.Conv1d(in_channels=latent_dim,
                                                    out_channels=latent_dim,
                                                    kernel_size=1),
                                          nn.Batchnorm1d(latent_dim),
                                          nn.ReLU(),
                                          nn.Conv1d(in_channels=latent_dim,
                                                    out_channels=latent_dim,
                                                    kernel_size=1),
                                          nn.Batchnorm1d(latent_dim),
                                          nn.ReLU(),
                                          nn.Conv1d(in_channels=latent_dim,
                                                    out_channels=latent_dim,
                                                    kernel_size=1),
                                          )

        self.classifier = nn.Conv1d(in_channels=latent_dim,
                                    out_channels=latent_dim,
                                    kernel_size=1)

    def get_rotation_matrix(self, origin, target):
        enc_o = torch.eye(self.in_dim)[origin]
        enc_t = torch.eye(self.in_dim)[target]
        rotation_matrix = enc_o.pinverse() @ enc_t
        return rotation_matrix

    def forward(self, x, origin, target):
        """
        :param x: [B, F, N]
        :param origin: torch.tensor[B]
        :param target: torch.tensor[B]
        :return:
        """

        local_feature0 = self.concat_block0(x)
        global_feature0 = local_feature0.max(dim=-1, keepdims=True)[0]
        x1 = torch.cat((x, global_feature0), dim=1)
        local_feature1 = self.sum_block1(x1)
        x2 = x1 + local_feature1
        local_feature2 = self.sum_block2(x2)
        x3 = x2 + local_feature2

        rotation_matrix = self.get_rotation_matrix(origin, target)

        y3 = x3 @ rotation_matrix
        y2 = y3 - local_feature2
        y1 = y2 - local_feature1
        y = y1[:, self.in_dim:, :]

        y_class = self.classifier(y3)
        x_class = self.classifier(x3)

        return y, y_class, x_class

