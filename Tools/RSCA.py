import torch
import torch.nn as nn


class RSCA(torch.nn.Module):
    def __init__(self):
        super(RSCA, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=1, num_heads=1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x, y) -> torch.Tensor:
        x = self.avgpool(x)
        identy_x = x
        identy_y = y
        B = identy_y.shape[0]
        N = identy_y.shape[1]
        x = x.view(-1, identy_x.shape[1], 1)
        y = y.view(-1, identy_y.shape[1], 1)
        self_x, _ = self.attention(x, x, x)
        self_y, _ = self.attention(y, y, y)
        cross_x, _ = self.attention(x, y, y)
        cross_y, _ = self.attention(y, x, x)

        f_x = identy_x.view(B, N) + self_x.view(B, N)
        f_y = identy_y.view(B, N) + self_y.view(B, N)
        c_f_x = cross_x.view(B, N) + f_x
        c_f_y = cross_y.view(B, N) + f_y
        cat_feature = torch.cat((c_f_x, c_f_y), dim=1)
        return cat_feature
