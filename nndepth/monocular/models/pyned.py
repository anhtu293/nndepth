import torch
import torch.nn as nn
from typing import Dict



class PydNet(nn.Module):
    """Pydnet from https://arxiv.org/pdf/1806.11430.pdf"""

    def __init__(self, config: Dict, **kwargs):
        super().__init__()
        self.config = config
        self.enc_0, self.enc_1, self.enc_2, self.enc_3, self.enc_4, self.enc_5 = self.build_encoder()
        self.dec_0, self.dec_1, self.dec_2, self.dec_3, self.dec_4, self.dec_5 = self.build_decoder()

    def build_encoder(self):
        raise NotImplementedError

    def build_decoder(self):
        raise NotImplementedError

    def _build_enc_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

    def _build_dec_block(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=1, padding=1),
        )

    def forward_enc(self, frame: torch.Tensor):
        raise NotImplementedError

    def forward(self, frame: torch.Tensor):
        raise NotImplementedError
