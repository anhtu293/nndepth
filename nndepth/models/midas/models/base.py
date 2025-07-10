import torch
import torch.nn as nn


class BaseDepthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder: nn.Module = self.build_encoder()
        self.decoder: nn.Module = self.build_decoder()
        self.last_conv: nn.Module = self.build_last_conv()

    def build_encoder(self) -> nn.Module:
        raise NotImplementedError

    def build_decoder(self) -> nn.Module:
        raise NotImplementedError

    def build_last_conv(self) -> nn.Module:
        raise NotImplementedError

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        raise NotImplementedError

    def forward_decoder(self, feats: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_encoder(x)
        output = self.forward_decoder(feats)
        output = self.last_conv(output)
        return output
