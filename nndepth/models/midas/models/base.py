import torch
import torch.nn as nn


class BaseDepthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder: nn.Module = self.build_encoder()
        self.decoder: nn.Module = self.build_decoder()
        self.last_conv: nn.Module = self.build_last_conv()

    def load_weights(self, weights: str, strict_load: bool = True):
        if weights.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(weights)
        elif weights.endswith(".pth"):
            state_dict = torch.load(weights)
        else:
            raise ValueError(f"Unsupported weight format: {weights}")

        self.load_state_dict(state_dict, strict=strict_load)

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
