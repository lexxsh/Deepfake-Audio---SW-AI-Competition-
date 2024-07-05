import numpy as np
import torch

from src import frontends
from src.models.whisper_main import ModelDimensions, Whisper, log_mel_spectrogram
from src.models.specrnet import SpecRNet
from src.commons import WHISPER_MODEL_WEIGHTS_PATH


class WhisperSpecRNet(SpecRNet):
    def __init__(self, input_channels, freeze_encoder, **kwargs):
        super().__init__(input_channels=input_channels, **kwargs)

        self.device = kwargs["device"]
        checkpoint = torch.load(WHISPER_MODEL_WEIGHTS_PATH)
        dims = ModelDimensions(**checkpoint["dims"].__dict__)
        model = Whisper(dims)
        model = model.to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.whisper_model = model
        if freeze_encoder:
            for param in self.whisper_model.parameters():
                param.requires_grad = False

    def compute_whisper_features(self, x):
        specs = []
        for sample in x:
            specs.append(log_mel_spectrogram(sample))
        x = torch.stack(specs)
        x = self.whisper_model(x)

        x = x.permute(0, 2, 1)  # (bs, frames, 3 x n_lfcc)
        x = x.unsqueeze(1)  # (bs, 1, frames, 3 x n_lfcc)
        x = x.repeat(
            (1, 1, 1, 2)
        )  # (bs, 1, frames, 3 x n_lfcc) -> (bs, 1, frames, 3000)
        return x

    def forward(self, x):
        # we assume that the data is correct (i.e. 30s)
        x = self.compute_whisper_features(x)
        out = self._compute_embedding(x)
        return out


class WhisperMultiFrontSpecRNet(WhisperSpecRNet):
    def __init__(self, input_channels, freeze_encoder, **kwargs):
        super().__init__(
            input_channels=input_channels,
            freeze_encoder=freeze_encoder,
            **kwargs,
        )
        self.frontend = frontends.get_frontend(kwargs["frontend_algorithm"])
        print(f"Using {self.frontend} frontend!")

    def forward(self, x):
        # Frontend computation
        frontend_x = self.frontend(x)
        x = self.compute_whisper_features(x)

        x = torch.cat([x, frontend_x], 1)
        out = self._compute_embedding(x)
        return out


if __name__ == "__main__":
    import numpy as np

    input_channels = 1
    config = {
        "filts": [input_channels, [input_channels, 20], [20, 64], [64, 64]],
        "nb_fc_node": 64,
        "gru_node": 64,
        "nb_gru_layer": 2,
        "nb_classes": 1,
    }
    device = "cpu"
    classifier = WhisperSpecRNet(
        input_channels,
        freeze_encoder=False,
        device=device,
    )
    input_channels = 2
    classifier_2 = WhisperMultiFrontSpecRNet(
        input_channels,
        freeze_encoder=False,
        device=device,
        frontend_algorithm="lfcc"
    )
    x = np.random.rand(2, 30 * 16_000).astype(np.float32)
    x = torch.from_numpy(x)

    out = classifier(x)
    print(out.shape)

    out = classifier_2(x)
    print(out.shape)