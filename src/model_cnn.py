from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class NoiseCNN(nn.Module):
    """CNN with optional structured auxiliary heads for source recognition."""

    def __init__(
        self,
        num_classes: int,
        auxiliary_heads: bool = False,
        input_channels: int = 1,
        architecture: str = "lightweight",
        base_channels: int = 32,
        dropout: float = 0.0,
        prediction_mode: str = "multilabel",
        combo_score_weight: float = 1.0,
        multilabel_score_weight: float = 0.3,
        count_score_weight: float = 0.2,
    ):
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if architecture not in {"lightweight", "residual"}:
            raise ValueError(f"Unsupported model architecture: {architecture}")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if prediction_mode not in {"multilabel", "structured"}:
            raise ValueError(f"Unsupported prediction mode: {prediction_mode}")
        if prediction_mode == "structured" and not auxiliary_heads:
            raise ValueError("Structured prediction requires auxiliary heads")

        self.architecture = architecture
        self.num_classes = num_classes
        self.auxiliary_heads = bool(auxiliary_heads)
        self.prediction_mode = prediction_mode
        self.combo_score_weight = float(combo_score_weight)
        self.multilabel_score_weight = float(multilabel_score_weight)
        self.count_score_weight = float(count_score_weight)

        if architecture == "residual":
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, base_channels, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
                ResidualBlock(base_channels, base_channels),
                ResidualBlock(base_channels, base_channels * 2, stride=2),
                ResidualBlock(base_channels * 2, base_channels * 2),
                ResidualBlock(base_channels * 2, base_channels * 4, stride=2),
                ResidualBlock(base_channels * 4, base_channels * 4),
                # Preserve coarse frequency location while aggregating time.
                # Full 1x1 pooling makes spectral-source recognition nearly
                # translation invariant along the frequency axis.
                nn.AdaptiveAvgPool2d((8, 1)),
            )
            embedding_dim = base_channels * 4 * 8
        else:
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            embedding_dim = 64

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.classifier = nn.Linear(embedding_dim, num_classes)
        if self.auxiliary_heads:
            self.combo_classifier = nn.Linear(embedding_dim, (2**num_classes) - 1)
            self.count_classifier = nn.Linear(embedding_dim, num_classes)
            combo_labels = []
            for value in range(1, 2**num_classes):
                combo_labels.append(
                    [(value >> (num_classes - 1 - index)) & 1 for index in range(num_classes)]
                )
            self.register_buffer(
                "combo_labels",
                torch.tensor(combo_labels, dtype=torch.float32),
                persistent=False,
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.dropout(torch.flatten(x, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))

    def forward_with_auxiliary(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.auxiliary_heads:
            raise RuntimeError("Auxiliary heads are not enabled for this model")
        embedding = self.encode(x)
        return (
            self.classifier(embedding),
            self.combo_classifier(embedding),
            self.count_classifier(embedding),
        )

    def probabilities_from_outputs(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Return genuine per-label probabilities for every prediction mode.

        Structured mode returns label marginals derived from the seven-way
        combination distribution. Final structured decisions must use
        ``decoded_labels_from_outputs`` instead of thresholding this tensor.
        """
        if self.prediction_mode == "structured":
            return self.label_marginal_probabilities_from_outputs(outputs)
        return torch.sigmoid(outputs[0])

    def multilabel_probabilities_from_outputs(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        return torch.sigmoid(outputs[0])

    def decoded_labels_from_outputs(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if self.prediction_mode == "structured":
            return self.structured_predictions(outputs)
        raise RuntimeError(
            "Multilabel decoding requires explicit thresholds and is handled by the runtime"
        )

    def structured_predictions(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        scores = self.structured_scores(outputs)
        labels = self.combo_labels.to(device=scores.device, dtype=scores.dtype)
        return labels[scores.argmax(dim=1)]

    def structured_scores(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if not self.auxiliary_heads:
            raise RuntimeError("Structured prediction requires auxiliary heads")
        multilabel_logits, combo_logits, count_logits = outputs
        labels = self.combo_labels.to(device=multilabel_logits.device, dtype=multilabel_logits.dtype)

        combo_scores = F.log_softmax(combo_logits, dim=1)
        positive_log_probs = F.logsigmoid(multilabel_logits).unsqueeze(1)
        negative_log_probs = F.logsigmoid(-multilabel_logits).unsqueeze(1)
        label_matrix = labels.unsqueeze(0)
        multilabel_scores = (
            label_matrix * positive_log_probs + (1.0 - label_matrix) * negative_log_probs
        ).mean(dim=2)
        count_log_probs = F.log_softmax(count_logits, dim=1)
        count_indices = labels.sum(dim=1).long() - 1
        count_scores = count_log_probs[:, count_indices]

        return (
            self.combo_score_weight * combo_scores
            + self.multilabel_score_weight * multilabel_scores
            + self.count_score_weight * count_scores
        )

    def structured_combo_probabilities(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        return F.softmax(self.structured_scores(outputs), dim=1)

    def label_marginal_probabilities_from_outputs(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if not self.auxiliary_heads:
            raise RuntimeError("Structured label marginals require auxiliary heads")
        combination_probabilities = self.structured_combo_probabilities(outputs)
        labels = self.combo_labels.to(
            device=combination_probabilities.device,
            dtype=combination_probabilities.dtype,
        )
        return combination_probabilities @ labels


def build_model(num_classes: int, config: dict) -> NoiseCNN:
    model_config = config.get("model", {})
    auxiliary_config = model_config.get("auxiliary_heads", {})
    prediction_config = model_config.get("prediction", {})
    input_representation = str(config.get("stft", {}).get("input_representation", "single"))
    input_channels_by_representation = {
        "single": 1,
        "absolute": 1,
        "legacy": 1,
        "absolute_relative": 2,
        "db_trace": 4,
    }
    normalized_representation = input_representation.strip().lower()
    if normalized_representation not in input_channels_by_representation:
        raise ValueError(f"Unsupported stft.input_representation: {input_representation}")
    input_channels = input_channels_by_representation[normalized_representation]
    return NoiseCNN(
        num_classes=num_classes,
        auxiliary_heads=bool(auxiliary_config.get("enabled", False)),
        input_channels=input_channels,
        architecture=str(model_config.get("architecture", "lightweight")),
        base_channels=int(model_config.get("base_channels", 32)),
        dropout=float(model_config.get("dropout", 0.0)),
        prediction_mode=str(prediction_config.get("mode", "multilabel")),
        combo_score_weight=float(prediction_config.get("combo_score_weight", 1.0)),
        multilabel_score_weight=float(prediction_config.get("multilabel_score_weight", 0.3)),
        count_score_weight=float(prediction_config.get("count_score_weight", 0.2)),
    )
