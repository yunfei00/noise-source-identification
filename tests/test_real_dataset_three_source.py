from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.build_real_index import build_real_index
from src.dataset import parse_group_label
from src.evaluate import compute_metrics, compute_ratio_accuracy
from src.features import apply_signal_normalization, compute_stft_feature, prepare_stft_channels
from src.model_cnn import NoiseCNN
from src.search_thresholds import threshold_values
from src.split_real_dataset import split_rows
from src.train import AsymmetricBCEWithLogitsLoss, MultiTaskLoss


class ThreeSourceRealDatasetTest(unittest.TestCase):
    def test_parse_three_source_mix_label(self) -> None:
        label = parse_group_label("source_1_source_3_source_5_mix", ["source_1", "source_3", "source_5"])
        self.assertEqual(label.astype(int).tolist(), [1, 1, 1])

    def test_build_index_summarizes_three_source_ratios(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            single_dir = root / "data" / "single"
            combo_dir = root / "data" / "real_dataset"
            for source in ("source_1", "source_3", "source_5"):
                csv_path = single_dir / source / "600.000MHz" / "000001.csv"
                csv_path.parent.mkdir(parents=True)
                csv_path.write_text("0,0\n1,1\n", encoding="utf-8")

            for ratio in ("ratio_1_1_1", "ratio_1_2_1", "ratio_1_2_4", "ratio_4_2_1"):
                csv_path = combo_dir / "source_1_source_3_source_5_mix" / ratio / "600.000MHz" / "000001.csv"
                csv_path.parent.mkdir(parents=True)
                csv_path.write_text("0,0\n1,1\n", encoding="utf-8")

            output = root / "outputs" / "reports" / "real_dataset_index.csv"
            rows, summary = build_real_index(single_dir=single_dir, combo_dir=combo_dir, output=output)

            self.assertEqual(summary["class_names"], ["source_1", "source_3", "source_5"])
            self.assertEqual(summary["group_counts"]["source_1_source_3_source_5_mix"], 4)
            self.assertEqual(summary["label_counts"]["[1,1,1]"], 4)
            self.assertEqual(summary["ratio_counts"]["ratio_1_1_1"], 1)
            self.assertEqual(summary["ratio_counts"]["ratio_1_2_1"], 1)
            self.assertEqual(summary["ratio_counts"]["ratio_1_2_4"], 1)
            self.assertEqual(summary["ratio_counts"]["ratio_4_2_1"], 1)

            with output.open("r", encoding="utf-8", newline="") as handle:
                labels = {row["group"]: row["label"] for row in csv.DictReader(handle)}
            self.assertEqual(labels["source_1_source_3_source_5_mix"], "[1,1,1]")
            self.assertEqual(len(rows), 7)

    def test_split_keeps_each_three_source_ratio_in_all_splits_when_possible(self) -> None:
        rows: list[dict[str, str]] = []
        for ratio in ("ratio_1_1_1", "ratio_1_2_1", "ratio_1_2_4", "ratio_4_2_1"):
            for index in range(6):
                rows.append(
                    {
                        "file": f"data/real_dataset/source_1_source_3_source_5_mix/{ratio}/600.000MHz/{index:06d}.csv",
                        "source_root": "real_dataset",
                        "group": "source_1_source_3_source_5_mix",
                        "condition_path": f"{ratio}/600.000MHz",
                        "label": "[1,1,1]",
                    }
                )

        split = split_rows(rows, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
        for ratio in ("ratio_1_1_1", "ratio_1_2_1", "ratio_1_2_4", "ratio_4_2_1"):
            ratio_splits = {row["split"] for row in split if row["condition_path"].startswith(ratio)}
            self.assertEqual(ratio_splits, {"train", "val", "test"})


    def test_metrics_include_exact_match_and_over_prediction_rate(self) -> None:
        probs = np.asarray([[0.9, 0.8, 0.7], [0.9, 0.7, 0.6]], dtype=float)
        targets = np.asarray([[1, 1, 1], [1, 1, 0]], dtype=int)

        metrics = compute_metrics(probs, targets, ["source_1", "source_3", "source_5"], 0.5)

        self.assertEqual(metrics["overall"]["exact_match"], 0.5)
        self.assertEqual(metrics["overall"]["over_prediction_rate"], 0.5)
        self.assertEqual(metrics["overall"]["under_prediction_rate"], 0.0)

    def test_ratio_accuracy_can_extract_ratio_from_full_file_paths(self) -> None:
        targets = np.asarray([[1, 1, 1], [1, 1, 1]], dtype=int)
        preds = np.asarray([[1, 1, 1], [1, 1, 0]], dtype=int)
        paths = [
            "data/real_dataset/source_1_source_3_source_5_mix/radio_1_2_4/a.csv",
            "data/real_dataset/source_1_source_3_source_5_mix/radio_1_2_4/b.csv",
        ]

        result = compute_ratio_accuracy(preds, targets, paths)

        self.assertEqual(result["ratio_1_2_4"]["num_samples"], 2)
        self.assertEqual(result["ratio_1_2_4"]["exact_match_accuracy"], 0.5)

    def test_threshold_values_are_inclusive(self) -> None:
        self.assertEqual(threshold_values(0.3, 0.4, 0.05), [0.3, 0.35, 0.4])

    def test_absolute_stft_feature_is_not_log_compressed(self) -> None:
        signal = np.asarray([0.0, 2.0, 0.0, -2.0, 0.0, 2.0, 0.0, -2.0], dtype=np.float32)

        absolute = compute_stft_feature(
            signal,
            sample_rate=8,
            nperseg=4,
            noverlap=0,
            target_freq_bins=4,
            target_time_bins=2,
            magnitude_scale="absolute",
        )
        log_feature = compute_stft_feature(
            signal,
            sample_rate=8,
            nperseg=4,
            noverlap=0,
            target_freq_bins=4,
            target_time_bins=2,
            magnitude_scale="log1p",
        )

        self.assertGreater(float(absolute.max()), float(log_feature.max()))

    def test_none_signal_normalization_preserves_values(self) -> None:
        signal = np.asarray([1.0, -2.0, 3.0], dtype=np.float32)
        normalized = apply_signal_normalization(signal, "none")
        np.testing.assert_allclose(normalized, signal)

    def test_absolute_relative_channels_keep_amplitude_and_scale_invariant_shape(self) -> None:
        feature = np.asarray([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32)

        channels = prepare_stft_channels(feature, "absolute_relative")
        scaled_channels = prepare_stft_channels(feature * 10.0, "absolute_relative")

        self.assertEqual(channels.shape, (2, 2, 2))
        np.testing.assert_allclose(channels[0], feature)
        np.testing.assert_allclose(channels[1], scaled_channels[1], rtol=1e-6, atol=1e-6)

    def test_auxiliary_heads_produce_multilabel_combo_and_count_logits(self) -> None:
        model = NoiseCNN(num_classes=3, auxiliary_heads=True)
        logits, combo_logits, count_logits = model.forward_with_auxiliary(
            torch.randn(2, 1, 128, 64)
        )

        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(combo_logits.shape), (2, 7))
        self.assertEqual(tuple(count_logits.shape), (2, 3))

    def test_multitask_loss_accepts_all_seven_nonempty_combinations(self) -> None:
        targets = torch.tensor(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=torch.float32,
        )
        outputs = (
            torch.zeros(7, 3, requires_grad=True),
            torch.zeros(7, 7, requires_grad=True),
            torch.zeros(7, 3, requires_grad=True),
        )
        criterion = MultiTaskLoss(
            nn.BCEWithLogitsLoss(),
            multilabel_weight=0.3,
            combo_weight=1.0,
            count_weight=0.3,
        )

        loss = criterion(outputs, targets)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))

    def test_source_specific_fn_penalty_increases_positive_loss(self) -> None:
        logits = torch.tensor([[0.0, 0.0, -2.0]])
        targets = torch.tensor([[0.0, 0.0, 1.0]])
        baseline = AsymmetricBCEWithLogitsLoss(
            gamma_neg=0,
            gamma_pos=0,
            label_smoothing=0,
            class_fn_penalty=torch.ones(3),
        )
        penalized = AsymmetricBCEWithLogitsLoss(
            gamma_neg=0,
            gamma_pos=0,
            label_smoothing=0,
            class_fn_penalty=torch.tensor([1.0, 1.0, 2.0]),
        )

        self.assertGreater(float(penalized(logits, targets)), float(baseline(logits, targets)))

    def test_residual_model_preserves_expected_output_shapes(self) -> None:
        model = NoiseCNN(
            num_classes=3,
            auxiliary_heads=True,
            architecture="residual",
            base_channels=8,
            dropout=0.1,
            prediction_mode="structured",
        )

        outputs = model.forward_with_auxiliary(torch.randn(2, 1, 128, 64))

        self.assertEqual(tuple(outputs[0].shape), (2, 3))
        self.assertEqual(tuple(outputs[1].shape), (2, 7))
        self.assertEqual(tuple(outputs[2].shape), (2, 3))

    def test_structured_decoder_returns_one_valid_combination(self) -> None:
        model = NoiseCNN(num_classes=3, auxiliary_heads=True, prediction_mode="structured")
        multilabel_logits = torch.zeros(1, 3)
        combo_logits = torch.full((1, 7), -10.0)
        combo_logits[0, 5] = 10.0  # Binary value 110 maps to class index 5.
        count_logits = torch.tensor([[-10.0, 10.0, -10.0]])

        prediction = model.probabilities_from_outputs(
            (multilabel_logits, combo_logits, count_logits)
        )

        self.assertEqual(prediction.int().tolist(), [[1, 1, 0]])


if __name__ == "__main__":
    unittest.main()
