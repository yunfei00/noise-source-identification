from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.audit_real_data import audit_real_data, compute_db_statistics
from src.build_real_index import build_real_index
from src.dataset import augment_real_stft_feature, parse_group_label
from src.evaluate import compute_metrics, compute_ratio_accuracy
from src.features import (
    apply_signal_normalization,
    compute_stft_feature,
    fix_model_signal_length,
    prepare_db_trace_channels,
    prepare_stft_channels,
)
from src.infer_metadata_folder import infer_metadata_folder
from src.model_cnn import NoiseCNN, build_model
from src.search_thresholds import threshold_values
from src.split_real_dataset import split_rows
from src.template_ensemble import (
    combo_label_matrix,
    labels_to_combo_indices,
    nnls_features,
    search_blend_weight,
)
from src.train import AsymmetricBCEWithLogitsLoss, MultiTaskLoss


class ThreeSourceRealDatasetTest(unittest.TestCase):
    def test_metadata_folder_inference_recurses_and_reports_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "metadata"
            class_names = ["source_1", "source_3", "source_4"]
            config = {
                "data": {
                    "sample_rate": 64,
                    "signal_length": 64,
                },
                "preprocessing": {"signal_normalization": "none"},
                "stft": {
                    "nperseg": 16,
                    "noverlap": 8,
                    "target_freq_bins": 16,
                    "target_time_bins": 8,
                    "magnitude_scale": "absolute",
                    "input_representation": "db_trace",
                    "db_level_range": [-110.0, -50.0],
                    "db_variation_scale": 15.0,
                },
                "model": {
                    "architecture": "lightweight",
                    "auxiliary_heads": {"enabled": True},
                    "prediction": {"mode": "structured"},
                },
            }
            model = build_model(num_classes=3, config=config)
            checkpoint = root / "best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "config": config,
                },
                checkpoint,
            )

            time = np.arange(64, dtype=np.float32)
            for relative, offset in (
                (Path("batch_a") / "nested" / "a.csv", 0.0),
                (Path("root_sample.csv"), 3.0),
            ):
                path = input_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                values = -80.0 + offset + 2.0 * np.sin(2.0 * np.pi * time / 8.0)
                path.write_text(
                    "time,db\n"
                    + "\n".join(f"{index},{value:.5f}" for index, value in enumerate(values))
                    + "\n",
                    encoding="utf-8",
                )
            bad_path = input_dir / "batch_b" / "bad.csv"
            bad_path.parent.mkdir(parents=True)
            bad_path.write_text("time,db\nnot,data\n", encoding="utf-8")

            output = root / "predictions.csv"
            summary = infer_metadata_folder(
                model_path=checkpoint,
                input_dir=input_dir,
                output=output,
                device_name="cpu",
                batch_size=2,
                progress_every=0,
            )

            self.assertTrue(output.exists())
            self.assertTrue(output.with_suffix(".summary.json").exists())
            self.assertEqual(summary["discovered_csv_files"], 3)
            self.assertEqual(summary["processed_files"], 2)
            self.assertEqual(summary["failed_files"], 1)
            self.assertEqual(len(summary["combination_distribution"]), 7)
            self.assertEqual(
                sum(
                    int(metrics["count"])
                    for metrics in summary["combination_distribution"].values()
                ),
                2,
            )
            self.assertEqual(set(summary["source_distribution"]), set(class_names))
            self.assertIn("batch_a", summary["by_top_level_folder"])
            self.assertIn("(root)", summary["by_top_level_folder"])

    def test_db_audit_writes_counts_ranges_recommendations_and_issues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            single_dir = root / "data" / "single"
            combo_dir = root / "data" / "real_dataset"
            normal_rows = "time,db\n0,-80\n1,-78\n2,-76\n3,-79\n"
            for source in ("source_1", "source_3", "source_4"):
                path = single_dir / source / "600.000MHz" / "000001.csv"
                path.parent.mkdir(parents=True)
                path.write_text(normal_rows, encoding="utf-8")
            nonfinite_path = single_dir / "source_1" / "600.000MHz" / "000001.csv"
            nonfinite_path.write_text(normal_rows + "4,nan\n", encoding="utf-8")
            flat_path = single_dir / "source_3" / "600.000MHz" / "000002.csv"
            flat_path.write_text("time,db\n0,-95\n1,-95\n2,-95\n3,-95\n", encoding="utf-8")
            bad_path = single_dir / "source_4" / "600.000MHz" / "bad.csv"
            bad_path.write_text("time,db\na,b\n", encoding="utf-8")

            combo_path = (
                combo_dir
                / "source_1_source_3_mix"
                / "radio_1_2"
                / "600.000MHz"
                / "000001.csv"
            )
            combo_path.parent.mkdir(parents=True)
            combo_path.write_text(normal_rows, encoding="utf-8")
            output = root / "outputs" / "real_data_audit.json"

            report = audit_real_data(
                single_dir=single_dir,
                combo_dir=combo_dir,
                output=output,
                class_names=["source_1", "source_3", "source_4"],
                expected_length=4,
                no_signal_threshold_db=-90.0,
            )

            self.assertTrue(output.exists())
            self.assertEqual(report["class_names"], ["source_1", "source_3", "source_4"])
            self.assertEqual(report["counts"]["discovered_files"], 6)
            self.assertEqual(report["counts"]["parsed_files"], 5)
            self.assertEqual(report["counts"]["parse_failed_files"], 1)
            self.assertEqual(report["counts"]["ratio_counts"]["ratio_1_2"], 1)
            self.assertEqual(
                report["counts"]["group_ratio_counts"]["source_1_source_3_mix"]["ratio_1_2"],
                1,
            )
            self.assertEqual(report["signal_summary"]["observed_db_min"], -95.0)
            self.assertEqual(report["recommended_preprocessing"]["signal_length"], 4)
            self.assertIn("parse_failure", report["issues"]["counts_by_type"])
            self.assertIn("flat_or_nearly_flat", report["issues"]["counts_by_type"])
            self.assertIn("nonfinite_values", report["issues"]["counts_by_type"])
            self.assertIn("possible_no_signal", report["issues"]["counts_by_type"])

    def test_db_audit_jump_statistics(self) -> None:
        stats = compute_db_statistics(
            np.asarray([-80.0, -79.0, -60.0, -61.0], dtype=np.float32),
            jump_threshold_db=12.0,
        )

        self.assertEqual(stats["num_samples"], 4)
        self.assertEqual(stats["max_jump_db"], 19.0)
        self.assertEqual(stats["jump_count"], 1)

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

    def test_db_trace_channels_remove_baseline_but_preserve_level(self) -> None:
        time = np.arange(64, dtype=np.float32)
        signal = -77.0 + 4.0 * np.sin(2.0 * np.pi * time / 8.0)

        channels = prepare_db_trace_channels(
            signal,
            sample_rate=64,
            nperseg=16,
            noverlap=8,
            target_freq_bins=12,
            target_time_bins=8,
        )
        shifted = prepare_db_trace_channels(
            signal + 10.0,
            sample_rate=64,
            nperseg=16,
            noverlap=8,
            target_freq_bins=12,
            target_time_bins=8,
        )

        self.assertEqual(channels.shape, (4, 12, 8))
        np.testing.assert_allclose(channels[:2], shifted[:2], rtol=1e-5, atol=1e-5)
        self.assertGreater(float(shifted[2, 0, 0]), float(channels[2, 0, 0]))
        np.testing.assert_allclose(channels[3], shifted[3], rtol=1e-6, atol=1e-6)

    def test_db_trace_length_padding_uses_median_level(self) -> None:
        signal = np.asarray([-93.0, -77.0, -67.0], dtype=np.float32)

        fixed = fix_model_signal_length(signal, 6, "db_trace")

        np.testing.assert_allclose(fixed, [-93.0, -77.0, -67.0, -77.0, -77.0, -77.0])

    def test_db_trace_configuration_builds_four_channel_model(self) -> None:
        model = build_model(
            num_classes=3,
            config={
                "stft": {"input_representation": "db_trace"},
                "model": {"architecture": "residual", "base_channels": 8},
            },
        )

        self.assertEqual(model.features[0].in_channels, 4)
        self.assertEqual(tuple(model(torch.randn(2, 4, 128, 64)).shape), (2, 3))

    def test_real_stft_augmentation_preserves_shape_and_input(self) -> None:
        absolute = np.linspace(0.0, 1.0, 32, dtype=np.float32).reshape(8, 4)
        feature = prepare_stft_channels(absolute, "absolute_relative")
        original = feature.copy()
        config = {
            "gain_range": [0.8, 1.2],
            "noise_snr_db_range": [30, 40],
            "frequency_shift_bins": 1,
            "time_shift_bins": 1,
            "frequency_mask_probability": 1.0,
            "frequency_mask_width": 2,
            "time_mask_probability": 1.0,
            "time_mask_width": 1,
        }

        augmented = augment_real_stft_feature(
            feature,
            "absolute_relative",
            config,
            np.random.default_rng(42),
        )

        self.assertEqual(augmented.shape, feature.shape)
        self.assertTrue(np.all(augmented >= 0.0))
        np.testing.assert_array_equal(feature, original)
        self.assertFalse(np.array_equal(augmented, original))

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

    def test_nnls_features_recover_dominant_template_coefficients(self) -> None:
        templates = np.asarray(
            [
                [0.7, 0.2, 0.1, 0.0],
                [0.0, 0.1, 0.2, 0.7],
                [0.1, 0.2, 0.6, 0.1],
            ],
            dtype=float,
        )
        power = 0.8 * templates[0] + 0.2 * templates[2]

        features = nnls_features(power, templates)

        self.assertGreater(features[0], features[2])
        self.assertLess(features[1], 1e-5)

    def test_combo_indices_follow_binary_label_order(self) -> None:
        labels = combo_label_matrix(3)

        indices = labels_to_combo_indices(labels)

        np.testing.assert_array_equal(indices, np.arange(7))

    def test_blend_search_can_select_template_model(self) -> None:
        targets = np.asarray([0, 1, 2], dtype=np.int64)
        neural = np.full((3, 7), 0.01)
        neural[:, 6] = 0.94
        template = np.full((3, 7), 0.01)
        template[np.arange(3), targets] = 0.94

        alpha, accuracy = search_blend_weight(targets, neural, template, step=0.25)

        self.assertGreaterEqual(alpha, 0.5)
        self.assertEqual(accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
