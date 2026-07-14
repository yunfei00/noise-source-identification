from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.build_real_index import build_real_index
from src.dataset import parse_group_label
from src.evaluate import compute_metrics
from src.features import apply_signal_normalization, compute_stft_feature
from src.search_thresholds import threshold_values
from src.split_real_dataset import split_rows


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


if __name__ == "__main__":
    unittest.main()
