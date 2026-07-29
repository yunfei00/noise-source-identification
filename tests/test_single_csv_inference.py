from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from src.dataset import RealCsvDataset
from src.features import (
    PreprocessConfig,
    fix_model_signal_length,
    load_csv_signal,
    preprocess_signal,
)
from src.inference.checkpoint import (
    inspect_checkpoint,
    load_checkpoint_artifact,
    load_state_dict_strict,
)
from src.inference.single_csv_predictor import (
    SingleCsvInferenceError,
    predict_single_csv,
    prepare_single_csv_input,
)
from src.model_cnn import build_model


def inference_config() -> dict:
    return {
        "data": {
            "class_names": ["source_1", "source_3", "source_5"],
            "sample_rate": 64,
            "signal_length": 32,
        },
        "preprocessing": {"signal_normalization": "standardize"},
        "stft": {
            "nperseg": 8,
            "noverlap": 4,
            "target_freq_bins": 16,
            "target_time_bins": 8,
            "magnitude_scale": "log1p",
            "input_representation": "single",
        },
        "model": {
            "architecture": "lightweight",
            "auxiliary_heads": {"enabled": False},
            "prediction": {"mode": "multilabel"},
        },
        "loss": {"type": "bce"},
        "train": {"threshold": 0.5},
    }


def csv_text(values: list[float], *, metadata: str = "instrument,analyzer\n") -> str:
    rows = "\n".join(f"{index},{value}" for index, value in enumerate(values))
    return f"{metadata}DATA\n{rows}\n"


class CsvSignalParserTest(unittest.TestCase):
    def test_data_section_after_metadata_uses_second_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            path.write_text(
                "instrument,model-x\nserial,123\nDaTa\n0,-80\n1,-79\n",
                encoding="utf-8",
            )

            parsed = load_csv_signal(path)

            self.assertTrue(parsed.found_data_line)
            self.assertEqual(parsed.data_marker_line, 3)
            self.assertEqual(parsed.data_start_line, 4)
            self.assertEqual(parsed.selected_columns, [1])
            self.assertEqual(len(parsed.metadata_rows), 2)
            np.testing.assert_allclose(parsed.raw_signal, [-80.0, -79.0])

    def test_missing_data_marker_is_rejected_in_strict_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            path.write_text("time,value\n0,1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "未找到 DATA 数据段"):
                load_csv_signal(path)

    def test_empty_rows_after_data_are_skipped_and_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            path.write_text("DATA\n0,1\n\n1,2\n", encoding="utf-8")

            parsed = load_csv_signal(path)

            self.assertEqual(parsed.skipped_empty_rows, 1)
            np.testing.assert_allclose(parsed.raw_signal, [1.0, 2.0])

    def test_invalid_data_value_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            path.write_text("DATA\n0,1\n1,not-a-number\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "not numeric"):
                load_csv_signal(path)

    def test_nonfinite_data_value_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            path.write_text("DATA\n0,1\n1,nan\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "not finite"):
                load_csv_signal(path)

    def test_data_row_with_too_few_columns_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            path.write_text("DATA\n1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "column index 1 is required"):
                load_csv_signal(path)


class SharedPreprocessingTest(unittest.TestCase):
    def test_length_padding_and_center_crop_match_training_function(self) -> None:
        short = np.asarray([1.0, 2.0], dtype=np.float32)
        long = np.arange(10, dtype=np.float32)
        short_config = PreprocessConfig(
            signal_length=5,
            sample_rate=16,
            nperseg=4,
            noverlap=2,
            target_freq_bins=4,
            target_time_bins=2,
        )
        long_config = PreprocessConfig(
            signal_length=4,
            sample_rate=16,
            nperseg=4,
            noverlap=2,
            target_freq_bins=4,
            target_time_bins=2,
        )

        padded = preprocess_signal(short, short_config)
        cropped = preprocess_signal(long, long_config)

        np.testing.assert_allclose(
            padded.resized_signal,
            fix_model_signal_length(short, 5, "single"),
        )
        np.testing.assert_allclose(
            cropped.resized_signal,
            fix_model_signal_length(long, 4, "single"),
        )
        self.assertEqual(padded.statistics["length_method"], "right_pad")
        self.assertEqual(cropped.statistics["crop_start"], 3)

    def test_nan_inf_and_constant_standardization_are_deterministic(self) -> None:
        config = PreprocessConfig(
            signal_length=4,
            sample_rate=16,
            nperseg=4,
            noverlap=2,
            target_freq_bins=4,
            target_time_bins=2,
            signal_normalization="standardize",
        )

        nonfinite = preprocess_signal(
            np.asarray([1.0, np.nan, np.inf, -np.inf], dtype=np.float32),
            config,
        )
        constant = preprocess_signal(
            np.asarray([3.0, 3.0, 3.0, 3.0], dtype=np.float32),
            config,
        )

        self.assertTrue(torch.isfinite(nonfinite.input_tensor).all())
        np.testing.assert_allclose(constant.normalized_signal, np.zeros(4))
        self.assertEqual(constant.statistics["normalization_parameters"]["sample_std"], 0.0)

    def test_db_trace_does_not_invent_db_to_linear_conversion(self) -> None:
        config = PreprocessConfig(
            signal_length=4,
            sample_rate=16,
            nperseg=4,
            noverlap=2,
            target_freq_bins=4,
            target_time_bins=2,
            input_representation="db_trace",
            signal_normalization="none",
        )
        signal = np.asarray([-80.0, -79.0, -78.0, -77.0], dtype=np.float32)

        result = preprocess_signal(signal, config)

        self.assertIsNone(result.linear_signal)
        self.assertFalse(result.statistics["linear_conversion_applied"])
        np.testing.assert_allclose(result.normalized_signal, signal)

    def test_real_dataset_and_single_input_use_identical_tensor(self) -> None:
        config = inference_config()
        values = np.sin(np.arange(32, dtype=np.float32)).tolist()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "source_1" / "sample.csv"
            csv_path.parent.mkdir(parents=True)
            csv_path.write_text(csv_text(values), encoding="utf-8")
            dataset = RealCsvDataset(
                root,
                config["data"]["class_names"],
                config,
                augment=False,
            )

            dataset_tensor, _ = dataset[0]
            _, inference_preprocess = prepare_single_csv_input(csv_path, config)

            torch.testing.assert_close(
                dataset_tensor,
                inference_preprocess.input_tensor,
            )


class CheckpointContractTest(unittest.TestCase):
    def test_inspects_pure_and_wrapped_state_dicts(self) -> None:
        config = inference_config()
        model = build_model(3, config)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pure_path = root / "pure.pt"
            model_state_dict_path = root / "model_state_dict.pt"
            state_dict_path = root / "state_dict.pt"
            torch.save(model.state_dict(), pure_path)
            torch.save({"model_state_dict": model.state_dict()}, model_state_dict_path)
            torch.save({"state_dict": model.state_dict()}, state_dict_path)

            self.assertEqual(inspect_checkpoint(pure_path)["state_dict_key"], "<root>")
            self.assertEqual(
                inspect_checkpoint(model_state_dict_path)["state_dict_key"],
                "model_state_dict",
            )
            self.assertEqual(
                inspect_checkpoint(state_dict_path)["state_dict_key"],
                "state_dict",
            )

    def test_module_prefix_is_removed_and_loaded_strictly(self) -> None:
        config = inference_config()
        source_model = build_model(3, config)
        prefixed = {
            f"module.{key}": value
            for key, value in source_model.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prefixed.pt"
            torch.save({"state_dict": prefixed}, path)

            loaded = load_checkpoint_artifact(path)
            target_model = build_model(3, config)
            load_state_dict_strict(target_model, loaded)

            self.assertTrue(loaded.inspection["module_prefix_present"])
            self.assertTrue(loaded.inspection["module_prefix_removed"])

    def test_model_structure_mismatch_reports_keys_and_shapes(self) -> None:
        config = inference_config()
        source_model = build_model(3, config)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mismatch.pt"
            torch.save({"model_state": source_model.state_dict()}, path)
            loaded = load_checkpoint_artifact(path)
            mismatched_model = build_model(2, {**config, "data": {"class_names": ["a", "b"]}})

            with self.assertRaisesRegex(RuntimeError, "shape_mismatches"):
                load_state_dict_strict(mismatched_model, loaded)

    def test_missing_checkpoint_config_is_rejected_without_config_argument(self) -> None:
        config = inference_config()
        model = build_model(3, config)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "state.pt"
            csv_path = root / "sample.csv"
            torch.save(model.state_dict(), checkpoint)
            csv_path.write_text(csv_text([1.0, 2.0, 3.0]), encoding="utf-8")

            with self.assertRaisesRegex(SingleCsvInferenceError, "missing the model/preprocessing config"):
                predict_single_csv(
                    csv_path,
                    checkpoint,
                    device="cpu",
                    report_dir=root / "report",
                )


class EndToEndSingleCsvTest(unittest.TestCase):
    def test_dataset_logits_probabilities_and_binary_prediction_match(self) -> None:
        torch.manual_seed(7)
        config = inference_config()
        labels = config["data"]["class_names"]
        model = build_model(len(labels), config)
        model.eval()
        values = (np.cos(np.arange(32, dtype=np.float32) / 3.0) * 2.0).tolist()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "source_1" / "sample.csv"
            csv_path.parent.mkdir(parents=True)
            csv_path.write_text(csv_text(values), encoding="utf-8")
            checkpoint = root / "best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": labels,
                    "config": config,
                    "epoch": 4,
                    "best_metric": 0.75,
                },
                checkpoint,
            )
            dataset = RealCsvDataset(root, labels, config, augment=False)
            dataset_tensor, _ = dataset[0]
            expected_logits = model(dataset_tensor.unsqueeze(0)).squeeze(0)
            expected_probabilities = torch.sigmoid(expected_logits)
            expected_binary = (expected_probabilities >= 0.5).int()

            result = predict_single_csv(
                csv_path,
                checkpoint,
                device="cpu",
                report_dir=root / "contract",
            )

            torch.testing.assert_close(
                torch.tensor(result.logits),
                expected_logits,
            )
            torch.testing.assert_close(
                torch.tensor(result.probabilities),
                expected_probabilities,
            )
            self.assertEqual(result.binary_prediction, expected_binary.tolist())
            prediction_payload = json.loads(
                Path(result.prediction_json_path).read_text(encoding="utf-8")
            )
            self.assertEqual(prediction_payload["binary_label"], result.binary_label)
            report = Path(result.report_path).read_text(encoding="utf-8")
            for section in range(1, 12):
                self.assertIn(f"## {section}.", report)
            self.assertIn("strict=True", report)
            self.assertIn("src.features.load_csv_signal", report)

    def test_pure_state_dict_can_use_exact_external_config(self) -> None:
        config = inference_config()
        model = build_model(3, config)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "state.pt"
            config_path = root / "train.yaml"
            csv_path = root / "sample.csv"
            torch.save(model.state_dict(), checkpoint)
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )
            csv_path.write_text(
                csv_text(np.arange(32, dtype=np.float32).tolist()),
                encoding="utf-8",
            )

            result = predict_single_csv(
                csv_path,
                checkpoint,
                config_path=config_path,
                threshold=[0.2, 0.5, 0.8],
                device="cpu",
                report_dir=root / "contract",
            )

            self.assertEqual(result.labels, config["data"]["class_names"])
            self.assertEqual(result.thresholds, [0.2, 0.5, 0.8])

    def test_structured_checkpoint_uses_validation_decoder_and_records_auxiliary_logits(self) -> None:
        config = inference_config()
        config["model"] = {
            "architecture": "lightweight",
            "auxiliary_heads": {"enabled": True},
            "prediction": {"mode": "structured"},
        }
        labels = config["data"]["class_names"]
        model = build_model(3, config)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "structured.pt"
            csv_path = root / "sample.csv"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": labels,
                    "config": config,
                },
                checkpoint,
            )
            csv_path.write_text(
                csv_text(np.arange(32, dtype=np.float32).tolist()),
                encoding="utf-8",
            )

            result = predict_single_csv(
                csv_path,
                checkpoint,
                device="cpu",
                report_dir=root / "contract",
            )

            self.assertIsNotNone(result.auxiliary_logits)
            assert result.auxiliary_logits is not None
            self.assertEqual(len(result.auxiliary_logits["combination_logits"]), 7)
            self.assertEqual(len(result.auxiliary_logits["count_logits"]), 3)
            self.assertTrue(all(value in {0.0, 1.0} for value in result.probabilities))
            self.assertGreaterEqual(sum(result.binary_prediction), 1)
            report = Path(result.report_path).read_text(encoding="utf-8")
            self.assertIn("structured combination argmax", report)


if __name__ == "__main__":
    unittest.main()
