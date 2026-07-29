from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from noise_source_runtime import InferenceSession
from noise_source_runtime.exceptions import ModelPackageError
from noise_source_runtime.package import (
    build_model_package,
    verify_model_package,
)
from noise_source_runtime.preprocessing import prepare_file_input
from src.dataset import RealCsvDataset
from src.model_cnn import build_model
from src.train import save_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def runtime_config(mode: str = "multilabel") -> dict:
    auxiliary = mode == "structured"
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
            "auxiliary_heads": {"enabled": auxiliary},
            "prediction": {"mode": mode},
        },
        "loss": {"type": "bce"},
        "train": {"threshold": 0.5},
    }


def write_data_csv(path: Path, offset: float = 0.0) -> None:
    values = np.sin(np.arange(32, dtype=np.float32) / 4.0) + offset
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "Instrument,runtime-test\n"
        "DATA\n"
        + "\n".join(f"{index},{value}" for index, value in enumerate(values))
        + "\n",
        encoding="utf-8",
    )


def write_checkpoint(path: Path, config: dict) -> torch.nn.Module:
    model = build_model(3, config)
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": config["data"]["class_names"],
            "config": config,
            "epoch": 2,
            "best_metric": 0.75,
        },
        path,
    )
    return model


class StructuredRuntimeSemanticsTest(unittest.TestCase):
    def test_structured_probabilities_marginals_and_argmax_are_distinct(self) -> None:
        torch.manual_seed(11)
        config = runtime_config("structured")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "structured.pt"
            csv_path = root / "sample.csv"
            write_checkpoint(checkpoint, config)
            write_data_csv(csv_path)

            with InferenceSession.load_model(
                checkpoint,
                device="cpu",
                threshold=[0.99, 0.99, 0.99],
            ) as session:
                result = session.predict_file(csv_path)

            self.assertEqual(result.decision_mode, "structured")
            self.assertFalse(result.thresholds_applicable)
            self.assertEqual(
                result.combination_labels,
                [
                    "001",
                    "010",
                    "011",
                    "100",
                    "101",
                    "110",
                    "111",
                ],
            )
            assert result.combination_probabilities is not None
            self.assertAlmostEqual(sum(result.combination_probabilities), 1.0, places=6)
            self.assertTrue(
                all(
                    0.0 <= value <= 1.0 for value in result.label_marginal_probabilities
                )
            )
            self.assertFalse(
                all(
                    value in {0.0, 1.0} for value in result.label_marginal_probabilities
                )
            )
            expected_index = int(np.argmax(result.combination_probabilities))
            expected_combination = result.combination_labels[expected_index]
            self.assertEqual(result.predicted_combination, expected_combination)
            self.assertEqual(
                result.decoded_label_vector,
                [int(value) for value in expected_combination],
            )
            np.testing.assert_allclose(
                result.multilabel_probabilities,
                torch.sigmoid(torch.tensor(result.multilabel_logits)).numpy(),
                rtol=1e-6,
                atol=1e-6,
            )
            self.assertNotEqual(
                result.label_marginal_probabilities,
                [float(value) for value in result.decoded_label_vector],
            )

    def test_multilabel_mode_uses_sigmoid_and_explicit_thresholds(self) -> None:
        torch.manual_seed(13)
        config = runtime_config("multilabel")
        thresholds = [0.0, 1.0, 0.5]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "multilabel.pt"
            csv_path = root / "sample.csv"
            write_checkpoint(checkpoint, config)
            write_data_csv(csv_path)

            with InferenceSession.load_model(
                checkpoint,
                device="cpu",
                threshold=thresholds,
            ) as session:
                result = session.predict_file(csv_path)

            expected = [
                int(probability >= threshold)
                for probability, threshold in zip(
                    result.multilabel_probabilities,
                    thresholds,
                )
            ]
            self.assertEqual(result.decision_mode, "multilabel")
            self.assertTrue(result.thresholds_applicable)
            self.assertIsNone(result.combination_probabilities)
            self.assertEqual(
                result.label_marginal_probabilities, result.multilabel_probabilities
            )
            self.assertEqual(result.decoded_label_vector, expected)


class InferenceSessionLifecycleTest(unittest.TestCase):
    def test_multiple_predictions_reuse_one_loaded_model_and_do_not_write(self) -> None:
        config = runtime_config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.pt"
            first_csv = root / "first.csv"
            second_csv = root / "second.csv"
            write_checkpoint(checkpoint, config)
            write_data_csv(first_csv, 0.0)
            write_data_csv(second_csv, 0.5)

            session = InferenceSession.load_model(checkpoint, device="cpu")
            model_identity = id(session.model)
            first = session.predict_file(first_csv)
            second = session.predict_file(second_csv)

            self.assertEqual(session.model_load_count, 1)
            self.assertEqual(id(session.model), model_identity)
            self.assertIsNone(first.prediction_json_path)
            self.assertIsNone(first.report_path)
            self.assertIsNone(second.prediction_json_path)
            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                ["first.csv", "model.pt", "second.csv"],
            )
            session.close()

    def test_dataset_and_runtime_prepare_identical_tensor(self) -> None:
        config = runtime_config()
        labels = config["data"]["class_names"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "source_1" / "sample.csv"
            write_data_csv(csv_path)
            dataset = RealCsvDataset(root, labels, config, augment=False)

            dataset_tensor, _ = dataset[0]
            _, runtime_processed = prepare_file_input(csv_path, config)

            torch.testing.assert_close(
                dataset_tensor,
                runtime_processed.input_tensor,
            )

    def test_predict_array_uses_same_session_without_file_contract(self) -> None:
        config = runtime_config()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model.pt"
            write_checkpoint(checkpoint, config)
            signal = np.arange(32, dtype=np.float32)

            with InferenceSession.load_model(checkpoint, device="cpu") as session:
                result = session.predict_array(signal, source_name="memory-buffer")

            self.assertEqual(result.csv_path, "memory-buffer")
            self.assertIsNone(result.csv_contract)


class RuntimePackagingAndCompatibilityTest(unittest.TestCase):
    def test_model_package_sha256_verification_and_tamper_detection(self) -> None:
        config = runtime_config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "best.pt"
            package_dir = root / "package"
            write_checkpoint(checkpoint, config)

            built = build_model_package(
                checkpoint,
                "1.2.3",
                output_dir=package_dir,
                model_name="noise-source-test",
            )
            verified = verify_model_package(package_dir)

            self.assertTrue(built["verification"]["valid"])
            self.assertTrue(verified["valid"])
            manifest = json.loads(
                (package_dir / "manifest.json").read_text(encoding="utf-8")
            )
            for key in (
                "package_schema_version",
                "model_name",
                "model_version",
                "runtime_version",
                "framework",
                "task_type",
                "prediction_mode",
                "labels",
                "input_contract",
                "checkpoint_sha256",
                "training_git_commit",
                "created_at",
                "best_epoch",
                "best_metric_name",
                "best_metric_value",
            ):
                self.assertIn(key, manifest)

            (package_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ModelPackageError, "SHA256 mismatch"):
                verify_model_package(package_dir)

    def test_legacy_checkpoint_loads_and_new_checkpoint_has_runtime_metadata(
        self,
    ) -> None:
        config = runtime_config()
        labels = config["data"]["class_names"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy_checkpoint = root / "legacy.pt"
            new_checkpoint = root / "new.pt"
            csv_path = root / "sample.csv"
            model = build_model(3, config)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": labels,
                    "config": config,
                },
                legacy_checkpoint,
            )
            write_data_csv(csv_path)

            with InferenceSession.load_model(
                legacy_checkpoint,
                device="cpu",
            ) as session:
                result = session.predict_file(csv_path)
            self.assertEqual(result.labels, labels)

            save_checkpoint(
                new_checkpoint,
                model,
                labels,
                config,
                epoch=3,
                best_metric=0.8,
                monitor_name="exact_match",
            )
            payload = torch.load(
                new_checkpoint,
                map_location="cpu",
                weights_only=True,
            )
            for key in (
                "checkpoint_schema_version",
                "created_at",
                "prediction_mode",
                "preprocessing_contract",
                "runtime_version",
                "training_git_commit",
                "monitor_name",
                "best_metric",
                "class_names",
            ):
                self.assertIn(key, payload)
            self.assertNotIn("model", payload)

    def test_cli_and_python_api_results_are_identical(self) -> None:
        config = runtime_config("structured")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.pt"
            csv_path = root / "sample.csv"
            output_dir = root / "cli-output"
            write_checkpoint(checkpoint, config)
            write_data_csv(csv_path)

            with InferenceSession.load_model(checkpoint, device="cpu") as session:
                api_result = session.predict_file(csv_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "predict_single_csv.py"),
                    "--csv",
                    str(csv_path),
                    "--checkpoint",
                    str(checkpoint),
                    "--device",
                    "cpu",
                    "--report-dir",
                    str(output_dir),
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0)
            cli_payload = json.loads(
                (output_dir / "sample_prediction.json").read_text(encoding="utf-8")
            )
            api_payload = api_result.to_dict()
            for key in (
                "decision_mode",
                "multilabel_logits",
                "multilabel_probabilities",
                "combination_labels",
                "combination_probabilities",
                "label_marginal_probabilities",
                "decoded_label_vector",
                "predicted_combination",
                "predicted_sources",
                "thresholds_applicable",
            ):
                self.assertEqual(cli_payload[key], api_payload[key])

    def test_runtime_package_does_not_import_training_module(self) -> None:
        runtime_root = PROJECT_ROOT / "src" / "noise_source_runtime"
        for path in runtime_root.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("src.train", source, path.name)


if __name__ == "__main__":
    unittest.main()
