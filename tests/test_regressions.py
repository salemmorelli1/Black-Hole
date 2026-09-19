"""Regression tests for audited numerical and artifact contracts."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from black_hole_information.artifacts import write_json_atomic
from black_hole_information.cli import load_experiment_config
from black_hole_information.engine import (
    CentralizedInformationEngine,
    PhysicsConfig,
    PlanckNumberMarks,
    diffusion_information,
    empirical_fisher,
    point_process_information,
)

torch.set_default_dtype(torch.float64)


class ValidationRegressionTests(unittest.TestCase):
    def test_config_rejects_nonfinite_and_degenerate_values(self) -> None:
        invalid = (
            {"horizon": math.nan},
            {"dt": math.inf},
            {"eps": 0.0},
            {"max_events": 1.5},
            {"horizon": 0.1, "dt": 1.0},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                PhysicsConfig(**changes).validate()

    def test_config_loader_rejects_nonfinite_theta(self) -> None:
        payload = {
            "horizon": 0.2,
            "dt": 0.01,
            "m_floor": 0.25,
            "theta": {"M0": 4.0, "gamma": math.nan, "alpha": 0.2},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "theta values must be finite"):
                load_experiment_config(path)

    def test_engine_rejects_nonphysical_theta_before_simulation(self) -> None:
        config = PhysicsConfig(horizon=0.02, dt=0.01, planck_mixture_terms=32)
        theta = torch.tensor([4.0, -1.0, 0.4], requires_grad=True)
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            CentralizedInformationEngine(config).evaluate(theta, replicates=2, seed=1)
        theta = torch.tensor([4.0, 4.0, 0.4], requires_grad=True)
        with self.assertRaisesRegex(ValueError, "replicates must be an integer"):
            CentralizedInformationEngine(config).evaluate(theta, replicates=2.5, seed=1)

    def test_planck_distribution_rejects_invalid_support(self) -> None:
        with self.assertRaisesRegex(ValueError, "k_max"):
            PlanckNumberMarks(k_max=0)
        marks = PlanckNumberMarks(k_max=32)
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            marks.log_prob_dimensionless(torch.tensor(-0.1))
        values = torch.tensor([0.5, 1.0])
        self.assertEqual(tuple(marks.log_prob_dimensionless(values).shape), (2,))
        self.assertEqual(tuple(marks.log_survival_dimensionless(values).shape), (2,))


class InformationRegressionTests(unittest.TestCase):
    def test_information_functionals_fail_closed(self) -> None:
        full = torch.ones(2)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            point_process_information(full, -full, dt=0.1)
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            diffusion_information(full, torch.zeros(2), observation_sd=0.0, dt=0.1)

    def test_empirical_fisher_rejects_empty_or_nonfinite_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            empirical_fisher(torch.empty((0, 3)))
        scores = torch.tensor([[1.0, math.nan, 0.0]])
        with self.assertRaisesRegex(ValueError, "finite"):
            empirical_fisher(scores)

    def test_engine_integrates_over_intervals_not_terminal_grid_point(self) -> None:
        config = PhysicsConfig(
            horizon=0.02,
            dt=0.01,
            m_floor=0.25,
            sigma_base=0.02,
            max_events=200,
            planck_mixture_terms=32,
        )
        theta = torch.tensor([4.0, 4.0, 0.4], requires_grad=True)
        reduced = torch.ones((2, 2))
        result = CentralizedInformationEngine(config).evaluate(
            theta,
            replicates=2,
            seed=7,
            reduced_rate_a=reduced,
            reduced_signal_b=reduced,
        )
        self.assertTrue(torch.isfinite(result["delta_information_B_minus_A_nats"]))

        terminal_grid_shape = torch.ones((2, 3))
        theta = torch.tensor([4.0, 4.0, 0.4], requires_grad=True)
        with self.assertRaisesRegex(ValueError, r"shape \(2, 2\)"):
            CentralizedInformationEngine(config).evaluate(
                theta,
                replicates=2,
                seed=7,
                reduced_rate_a=terminal_grid_shape,
                reduced_signal_b=terminal_grid_shape,
            )


class ArtifactRegressionTests(unittest.TestCase):
    def test_atomic_json_rejects_nonfinite_values_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            original = '{"status": "stable"}\n'
            path.write_text(original, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Out of range float values"):
                write_json_atomic(path, {"value": math.nan})
            self.assertEqual(path.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
