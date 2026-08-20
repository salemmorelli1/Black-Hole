"""Numerical tests for the black-hole information engine."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from black_hole_information.artifacts import write_json_atomic
from black_hole_information.engine import (
    CentralizedInformationEngine,
    ContinuousMassDiffusion,
    PacketJumpDiffusion,
    PhysicsConfig,
    PlanckNumberMarks,
    mean_drift_matching,
)

torch.set_default_dtype(torch.float64)


class InformationEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PhysicsConfig(
            horizon=0.20,
            dt=0.01,
            m_floor=0.25,
            sigma_base=0.02,
            max_events=2000,
            planck_mixture_terms=512,
        )
        self.theta = torch.tensor([4.0, 4.0, 0.4], requires_grad=True)

    def test_planck_moments(self) -> None:
        marks = PlanckNumberMarks(k_max=4096)
        self.assertAlmostEqual(marks.moment(1, self.theta).item(), 2.701178, places=5)
        self.assertAlmostEqual(marks.moment(2, self.theta).item(), 10.35153, places=4)

    def test_matching_constants_are_positive(self) -> None:
        constants = mean_drift_matching(self.config, self.theta)
        for value in constants.values():
            self.assertTrue(torch.isfinite(value).all())
            self.assertGreater(float(value), 0.0)

    def test_jump_likelihood_and_fixed_data_score(self) -> None:
        generator = torch.Generator().manual_seed(11)
        model = PacketJumpDiffusion(self.config)
        path = model.simulate(self.theta, generator)
        ll = model.complete_path_loglik(path.detached(), self.theta, self.theta[0].detach())
        (score,) = torch.autograd.grad(ll, self.theta)
        self.assertTrue(torch.isfinite(ll))
        self.assertEqual(tuple(score.shape), (3,))
        self.assertEqual(float(score[2]), 0.0)

    def test_diffusion_likelihood_and_girsanov_score(self) -> None:
        generator = torch.Generator().manual_seed(12)
        model = ContinuousMassDiffusion(self.config)
        path = model.simulate(self.theta, generator)
        ll = model.girsanov_loglik(path.detached(), self.theta, self.theta[0].detach())
        (score,) = torch.autograd.grad(ll, self.theta)
        self.assertTrue(torch.isfinite(ll))
        self.assertEqual(tuple(score.shape), (3,))
        self.assertEqual(float(score[1]), 0.0)

    def test_central_engine_shapes_and_psd_fisher(self) -> None:
        result = CentralizedInformationEngine(self.config).evaluate(
            self.theta, replicates=16, seed=13
        )
        for key in ("fisher_a_native", "fisher_b_native"):
            matrix = result[key]
            self.assertEqual(tuple(matrix.shape), (3, 3))
            self.assertTrue(torch.isfinite(matrix).all())
            self.assertGreaterEqual(float(torch.linalg.eigvalsh(matrix).min()), -1.0e-10)
        self.assertEqual(tuple(result["delta_fisher_common_B_minus_A"].shape), (2, 2))
        self.assertTrue(torch.isfinite(result["delta_information_pathwise_gradient"]).all())

    def test_atomic_json_serializes_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            write_json_atomic(path, {"value": torch.tensor([1.0, 2.0])})
            with path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            self.assertEqual(payload["value"], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
