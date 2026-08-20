"""Black Hole Information Dynamics research package."""

from .engine import (
    CentralizedInformationEngine,
    ContinuousMassDiffusion,
    DiffusionPath,
    JumpPath,
    PacketJumpDiffusion,
    PhysicsConfig,
    PlanckNumberMarks,
    common_parameter_pullback,
    diffusion_information,
    empirical_fisher,
    mean_drift_matching,
    point_process_information,
    positive_parameters,
    score_vector,
)

__version__ = "0.1.0"

__all__ = [
    "CentralizedInformationEngine",
    "ContinuousMassDiffusion",
    "DiffusionPath",
    "JumpPath",
    "PacketJumpDiffusion",
    "PhysicsConfig",
    "PlanckNumberMarks",
    "common_parameter_pullback",
    "diffusion_information",
    "empirical_fisher",
    "mean_drift_matching",
    "point_process_information",
    "positive_parameters",
    "score_vector",
]
