"""
Epigraph: Safe Multi-Agent Reinforcement Learning with Epigraph Form Constraints.

This package implements the Epigraph algorithm for safe MARL, featuring:
- Dual value functions (task Vl and safety Vh)
- Z-encoding for constraint parameterization
- Root finding for evaluation (Vh(o,z) - z = 0)
- PPO-based policy optimization with dual GAE
"""

from .epigraph_core import (
    ZEncoder,
    ActorRNN,
    CriticVlRNN,
    CriticVhRNN,
    RootFinder,
    TanhGaussian,
)

from .rollout_buffer_z import RolloutBufferZ

from .trainer import EpigraphTrainer


__all__ = [
    # Core networks
    "ZEncoder",
    "ActorRNN",
    "CriticVlRNN",
    "CriticVhRNN",
    "RootFinder",
    "TanhGaussian",
    
    # Buffer
    "RolloutBufferZ",
    
    # Trainer
    "EpigraphTrainer",
]

__version__ = "1.0.0"