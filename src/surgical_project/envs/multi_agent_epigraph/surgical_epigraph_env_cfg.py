"""EPIGRAPH environment configuration aligned with the shared MARL task."""

from isaaclab.utils import configclass

from surgical_project.envs.multi_agent.surgical_direct_marl_env_cfg import (
    SurgicalDirectMARLEnvCfg,
)


@configclass
class SurgicalEpigraphEnvCfg(SurgicalDirectMARLEnvCfg):
    """Use exactly the same physics, action, and observation contract as RMAPPO."""

    pass
