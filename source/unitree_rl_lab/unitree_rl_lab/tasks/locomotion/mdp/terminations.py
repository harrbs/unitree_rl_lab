from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def radial_distance_from_origin_exceeds(
    env: ManagerBasedRLEnv,
    maximum_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot drifts outside its assigned terrain tile."""
    asset = env.scene[asset_cfg.name]
    displacement = asset.data.root_pos_w[:, :2] - env.scene.terrain.env_origins[:, :2]
    return torch.linalg.norm(displacement, dim=1) > maximum_distance
