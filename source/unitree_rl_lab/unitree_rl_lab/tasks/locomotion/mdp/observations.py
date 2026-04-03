from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def _sensor_hits_in_base_frame(env: ManagerBasedRLEnv, sensor_cfg) -> torch.Tensor:
    """Convert ray hit positions from a ray-caster sensor into robot base frame."""
    from isaaclab.utils.math import quat_apply_inverse

    sensor = env.scene.sensors[sensor_cfg.name]
    robot = env.scene["robot"]

    hits_w = sensor.data.ray_hits_w.clone().float()
    sensor_pos_w = sensor.data.pos_w.unsqueeze(1).expand_as(hits_w)
    invalid = ~torch.isfinite(hits_w).all(dim=-1, keepdim=True)
    hits_w = torch.where(invalid, sensor_pos_w, hits_w)

    n_envs, num_points, _ = hits_w.shape
    base_pos_w = robot.data.root_pos_w
    base_quat_w = robot.data.root_quat_w
    hits_rel = hits_w - base_pos_w.unsqueeze(1)
    hits_b = quat_apply_inverse(
        base_quat_w.unsqueeze(1).expand(-1, num_points, -1).reshape(n_envs * num_points, 4),
        hits_rel.reshape(n_envs * num_points, 3),
    ).reshape(n_envs, num_points, 3)
    return hits_b


def ray_hits_robot_frame(env: ManagerBasedRLEnv, sensor_cfg) -> torch.Tensor:
    """3D ray hit positions expressed in robot base frame.

    Uses the same RayCaster sensor as height_scan but returns full (x, y, z)
    coordinates instead of projected heights.  No manual grid construction is
    needed — the PointCloudEncoder learns how to extract terrain features
    directly from these 3-D positions.

    The function handles invalid rays (misses → inf/nan) by clamping them to
    the sensor origin, which produces a neutral "no obstacle" signal.

    Args:
        env: The Isaac Lab environment.
        sensor_cfg: SceneEntityCfg pointing to the RayCaster sensor.

    Returns:
        (N, num_rays * 3) flat tensor of hit positions in robot base frame.
        For the default 12×8 scanner this is (N, 288).
    """
    hits_b = _sensor_hits_in_base_frame(env, sensor_cfg)
    n_envs, num_points, _ = hits_b.shape
    return hits_b.reshape(n_envs, num_points * 3)


def blended_lidar_pointcloud(
    env: ManagerBasedRLEnv,
    sensor_cfg,
    ref_sensor_cfg,
) -> torch.Tensor:
    """Blend complete GT terrain observations with LiDAR scanner observations.

    Returns point-cloud observations in robot base frame:
      p_ref   : complete reference terrain points from the GT scanner
      p_lidar : direct raycast from the LiDAR scanner at the real mounting pose
      o_pc    : (1-a) p_ref + a p_lidar

    The final observation stays in robot base frame so deployment can feed
    real LiDAR points in the same representation.
    """
    alpha = float(getattr(env, "pc_blend_alpha", 0.0))
    p_ref = _sensor_hits_in_base_frame(env, ref_sensor_cfg)
    p_lidar = _sensor_hits_in_base_frame(env, sensor_cfg)
    n_envs, num_points, _ = p_ref.shape
    o_pc = (1.0 - alpha) * p_ref + alpha * p_lidar
    return o_pc.reshape(n_envs, num_points * 3)
