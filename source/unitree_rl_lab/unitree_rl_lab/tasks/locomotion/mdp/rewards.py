from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def roll_pitch_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pitch_weight: float = 0.25,
    roll_weight: float = 1.0,
) -> torch.Tensor:
    """Penalize roll much more strongly than pitch using projected gravity.

    In Isaac Lab's body frame convention for legged robots:
      - projected_gravity_b[:, 0] is dominated by pitch deviation
      - projected_gravity_b[:, 1] is dominated by roll deviation

    Stair climbing requires some fore-aft body pitch, so this term keeps pitch
    softly regularized while strongly suppressing side-to-side roll.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    pitch_err = torch.square(asset.data.projected_gravity_b[:, 0])
    roll_err = torch.square(asset.data.projected_gravity_b[:, 1])
    return pitch_weight * pitch_err + roll_weight * roll_err


def roll_pitch_ang_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pitch_weight: float = 0.25,
    roll_weight: float = 1.0,
) -> torch.Tensor:
    """Penalize roll rate much more strongly than pitch rate."""
    asset: RigidObject = env.scene[asset_cfg.name]
    roll_rate = torch.square(asset.data.root_ang_vel_b[:, 0])
    pitch_rate = torch.square(asset.data.root_ang_vel_b[:, 1])
    return roll_weight * roll_rate + pitch_weight * pitch_rate


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def foot_clearance_reward_terrain_rel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_clearance: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward swinging feet for clearing terrain by target_clearance metres.

    Unlike foot_clearance_reward, this uses terrain-relative clearance estimated
    from the height scanner, so the signal is meaningful on ascending/descending
    stairs where absolute world z is misleading.

    The height scanner (top-down, z=20m) provides terrain surface z at each ray
    hit.  For each foot we find the nearest ray hit in the xy plane and compute:
        clearance = foot_z_world - terrain_z_below_foot

    Reward fires only on swinging feet (foot moving horizontally), same as the
    original foot_clearance_reward.
    """
    from isaaclab.sensors import RayCaster

    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    # Foot world positions/velocities: (N, num_feet, 3)
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]

    # Ray hit positions from height scanner: (N, P, 3)
    ray_hits = sensor.data.ray_hits_w.clone()

    # Replace invalid (inf/nan) hits with sensor position → neutral signal
    sensor_pos = sensor.data.pos_w.unsqueeze(1).expand_as(ray_hits)
    invalid = ~torch.isfinite(ray_hits).all(dim=-1, keepdim=True)
    ray_hits = torch.where(invalid, sensor_pos, ray_hits)

    # Nearest ray hit (xy plane) for each foot
    # foot_xy: (N, num_feet, 1, 2)   ray_xy: (N, 1, P, 2)
    foot_xy = foot_pos_w[:, :, :2].unsqueeze(2)
    ray_xy  = ray_hits[:, :, :2].unsqueeze(1)
    dist_sq = ((foot_xy - ray_xy) ** 2).sum(dim=-1)  # (N, num_feet, P)
    nearest = dist_sq.argmin(dim=-1)                  # (N, num_feet)

    N, num_feet = nearest.shape
    env_idx   = torch.arange(N, device=env.device).unsqueeze(1).expand_as(nearest)
    terrain_z = ray_hits[env_idx, nearest, 2]         # (N, num_feet)

    clearance     = foot_pos_w[:, :, 2] - terrain_z   # (N, num_feet)
    clearance_err = torch.square(clearance - target_clearance)

    foot_vel_tanh = torch.tanh(tanh_mult * torch.norm(foot_vel_w[:, :, :2], dim=-1))
    reward = clearance_err * foot_vel_tanh
    return torch.exp(-reward.sum(dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Stair rewards.
"""


def _terrain_type_masks(
    env: ManagerBasedRLEnv,
    flat_cols: int,
    ascend_cols: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return flat / ascend / descend masks from terrain column ids."""
    tt = env.scene.terrain.terrain_types.to(env.device)
    flat_mask = tt < flat_cols
    ascend_mask = (tt >= flat_cols) & (tt < flat_cols + ascend_cols)
    descend_mask = tt >= flat_cols + ascend_cols
    return flat_mask, ascend_mask, descend_mask


def ascend_forward_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    flat_cols: int = 2,
    ascend_cols: int = 9,
    max_forward_speed: float = 0.8,
    max_upward_speed: float = 0.35,
) -> torch.Tensor:
    """Reward active forward-and-upward progress on ascending stair terrains.

    This term is intentionally task-specific: it only fires on ascending
    terrains and only when there is a meaningful planar command. The reward is
    strongest when the robot moves forward while also generating positive
    upward motion, which is the behavior missing in plateaued policies.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    _, ascend_mask, _ = _terrain_type_masks(env, flat_cols=flat_cols, ascend_cols=ascend_cols)
    cmd_xy = torch.linalg.norm(env.command_manager.get_command("base_velocity")[:, :2], dim=1)
    cmd_active = cmd_xy > 0.05

    forward_speed = torch.clamp(asset.data.root_lin_vel_b[:, 0], min=0.0, max=max_forward_speed)
    upward_speed = torch.clamp(asset.data.root_lin_vel_w[:, 2], min=0.0, max=max_upward_speed)
    upward_scale = upward_speed / max(max_upward_speed, 1e-6)

    return ascend_mask.float() * cmd_active.float() * forward_speed * (1.0 + upward_scale)


def descend_stable_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    flat_cols: int = 2,
    ascend_cols: int = 9,
    max_forward_speed: float = 0.8,
    orientation_scale: float = 5.0,
    ang_vel_scale: float = 0.75,
    pitch_weight: float = 0.6,
    roll_weight: float = 1.0,
    pitch_rate_weight: float = 0.75,
    roll_rate_weight: float = 1.0,
) -> torch.Tensor:
    """Reward controlled forward progress on descending stair terrains.

    Descending failures often look like forward face-plants. This term keeps
    rewarding forward motion, but discounts it when body tilt or roll/pitch
    rate grow too large, biasing the policy toward composed, deliberate
    descents instead of diving down the staircase.

    pitch_weight controls how strongly forward body lean is penalised during
    descent.  Set it low (e.g. 0.08) to allow the natural forward lean that
    occurs when stepping down stairs.  roll_weight should stay high to prevent
    sideways toppling.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    _, _, descend_mask = _terrain_type_masks(env, flat_cols=flat_cols, ascend_cols=ascend_cols)
    cmd_xy = torch.linalg.norm(env.command_manager.get_command("base_velocity")[:, :2], dim=1)
    cmd_active = cmd_xy > 0.05

    forward_speed = torch.clamp(asset.data.root_lin_vel_b[:, 0], min=0.0, max=max_forward_speed)
    pitch_err = torch.square(asset.data.projected_gravity_b[:, 0])
    roll_err = torch.square(asset.data.projected_gravity_b[:, 1])
    roll_rate = torch.square(asset.data.root_ang_vel_b[:, 0])
    pitch_rate = torch.square(asset.data.root_ang_vel_b[:, 1])

    stability = torch.exp(
        -orientation_scale * (pitch_weight * pitch_err + roll_weight * roll_err)
        -ang_vel_scale * (pitch_rate_weight * pitch_rate + roll_rate_weight * roll_rate)
    )
    return descend_mask.float() * cmd_active.float() * forward_speed * stability


def lin_vel_z_up(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize upward base linear velocity (positive vz only).

    Unlike lin_vel_z_l2 which penalises all z motion, this only fires when
    the robot moves upward.  Useful for descend tasks where downward vz is
    natural and should not be penalised.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.clamp(asset.data.root_lin_vel_w[:, 2], min=0.0) ** 2


def lin_vel_z_down(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize downward base linear velocity (negative vz only).

    Useful for ascend tasks where upward vz is natural and should not be
    penalised, but downward vz (robot descending after reaching the top)
    should be suppressed.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.clamp(-asset.data.root_lin_vel_w[:, 2], min=0.0) ** 2


def stair_height_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward upward body velocity — encourages active stair climbing.

    Only fires when the robot has a non-zero forward command so it does not
    interfere with flat-ground or standing behaviour.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    vz = asset.data.root_lin_vel_w[:, 2]
    cmd_norm = torch.linalg.norm(env.command_manager.get_command("base_velocity")[:, :2], dim=1)
    return torch.clamp(vz, min=0.0) * (cmd_norm > 0.1)


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward
