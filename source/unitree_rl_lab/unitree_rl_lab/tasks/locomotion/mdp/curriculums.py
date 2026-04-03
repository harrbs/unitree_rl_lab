from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_mean_terrain_level(env) -> float:
    try:
        return env.scene.terrain.terrain_levels.float().mean().item()
    except Exception:
        return 0.0


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    if env.common_step_counter % env.max_episode_length == 0:
        # ascend 성공률 기반 속도 증가: flat/descend가 높아도 ascend가 낮으면 속도 고정
        ascend_success_rate = getattr(env, "ascend_success_rate", 0.0)
        mean_terrain_level  = _get_mean_terrain_level(env)
        # Let speed open much earlier once the policy shows any meaningful
        # ascending capability. The previous thresholds kept commands nearly
        # frozen at the initial low-speed regime.
        if ascend_success_rate > 0.4 and mean_terrain_level >= 0.05:
            delta_x = torch.tensor([-0.10, 0.10], device=env.device)
            delta_y = torch.tensor([-0.05, 0.05], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_x,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_y,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def terrain_levels_vel_smooth(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
) -> torch.Tensor:
    """Terrain curriculum with command-relative promotion/demotion thresholds.

    Default isaaclab ``terrain_levels_vel`` uses a fixed 4 m threshold for
    promotion.  At low command speeds (e.g. 0.1 m/s × 20 s = 2 m max travel)
    the robot can *never* reach 4 m, so terrain level stays at 0 indefinitely.

    This version uses relative thresholds:
      move_up   : dist > cmd * time * 0.2
      move_down : dist < cmd * time * 0.05

    This deliberately makes early promotion easier so the policy sees harder
    stair layouts before it collapses into a stable low-speed local optimum.
    """
    robot = env.scene["robot"]
    dist = torch.norm(
        robot.data.root_pos_w[env_ids, :2] - env.scene.terrain.env_origins[env_ids, :2],
        dim=1,
    )
    cmd = torch.norm(env.command_manager.get_command("base_velocity")[env_ids, :2], dim=1)
    t = env.max_episode_length_s

    # Exclude only near-zero commands. Early stair learning needs much easier
    # promotion than the previous curriculum provided.
    has_cmd   = cmd > 0.05
    move_up   = (dist > cmd * t * 0.4)  & has_cmd   # 0.2→0.4: min 0.8m to promote
    move_down = (dist < cmd * t * 0.05) & has_cmd

    env.scene.terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(env.scene.terrain.terrain_levels.float())


def terrain_levels_two_axis_descend(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    flat_max_col: int = 1,
    tiny_max_col: int = 4,
    medium_max_col: int = 7,
    hard_max_col: int = 9,
    tiny_first_col: int = 2,
    medium_first_col: int = 5,
    hard_first_col: int = 8,
    tiny_unlock_thresh: float = 0.45,
    medium_unlock_thresh: float = 0.50,
) -> torch.Tensor:
    """Two-axis terrain curriculum for descending stairs.

    Same column-type gating logic as terrain_levels_two_axis_ascend but
    stores attributes as descend_tiny_success_rate / descend_medium_success_rate
    so TensorBoard logging does not collide when both tasks run.

    Column layout assumed (num_cols=10):
      flat          cols 0-1
      descend_tiny  cols 2-4
      descend_medium cols 5-7
      descend_hard  cols 8-9
    """
    robot = env.scene["robot"]
    t = env.max_episode_length_s

    all_dist = torch.norm(
        robot.data.root_pos_w[:, :2] - env.scene.terrain.env_origins[:, :2], dim=1
    )
    all_cmd = torch.norm(env.command_manager.get_command("base_velocity")[:, :2], dim=1)
    has_cmd = all_cmd > 0.05
    tt = env.scene.terrain.terrain_types

    def _sr(col_lo: int, col_hi: int) -> float:
        mask = (tt >= col_lo) & (tt <= col_hi) & has_cmd
        if mask.sum() < 4:
            return 0.0
        return (all_dist[mask] > all_cmd[mask] * t * 0.25).float().mean().item()

    tiny_sr   = _sr(tiny_first_col,   tiny_max_col)
    medium_sr = _sr(medium_first_col, medium_max_col)

    if medium_sr >= medium_unlock_thresh:
        max_col = hard_max_col
    elif tiny_sr >= tiny_unlock_thresh:
        max_col = medium_max_col
    else:
        max_col = tiny_max_col

    env.descend_tiny_success_rate   = tiny_sr
    env.descend_medium_success_rate = medium_sr
    env.descend_max_accessible_col  = max_col

    dist_ids = all_dist[env_ids]
    cmd_ids  = all_cmd[env_ids]
    hc_ids   = cmd_ids > 0.05
    move_up   = (dist_ids > cmd_ids * t * 0.4)  & hc_ids
    move_down = (dist_ids < cmd_ids * t * 0.05) & hc_ids
    env.scene.terrain.update_env_origins(env_ids, move_up, move_down)

    n_ids = len(env_ids)
    env.scene.terrain.terrain_types[env_ids] = torch.randint(
        0, max_col + 1, (n_ids,), device=env.device
    )
    env.scene.terrain.env_origins[env_ids] = env.scene.terrain.terrain_origins[
        env.scene.terrain.terrain_levels[env_ids],
        env.scene.terrain.terrain_types[env_ids],
    ]

    return torch.mean(env.scene.terrain.terrain_levels.float())


def lin_vel_cmd_levels_descend(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
) -> torch.Tensor:
    """Speed curriculum gated on descend success rate (for StairDescend task)."""
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    if env.common_step_counter % env.max_episode_length == 0:
        descend_success_rate = getattr(env, "descend_success_rate", 0.0)
        mean_terrain_level   = _get_mean_terrain_level(env)
        if descend_success_rate > 0.4 and mean_terrain_level >= 0.05:
            delta_x = torch.tensor([-0.10, 0.10], device=env.device)
            delta_y = torch.tensor([-0.05, 0.05], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_x,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_y,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def terrain_levels_two_axis_ascend(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    flat_max_col: int = 1,
    tiny_max_col: int = 4,
    medium_max_col: int = 7,
    hard_max_col: int = 9,
    tiny_first_col: int = 2,
    medium_first_col: int = 5,
    hard_first_col: int = 8,
    tiny_unlock_thresh: float = 0.45,
    medium_unlock_thresh: float = 0.50,
) -> torch.Tensor:
    """Two-axis terrain curriculum for ascending stairs.

    Axis 1 (row): standard promote/demote based on distance traveled.
    Axis 2 (col): gate terrain difficulty category (flat→tiny→medium→hard)
      based on observed per-category success rates.

    Column layout assumed (num_cols=10):
      flat     cols 0-1  (flat_max_col=1)
      tiny     cols 2-4  (tiny_max_col=4)
      medium   cols 5-7  (medium_max_col=7)
      hard     cols 8-9  (hard_max_col=9)
    """
    robot = env.scene["robot"]
    t = env.max_episode_length_s

    # ── Compute per-category success rates over ALL envs ──────────────────
    all_dist = torch.norm(
        robot.data.root_pos_w[:, :2] - env.scene.terrain.env_origins[:, :2], dim=1
    )
    all_cmd = torch.norm(env.command_manager.get_command("base_velocity")[:, :2], dim=1)
    has_cmd = all_cmd > 0.05
    tt = env.scene.terrain.terrain_types  # (num_envs,)

    def _sr(col_lo: int, col_hi: int) -> float:
        mask = (tt >= col_lo) & (tt <= col_hi) & has_cmd
        if mask.sum() < 4:
            return 0.0
        return (all_dist[mask] > all_cmd[mask] * t * 0.25).float().mean().item()

    tiny_sr   = _sr(tiny_first_col,   tiny_max_col)
    medium_sr = _sr(medium_first_col, medium_max_col)

    # Determine currently unlocked max column
    if medium_sr >= medium_unlock_thresh:
        max_col = hard_max_col
    elif tiny_sr >= tiny_unlock_thresh:
        max_col = medium_max_col
    else:
        max_col = tiny_max_col

    # Store for external logging (train_stair_single.py)
    env.tiny_success_rate   = tiny_sr
    env.medium_success_rate = medium_sr
    env.max_accessible_col  = max_col

    # ── Row promotion/demotion for resetting envs ─────────────────────────
    dist_ids = all_dist[env_ids]
    cmd_ids  = all_cmd[env_ids]
    hc_ids   = cmd_ids > 0.05
    move_up   = (dist_ids > cmd_ids * t * 0.4)  & hc_ids
    move_down = (dist_ids < cmd_ids * t * 0.05) & hc_ids
    env.scene.terrain.update_env_origins(env_ids, move_up, move_down)

    # ── Reassign ALL resetting envs uniformly within [0, max_col] ─────────
    # Must reassign ALL (not just too_high) so that when max_col increases
    # (e.g. tiny→medium unlock), existing envs in 0-4 can actually reach 5-7.
    n_ids = len(env_ids)
    env.scene.terrain.terrain_types[env_ids] = torch.randint(
        0, max_col + 1, (n_ids,), device=env.device
    )
    env.scene.terrain.env_origins[env_ids] = env.scene.terrain.terrain_origins[
        env.scene.terrain.terrain_levels[env_ids],
        env.scene.terrain.terrain_types[env_ids],
    ]

    return torch.mean(env.scene.terrain.terrain_levels.float())


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
