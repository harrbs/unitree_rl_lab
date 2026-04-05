"""Quantitative evaluation script for stair ascend / descend policies.

Collects per-terrain-type metrics over N episodes and prints a summary table.
Supports a --blind_pc flag that zeros the point-cloud observation at inference
time, so you can compare:

  Normal  : prop + PC  (Baseline E, full signal)
  Blind   : prop + zeros  (ablation — tests whether PC is actually used)
  PropOnly: load a Baseline-A checkpoint with --baseline A

Metrics collected
-----------------
  success_rate  : episode timed-out AND moved forward AND correct Δz
  fall_rate     : 1 - timeout_rate  (fell / got stuck)
  mean_energy   : mean |τ·ω| per step (lower = more efficient)
  stumble_rate  : fraction of steps with large horizontal foot contact forces

Usage
-----
  # Full PC (normal)
  python scripts/rsl_rl/eval_stair.py \\
      --task Unitree-Go2-StairAscend \\
      --checkpoint logs/rsl_rl/go2_stair_ascend/.../model_8700.pt \\
      --n_episodes 500

  # Blind ablation (zero PC)
  python scripts/rsl_rl/eval_stair.py \\
      --task Unitree-Go2-StairAscend \\
      --checkpoint logs/rsl_rl/go2_stair_ascend/.../model_8700.pt \\
      --n_episodes 500 --blind_pc

  # Prop-only baseline
  python scripts/rsl_rl/eval_stair.py \\
      --task Unitree-Go2-StairAscend \\
      --checkpoint logs/rsl_rl/go2_stair_A_prop_mlp/.../model_XXXX.pt \\
      --n_episodes 500 --baseline A
"""

import argparse
import pathlib
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401
sys.path.pop(0)

import cli_args  # isort: skip

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Quantitative stair policy evaluation.")
parser.add_argument("--task", type=str,
                    choices=["Unitree-Go2-StairAscend", "Unitree-Go2-StairDescend"],
                    default="Unitree-Go2-StairAscend")
parser.add_argument("--baseline", type=str, default="E",
                    choices=["A", "B", "C", "D", "E"],
                    help="Which agent config to load (A=prop-only, E=pointcloud).")
parser.add_argument("--blind_pc", action="store_true",
                    help="Zero-out point-cloud at inference (ablation).")
parser.add_argument("--n_episodes", type=int, default=500,
                    help="Total episodes to collect across all envs.")
parser.add_argument("--force_lin_vel_x", type=float, default=0.5,
                    help="Fixed forward command speed during eval (m/s).")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

_AGENT_KEY = {
    "A": "rsl_rl_baseline_A",
    "B": "rsl_rl_baseline_B",
    "C": "rsl_rl_baseline_C",
    "D": "rsl_rl_cfg_entry_point",
    "E": "rsl_rl_cfg_entry_point",
}
args_cli.agent = _AGENT_KEY[args_cli.baseline]

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ────────────────────────────────────────────────────────
import collections
import os

import torch

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from unitree_rl_lab.tasks.locomotion.robots.go2.policies import (
    GRUCNNActorCritic, PointCloudActorCritic, StairAwareActorCritic,
)
import rsl_rl.runners.on_policy_runner as _runner_module
_runner_module.StairAwareActorCritic = StairAwareActorCritic
_runner_module.GRUCNNActorCritic     = GRUCNNActorCritic
_runner_module.PointCloudActorCritic = PointCloudActorCritic

# ── Terrain column layout ──────────────────────────────────────────────────────
# Both stair tasks: cols 0-1 = flat, cols 2-4 = tiny, 5-7 = medium, 8-9 = hard
_TERRAIN_BINS = {
    "flat":   (0, 1),
    "tiny":   (2, 4),
    "medium": (5, 7),
    "hard":   (8, 9),
}


def _col_label(col: int) -> str:
    for label, (lo, hi) in _TERRAIN_BINS.items():
        if lo <= col <= hi:
            return label
    return "unknown"


# ── Metric accumulator ─────────────────────────────────────────────────────────

class TerrainMetrics:
    """Collects per-terrain-bin episode outcomes and per-step cost metrics."""

    def __init__(self):
        self.success  = collections.defaultdict(list)   # per bin, bool per episode
        self.timeout  = collections.defaultdict(list)   # timed-out (didn't fall)
        self.energy   = []                              # per step across all envs
        self.stumble  = []                              # per step, bool

    def record_episode(self, label: str, succeeded: bool, timed_out: bool):
        self.success[label].append(float(succeeded))
        self.timeout[label].append(float(timed_out))

    def record_step(self, energy: float, stumble: float):
        self.energy.append(energy)
        self.stumble.append(stumble)

    def total_episodes(self) -> int:
        return sum(len(v) for v in self.success.values())

    def print_table(self, label: str):
        bins = ["flat", "tiny", "medium", "hard"]
        header = f"{'Terrain':<10} {'Success%':>9} {'Fall%':>8} {'Episodes':>10}"
        sep = "-" * len(header)

        print(f"\n{'='*55}")
        print(f"  Mode : {label}")
        print(f"{'='*55}")
        print(header)
        print(sep)

        all_succ, all_ep = [], 0
        for b in bins:
            s = self.success[b]
            if not s:
                print(f"{b:<10} {'N/A':>9} {'N/A':>8} {'0':>10}")
                continue
            sr  = 100.0 * sum(s) / len(s)
            fr  = 100.0 * (1.0 - sum(self.timeout[b]) / len(self.timeout[b]))
            all_succ.extend(s); all_ep += len(s)
            print(f"{b:<10} {sr:>8.1f}% {fr:>7.1f}% {len(s):>10}")

        print(sep)
        if all_ep:
            sr = 100.0 * sum(all_succ) / all_ep
            print(f"{'ALL':<10} {sr:>8.1f}% {'':>8} {all_ep:>10}")

        if self.energy:
            print(f"\n  mean_energy  : {sum(self.energy)/len(self.energy):.4f}  (|τ·ω| mean)")
        if self.stumble:
            print(f"  stumble_rate : {100.*sum(self.stumble)/len(self.stumble):.2f}%  (horiz force > 4×vert)")
        print(f"{'='*55}\n")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_base_z(isaac_env) -> torch.Tensor:
    return isaac_env.scene["robot"].data.root_pos_w[:, 2].clone()


def _get_terrain_col(isaac_env, device) -> torch.Tensor:
    return isaac_env.scene.terrain.terrain_types.to(device)


def _compute_energy(isaac_env) -> float:
    robot = isaac_env.scene["robot"]
    return (robot.data.applied_torque.abs() * robot.data.joint_vel.abs()).mean().item()


def _compute_stumble(isaac_env, contact_sensor_name: str = "contact_forces") -> float:
    """Fraction of foot contacts where horizontal force > 4 × vertical force."""
    sensor = isaac_env.scene.sensors[contact_sensor_name]
    # body_ids for feet — sensor tracks all bodies; pick foot indices
    try:
        foot_ids = sensor.find_bodies(".*_foot")[0]
        forces = sensor.data.net_forces_w[:, foot_ids, :]   # (N, 4, 3)
        fz = torch.abs(forces[..., 2])
        fxy = torch.linalg.norm(forces[..., :2], dim=-1)
        return torch.any(fxy > 4 * fz, dim=-1).float().mean().item()
    except Exception:
        return 0.0


def _resolve_isaac_env(env):
    e = env
    for _ in range(5):
        if hasattr(e, "scene"):
            return e
        e = getattr(e, "env", getattr(e, "unwrapped", None))
        if e is None:
            break
    return None


# ── Main evaluation loop ───────────────────────────────────────────────────────

def main():
    # ── Load agent config (for checkpoint path resolution) ─────────────────────
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(
        args_cli.task, args_cli.agent
    )
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.seed = args_cli.seed

    # ── Load PLAY env config (50 envs, all terrain difficulties, no curriculum)
    # RobotPlayEnvCfg: num_rows=5, max_init_terrain_level=4, corruption disabled,
    # column gating disabled (terrain_levels_vel_smooth used instead).
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=agent_cfg.device,
        num_envs=args_cli.num_envs,   # None → keeps play default (50)
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.seed = args_cli.seed

    # Override velocity command to fixed evaluation speed
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (
        args_cli.force_lin_vel_x, args_cli.force_lin_vel_x
    )
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0

    # Disable curriculum updates during eval (terrain stays fixed)
    from isaaclab.managers import CurriculumTermCfg
    env_cfg.curriculum.terrain_levels = CurriculumTermCfg(
        func=lambda env, env_ids: torch.zeros(1)
    )
    if hasattr(env_cfg.curriculum, "lin_vel_cmd_levels"):
        env_cfg.curriculum.lin_vel_cmd_levels = CurriculumTermCfg(
            func=lambda env, env_ids: torch.zeros(1)
        )

    # ── Build env ──────────────────────────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    device = agent_cfg.device
    isaac_env = _resolve_isaac_env(env)

    # ── Load checkpoint ────────────────────────────────────────────────────────
    log_root_path = os.path.abspath(
        os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    )
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )

    train_cfg_dict = agent_cfg.to_dict()
    for k in ["optimizer", "share_cnn_encoders"]:
        train_cfg_dict["algorithm"].pop(k, None)

    runner = OnPolicyRunner(env, train_cfg_dict, log_dir=None, device=device)
    runner.load(resume_path)

    policy_nn = getattr(runner.alg, "actor_critic", None) or getattr(runner.alg, "policy", None)
    policy_nn.eval()

    has_pc = hasattr(policy_nn, "pc_enc_a")  # True for PointCloudActorCritic

    if args_cli.blind_pc and not has_pc:
        print("[WARN] --blind_pc set but policy has no pc_enc_a — flag has no effect.")

    # ── Determine task direction ───────────────────────────────────────────────
    is_ascend = "Ascend" in args_cli.task

    # ── Eval loop ──────────────────────────────────────────────────────────────
    metrics = TerrainMetrics()

    # Set pc_blend_alpha=1.0 so blended_lidar_pointcloud returns the actual
    # lidar scan (not reference) throughout evaluation.
    if isaac_env is not None:
        isaac_env.pc_blend_alpha = 1.0
        env.pc_blend_alpha = 1.0

    obs, _ = env.get_observations()
    obs = obs.to(device) if not isinstance(obs, dict) else {
        k: v.to(device) for k, v in obs.items()
    }

    # Record starting z per env for delta-z success check
    ep_start_z = _get_base_z(isaac_env).to(device) if isaac_env else None

    policy_nn.reset()   # zero GRU hidden state for all envs

    print(f"\n[eval_stair] task={args_cli.task}  baseline={args_cli.baseline}"
          f"  blind_pc={args_cli.blind_pc}  n_episodes={args_cli.n_episodes}"
          f"  speed={args_cli.force_lin_vel_x} m/s")
    print(f"[eval_stair] checkpoint: {resume_path}\n")

    while metrics.total_episodes() < args_cli.n_episodes and simulation_app.is_running():
        with torch.inference_mode():
            # ── Blind ablation: zero point cloud ─────────────────────────────
            eval_obs = obs
            if args_cli.blind_pc and has_pc and isinstance(obs, dict) and "pointcloud" in obs:
                eval_obs = dict(obs)
                eval_obs["pointcloud"] = torch.zeros_like(obs["pointcloud"])

            actions = policy_nn.act_inference(eval_obs)

        obs, _rewards, dones, extras = env.step(actions)
        obs = obs.to(device) if not isinstance(obs, dict) else {
            k: v.to(device) for k, v in obs.items()
        }

        # ── Per-step metrics ──────────────────────────────────────────────────
        if isaac_env is not None:
            metrics.record_step(
                energy=_compute_energy(isaac_env),
                stumble=_compute_stumble(isaac_env),
            )

        # ── Episode-end bookkeeping ───────────────────────────────────────────
        dones_cpu = dones.cpu()
        time_outs = extras.get("time_outs", torch.zeros_like(dones)).cpu().bool()
        done_ids  = dones_cpu.nonzero(as_tuple=False).squeeze(-1)

        if len(done_ids) > 0 and isaac_env is not None:
            cur_z   = _get_base_z(isaac_env).to(device)
            delta_z = cur_z - ep_start_z
            cols    = _col_label_batch(_get_terrain_col(isaac_env, device))

            cur_pos  = isaac_env.scene["robot"].data.root_pos_w[:, :2].to(device)
            origins  = isaac_env.scene.terrain.env_origins[:, :2].to(device)
            fwd_dist = torch.norm(cur_pos - origins, dim=1)
            cmd      = torch.norm(
                isaac_env.command_manager.get_command("base_velocity")[:, :2].to(device),
                dim=1,
            )
            progress_ok = fwd_dist > torch.clamp(
                cmd * float(isaac_env.max_episode_length_s) * 0.3, min=0.5
            )

            for i in done_ids.tolist():
                to   = bool(time_outs[i])
                prog = bool(progress_ok[i])
                dz   = float(delta_z[i])

                if is_ascend:
                    success = to and prog and dz > 0.05
                else:
                    success = to and prog and dz < -0.05

                metrics.record_episode(cols[i], success, to)

            # Reset starting z for done envs
            ep_start_z[done_ids] = cur_z[done_ids]

        # Reset GRU hidden state for done environments
        policy_nn.reset(dones.to(device))

        if metrics.total_episodes() % 50 == 0 and metrics.total_episodes() > 0:
            print(f"  collected {metrics.total_episodes()} / {args_cli.n_episodes} episodes ...")

    # ── Report ─────────────────────────────────────────────────────────────────
    mode_label = (
        f"Baseline-{args_cli.baseline} (blind PC)"
        if args_cli.blind_pc
        else f"Baseline-{args_cli.baseline} (normal)"
    )
    metrics.print_table(mode_label)

    env.close()


def _col_label_batch(terrain_cols: torch.Tensor) -> list[str]:
    return [_col_label(int(c)) for c in terrain_cols.cpu().tolist()]


if __name__ == "__main__":
    main()
    simulation_app.close()
