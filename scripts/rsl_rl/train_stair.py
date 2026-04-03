"""Training script for the Go2 Stair-Aware Locomotion Policy ablation study.

  A  prop-only MLP          --baseline A
  B  raw concat MLP         --baseline B
  C  GRU + CNN encoder      --baseline C
  D  GRU + StepEdge (ours)  --baseline D  [default]

Usage:
  ./unitree_rl_lab.sh -p scripts/rsl_rl/train_stair.py \\
      --baseline D --num_envs 4096 --headless
"""

import gymnasium as gym
import pathlib
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401
sys.path.pop(0)

import argparse
import argcomplete
from isaaclab.app import AppLauncher
import cli_args  # isort: skip

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train Go2 stair-aware locomotion policy.")
parser.add_argument("--video",          action="store_true", default=False)
parser.add_argument("--video_length",   type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs",       type=int, default=None,
                    help="Number of parallel envs (overrides env cfg).")
parser.add_argument("--task",           type=str, default="Unitree-Go2-StairAware",
                    help="Gym task ID.")
parser.add_argument("--seed",           type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument(
    "--baseline", type=str, default="D", choices=["A", "B", "C", "D", "E"],
    help=(
        "Ablation: A=prop MLP, B=raw concat, C=GRU+CNN, "
        "D=GRU+StepEdge (explicit), E=GRU+PointNet (learned end-to-end)"
    ),
)
parser.add_argument("--distributed", action="store_true", default=False)
parser.add_argument("--agent", type=str, default=None)  # consumed before Hydra sees it

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# Map --baseline to registered entry-point key
_AGENT_KEY = {
    "A": "rsl_rl_baseline_A",
    "B": "rsl_rl_baseline_B",
    "C": "rsl_rl_baseline_C",
    "D": "rsl_rl_baseline_D",
    "E": "rsl_rl_baseline_E",
}
args_cli.agent = _AGENT_KEY[args_cli.baseline]

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ────────────────────────────────────────────────────────
import collections
import inspect
import os
import shutil
import statistics
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg,
    ManagerBasedRLEnvCfg, multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

# ── Inject custom policy classes into rsl_rl runner namespace ─────────────────
from unitree_rl_lab.tasks.locomotion.robots.go2.policies import (
    GRUCNNActorCritic,
    PointCloudActorCritic,
    StairAwareActorCritic,
)
import rsl_rl.runners.on_policy_runner as _runner_module
_runner_module.StairAwareActorCritic  = StairAwareActorCritic
_runner_module.GRUCNNActorCritic      = GRUCNNActorCritic
_runner_module.PointCloudActorCritic  = PointCloudActorCritic

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.deterministic    = False
torch.backends.cudnn.benchmark        = False

# ── Terrain layout constants (must match STAIR_AWARE_TERRAIN_CFG) ──────────────
# Progressive terrain: flat(30%) + ascend_tiny/medium/hard(35%) + descend_*(35%)
_NUM_COLS     = 20
_FLAT_COLS    = int(0.30 * _NUM_COLS)   # 6   → cols  0- 5
_ASCEND_COLS  = int(0.35 * _NUM_COLS)   # 7   → cols  6-12
# descend                                # 7   → cols 13-19


# ══════════════════════════════════════════════════════════════════════════════
# StairOnPolicyRunner: success rate + energy logging
# ══════════════════════════════════════════════════════════════════════════════

class StairOnPolicyRunner(OnPolicyRunner):
    """OnPolicyRunner with per-terrain success rate, energy logging, and
    auxiliary velocity-prediction supervised loss.

    Extra TensorBoard scalars:
      Stair/success_rate_flat / ascend / descend
      Stair/mean_energy
      Stair/mean_terrain_level
      Stair/aux_vel_loss   (only when actor_critic has vel_decoder)
    """

    _SUCCESS_WINDOW = 200

    # ── Adaptive pc_blend_alpha constants ──────────────────────────────────
    # From iter 0 to _PC_BLEND_START_ITER: alpha fixed at 0 (GT only).
    # From _PC_BLEND_START_ITER onwards: every _PC_BLEND_CHECK_EVERY iters,
    # compute cosine similarity between PointNet features of GT vs LiDAR
    # sensor hits (z_pc_ref vs z_pc_lidar, both using pc_enc_a).
    # An EMA of this similarity gates alpha increases: alpha only rises when
    # the encoder produces similar features for both sensors, meaning the
    # policy is already robust to the geometric difference.
    _PC_BLEND_START_ITER  : int   = 5000
    _PC_BLEND_CHECK_EVERY : int   = 50
    _PC_SIM_THRESH        : float = 0.60   # EMA cosine sim needed to allow increase
    _PC_ALPHA_LR          : float = 0.003  # max alpha increase per check step
    _PC_SIM_EMA_DECAY     : float = 0.90   # smoothing for similarity EMA

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self._current_pc_alpha: float = 0.0
        self._pc_sim_ema:       float = 0.0
        self._success_buf = {
            "flat":    collections.deque(maxlen=self._SUCCESS_WINDOW),
            "ascend":  collections.deque(maxlen=self._SUCCESS_WINDOW),
            "descend": collections.deque(maxlen=self._SUCCESS_WINDOW),
        }
        self._progress_buf = {
            "flat":    collections.deque(maxlen=self._SUCCESS_WINDOW),
            "ascend":  collections.deque(maxlen=self._SUCCESS_WINDOW),
            "descend": collections.deque(maxlen=self._SUCCESS_WINDOW),
        }
        self._energy_buf = collections.deque(maxlen=100)
        self._isaac_env  = self._resolve_isaac_env()
        # Track episode start root poses for terrain-specific success checks.
        self._ep_start_pos: torch.Tensor | None = None

        # ── Auxiliary supervised loss setup ───────────────────────────────
        # Trains actor GRU + vel_decoder to predict base_lin_vel from
        # proprioception history, providing dense gradient signal to GRU.
        self._lambda_vel: float = float(train_cfg.get("lambda_vel", 0.5))
        self._aux_policy_buf: list[torch.Tensor] = []   # (N, 45) per step
        self._aux_linvel_buf: list[torch.Tensor] = []   # (N, 3)  per step

        # RSL-RL version compatibility: attribute name differs across versions
        self._ac = (getattr(self.alg, "actor_critic", None)
                    or getattr(self.alg, "policy", None))

        ac = self._ac
        if ac is not None and hasattr(ac, "vel_decoder") and self._lambda_vel > 0.0:
            _aux_params = (
                list(ac.memory_a.parameters()) +
                list(ac.vel_decoder.parameters())
            )
            self._aux_optimizer: torch.optim.Optimizer | None = (
                torch.optim.Adam(_aux_params, lr=1e-3)
            )
            self._aux_max_grad_norm: float = float(
                getattr(self.alg, "max_grad_norm", 1.0)
            )
            print(f"[StairOnPolicyRunner] aux vel loss enabled  "
                  f"(lambda={self._lambda_vel}, "
                  f"params={sum(p.numel() for p in _aux_params):,})")
        else:
            self._aux_optimizer = None
            print("[StairOnPolicyRunner] aux vel loss disabled")

    def _resolve_isaac_env(self):
        e = self.env
        for _ in range(5):
            if hasattr(e, "scene"):
                return e
            e = getattr(e, "env", getattr(e, "unwrapped", None))
            if e is None:
                break
        return None

    def _terrain_masks(self):
        if self._isaac_env is None or not hasattr(self._isaac_env.scene, "terrain"):
            n = self.env.num_envs
            z = torch.zeros(n, dtype=torch.bool)
            return z, z, z
        tt = self._isaac_env.scene.terrain.terrain_types.to(self.device)
        flat_mask    = tt < _FLAT_COLS
        ascend_mask  = (tt >= _FLAT_COLS) & (tt < _FLAT_COLS + _ASCEND_COLS)
        descend_mask = tt >= _FLAT_COLS + _ASCEND_COLS
        return flat_mask, ascend_mask, descend_mask

    def _record_episode_outcomes(self, dones, time_outs):
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) == 0:
            return
        flat_mask, ascend_mask, descend_mask = self._terrain_masks()

        # Per-episode terrain progress for done environments.
        fwd_dist: torch.Tensor | None = None
        delta_z: torch.Tensor | None = None
        cmd_dist: torch.Tensor | None = None
        if (self._isaac_env is not None
                and self._ep_start_pos is not None
                and hasattr(self._isaac_env.scene["robot"].data, "root_pos_w")):
            cur_root = self._isaac_env.scene["robot"].data.root_pos_w.to(self.device)
            cur_pos = cur_root[:, :2]
            fwd_dist = torch.norm(cur_pos - self._ep_start_pos[:, :2], dim=1)
            delta_z = cur_root[:, 2] - self._ep_start_pos[:, 2]
            commands = self._isaac_env.command_manager.get_command("base_velocity").to(self.device)
            cmd_xy = torch.norm(commands[:, :2], dim=1)
            cmd_dist = cmd_xy * float(self._isaac_env.max_episode_length_s)
            # Reset start pos for done envs
            self._ep_start_pos[done_ids] = cur_root[done_ids]

        for name, mask in [("flat", flat_mask), ("ascend", ascend_mask), ("descend", descend_mask)]:
            ids = done_ids[mask[done_ids]]
            if len(ids) == 0:
                continue

            if fwd_dist is not None and cmd_dist is not None:
                # Treat "success" as timeout + meaningful commanded progress.
                progress_ok = fwd_dist[ids] > torch.clamp(cmd_dist[ids] * 0.3, min=0.5)
                if name == "ascend" and delta_z is not None:
                    success = time_outs[ids].bool() & progress_ok & (delta_z[ids] > 0.05)
                elif name == "descend" and delta_z is not None:
                    success = time_outs[ids].bool() & progress_ok & (delta_z[ids] < -0.05)
                else:
                    success = time_outs[ids].bool() & progress_ok
                for s in success.float().tolist():
                    self._success_buf[name].append(s)
                for d in fwd_dist[ids].tolist():
                    self._progress_buf[name].append(d)
            else:
                for s in time_outs[ids].float().tolist():
                    self._success_buf[name].append(s)

        # ascend 성공률을 env에 저장 → curriculums.py에서 속도 커리큘럼 판단에 사용
        if self._isaac_env is not None and len(self._success_buf["ascend"]) > 0:
            self._isaac_env.ascend_success_rate = statistics.mean(self._success_buf["ascend"])

    def _compute_energy(self):
        if self._isaac_env is None:
            return None
        try:
            robot = self._isaac_env.scene["robot"]
            return (robot.data.applied_torque.abs() * robot.data.joint_vel.abs()).mean().item()
        except Exception:
            return None

    def _mean_terrain_level(self):
        if self._isaac_env is None:
            return None
        try:
            return self._isaac_env.scene.terrain.terrain_levels.float().mean().item()
        except Exception:
            return None

    def _compute_pc_similarity(self) -> float | None:
        """Cosine similarity between GT and LiDAR PointNet features.

        Both sensors are encoded with the actor's pc_enc_a (no extra training).
        Returns mean cosine similarity across all envs, or None on failure.

        This measures how similarly the PointCloudEncoder responds to the two
        sensors' hit patterns.  When similarity is high the policy is already
        extracting equivalent terrain features from both, making it safe to
        increase pc_blend_alpha toward 1.
        """
        if self._isaac_env is None or self._ac is None:
            return None
        if not hasattr(self._ac, "pc_enc_a"):
            return None
        try:
            from isaaclab.utils.math import quat_apply_inverse

            robot       = self._isaac_env.scene["robot"]
            base_pos_w  = robot.data.root_pos_w
            base_quat_w = robot.data.root_quat_w
            n_envs      = base_pos_w.shape[0]

            def to_base_frame(sensor):
                hits_w = sensor.data.ray_hits_w.clone().float()
                sensor_pos_w = sensor.data.pos_w.unsqueeze(1).expand_as(hits_w)
                invalid = ~torch.isfinite(hits_w).all(dim=-1, keepdim=True)
                hits_w = torch.where(invalid, sensor_pos_w, hits_w)
                n, p, _ = hits_w.shape
                hits_rel = hits_w - base_pos_w.unsqueeze(1)
                hits_b = quat_apply_inverse(
                    base_quat_w.unsqueeze(1).expand(-1, p, -1).reshape(n * p, 4),
                    hits_rel.reshape(n * p, 3),
                ).reshape(n, p, 3)
                return hits_b.reshape(n, p * 3).to(self.device)

            ref_sensor   = self._isaac_env.scene.sensors["height_scanner"]
            lidar_sensor = self._isaac_env.scene.sensors["lidar_scanner"]

            with torch.inference_mode():
                z_ref   = self._ac.pc_enc_a(to_base_frame(ref_sensor))    # (N, 64)
                z_lidar = self._ac.pc_enc_a(to_base_frame(lidar_sensor))  # (N, 64)
                sim = F.cosine_similarity(z_ref, z_lidar, dim=-1).mean().item()
            return sim
        except Exception:
            return None

    def _update_pc_blend_alpha(self, iteration: int) -> float:
        """Adaptively update pc_blend_alpha based on encoder similarity.

        Phase 1 (iter < start): alpha = 0, GT only, policy learns locomotion.
        Phase 2 (iter >= start): check every N iters; increase alpha by alpha_lr
            only when the EMA similarity exceeds the threshold, meaning the
            PointNet encoder produces near-identical features for GT and LiDAR.
        """
        if iteration < self._PC_BLEND_START_ITER:
            alpha = 0.0
        else:
            if iteration % self._PC_BLEND_CHECK_EVERY == 0:
                sim = self._compute_pc_similarity()
                if sim is not None:
                    self._pc_sim_ema = (
                        self._PC_SIM_EMA_DECAY * self._pc_sim_ema
                        + (1.0 - self._PC_SIM_EMA_DECAY) * sim
                    )
                    if self._pc_sim_ema > self._PC_SIM_THRESH:
                        self._current_pc_alpha = min(
                            self._current_pc_alpha + self._PC_ALPHA_LR,
                            1.0,
                        )
            alpha = self._current_pc_alpha

        for target in (self._isaac_env, self.env):
            if target is not None:
                try:
                    setattr(target, "pc_blend_alpha", alpha)
                except Exception:
                    pass
        return alpha

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        self._prepare_logging_writer()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        # Initialise episode start positions for progress tracking
        if self._isaac_env is not None and hasattr(self._isaac_env.scene["robot"].data, "root_pos_w"):
            self._ep_start_pos = self._isaac_env.scene["robot"].data.root_pos_w.clone().to(self.device)

        ep_infos          = []
        rewbuffer         = collections.deque(maxlen=100)
        lenbuffer         = collections.deque(maxlen=100)
        cur_reward_sum    = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length= torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter   = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            t_start = time.time()
            pc_blend_alpha = self._update_pc_blend_alpha(it)

            # Clear aux buffers for this iteration
            self._aux_policy_buf.clear()
            self._aux_linvel_buf.clear()

            with torch.inference_mode():
                energy_sum, energy_steps = 0.0, 0
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs     = obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones   = dones.to(self.device)
                    self.alg.process_env_step(obs, rewards, dones, extras)

                    time_outs = extras.get("time_outs", torch.zeros_like(dones))
                    self._record_episode_outcomes(dones, time_outs.to(self.device))

                    e = self._compute_energy()
                    if e is not None:
                        energy_sum   += e
                        energy_steps += 1

                    # Buffer obs for auxiliary supervised loss.
                    # obs["policy"] = proprioception (45), used as GRU input sequence.
                    # obs["critic"][:, :3] = base_lin_vel (privileged ground truth).
                    if (self._aux_optimizer is not None
                            and isinstance(obs, dict)
                            and "policy" in obs
                            and "critic" in obs):
                        self._aux_policy_buf.append(obs["policy"].clone())
                        self._aux_linvel_buf.append(obs["critic"][:, :3].clone())

                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        cur_reward_sum    += rewards
                        cur_episode_length+= 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().tolist())
                        cur_reward_sum[new_ids]     = 0
                        cur_episode_length[new_ids] = 0

                collection_time = time.time() - t_start
                if energy_steps > 0:
                    self._energy_buf.append(energy_sum / energy_steps)
                self.alg.compute_returns(obs)

            t_update  = time.time()
            loss_dict = self.alg.update()
            learn_time = time.time() - t_update

            # Auxiliary supervised loss: GRU_actor + vel_decoder
            # Runs after PPO update, outside inference_mode → gradients active.
            aux_vel_loss = self._aux_loss_step()

            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                self._log_stair(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()

            if it == start_iter and not self.disable_logs:
                from rsl_rl.utils import store_code_state
                git_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_paths:
                    for p in git_paths:
                        self.writer.save_file(p)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _aux_loss_step(self) -> float:
        """Compute and apply auxiliary velocity-prediction loss.

        Feeds the buffered proprioceptive sequence through the actor GRU and
        vel_decoder, then minimises MSE vs the buffered true base_lin_vel.

        - Runs *outside* inference_mode (after alg.update()) so gradients flow.
        - Resets GRU hidden state internally — safe because the next rollout
          will rebuild context from scratch anyway.
        - Updates only: memory_a (actor GRU) + vel_decoder.

        Returns:
            Unscaled MSE value for logging, or 0.0 if skipped.
        """
        if self._aux_optimizer is None or len(self._aux_policy_buf) == 0:
            return 0.0

        # (T, N, 45): full rollout sequence; gives GRU temporal context
        policy_seq   = torch.stack(self._aux_policy_buf, dim=0)   # detached targets
        true_lin_vel = self._aux_linvel_buf[-1]                    # (N, 3), last step

        predicted_vel = self._ac.predict_lin_vel(policy_seq)
        aux_loss      = F.mse_loss(predicted_vel, true_lin_vel) * self._lambda_vel

        self._aux_optimizer.zero_grad()
        aux_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self._ac.memory_a.parameters()) +
            list(self._ac.vel_decoder.parameters()),
            self._aux_max_grad_norm,
        )
        self._aux_optimizer.step()

        return (aux_loss / self._lambda_vel).item()   # unscaled for logging

    def _lin_vel_cmd_level(self) -> float | None:
        """Return current lin_vel_x upper bound from the command curriculum."""
        if self._isaac_env is None:
            return None
        try:
            return self._isaac_env.command_manager.get_term("base_velocity").cfg.ranges.lin_vel_x[1]
        except Exception:
            return None

    def _log_stair(self, locs):
        self.log(locs)
        it = locs["it"]
        for terrain, buf in self._success_buf.items():
            if len(buf) > 0:
                self.writer.add_scalar(f"Stair/success_rate_{terrain}", statistics.mean(buf), it)
        for terrain, buf in self._progress_buf.items():
            if len(buf) > 0:
                self.writer.add_scalar(f"Stair/progress_rate_{terrain}", statistics.mean(buf), it)
        if len(self._energy_buf) > 0:
            self.writer.add_scalar("Stair/mean_energy", statistics.mean(self._energy_buf), it)
        level = self._mean_terrain_level()
        if level is not None:
            self.writer.add_scalar("Stair/mean_terrain_level", level, it)
        cmd_level = self._lin_vel_cmd_level()
        if cmd_level is not None:
            self.writer.add_scalar("Stair/lin_vel_cmd_level", cmd_level, it)
        self.writer.add_scalar("Stair/pc_blend_alpha", self._current_pc_alpha, it)
        self.writer.add_scalar("Stair/pc_sim_ema", self._pc_sim_ema, it)
        aux_vel_loss = locs.get("aux_vel_loss", 0.0)
        if aux_vel_loss > 0.0:
            self.writer.add_scalar("Stair/aux_vel_loss", aux_vel_loss, it)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed:
        env_cfg.sim.device  = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device    = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = agent_cfg.seed = seed

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging to: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "train"),
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    train_cfg_dict = agent_cfg.to_dict()

    print("algorithm cfg before:", train_cfg_dict["algorithm"])

    unsupported_alg_keys = [
        "optimizer",
        "share_cnn_encoders",
    ]

    for k in unsupported_alg_keys:
        train_cfg_dict["algorithm"].pop(k, None)

    print("algorithm cfg after:", train_cfg_dict["algorithm"])

    runner = StairOnPolicyRunner(
        env,
        train_cfg_dict,
        log_dir=log_dir,
        device=agent_cfg.device,
    )

    # runner = StairOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if agent_cfg.resume:
        print(f"[INFO] Loading checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"),   env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    print(f"\n{'='*60}")
    print(f"  Baseline : {args_cli.baseline}  ({agent_cfg.experiment_name})")
    print(f"  Policy   : {agent_cfg.policy.class_name}")
    print(f"  Envs     : {env_cfg.scene.num_envs}")
    print(f"  Iters    : {agent_cfg.max_iterations}")
    print(f"{'='*60}\n")

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    sys.argv = [sys.argv[0]] + hydra_args
    main()
    simulation_app.close()
