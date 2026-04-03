"""Training script for single-direction stair tasks (Ascend or Descend).

Always uses Baseline E (GRU + PointCloudEncoder) — no ablation flag needed.

Usage:
  # Stair Ascend
  python scripts/rsl_rl/train_stair_single.py \\
      --task Unitree-Go2-StairAscend --num_envs 4096 --headless

  # Stair Descend
  python scripts/rsl_rl/train_stair_single.py \\
      --task Unitree-Go2-StairDescend --num_envs 4096 --headless

This script is intentionally separate from train_stair.py so that the
StairAware multi-direction setup can be re-run without interference.
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
parser = argparse.ArgumentParser(description="Train Go2 single-direction stair policy (Baseline E).")
parser.add_argument("--video",          action="store_true", default=False)
parser.add_argument("--video_length",   type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs",       type=int, default=None)
parser.add_argument(
    "--task", type=str, default="Unitree-Go2-StairAscend",
    choices=["Unitree-Go2-StairAscend", "Unitree-Go2-StairDescend"],
    help="Which single-direction task to train.",
)
parser.add_argument("--seed",           type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--distributed",    action="store_true", default=False)
parser.add_argument("--agent",          type=str, default=None)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

# Always Baseline E — fixed entry-point key
args_cli.agent = "rsl_rl_cfg_entry_point"

if args_cli.video:
    args_cli.enable_cameras = True

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

# ── Terrain layout constants ───────────────────────────────────────────────────
# StairAscend:  flat(20%) + ascend*(80%)  → _FLAT_COLS=2, _ASCEND_COLS=8, num_cols=10
# StairDescend: flat(20%) + descend*(80%) → _FLAT_COLS=2, _ASCEND_COLS=0, num_cols=10
_TERRAIN_LAYOUT = {
    "Unitree-Go2-StairAscend":  {"num_cols": 10, "flat_cols": 2, "ascend_cols": 8},
    "Unitree-Go2-StairDescend": {"num_cols": 10, "flat_cols": 2, "ascend_cols": 0},
}


# ══════════════════════════════════════════════════════════════════════════════
# SingleStairOnPolicyRunner
# ══════════════════════════════════════════════════════════════════════════════

class SingleStairOnPolicyRunner(OnPolicyRunner):
    """OnPolicyRunner for single-direction stair tasks.

    Tracks the relevant success rate (ascend or descend) and writes it to
    the env so the speed curriculum can gate on it.

    Extra TensorBoard scalars:
      Stair/success_rate_flat / ascend / descend
      Stair/mean_energy
      Stair/mean_terrain_level
      Stair/lin_vel_cmd_level
      Stair/pc_blend_alpha
      Stair/aux_vel_loss
    """

    _SUCCESS_WINDOW       = 200
    _PC_BLEND_START_ITER  : int   = 5000
    _PC_BLEND_CHECK_EVERY : int   = 50
    _PC_SIM_THRESH        : float = 0.60
    _PC_ALPHA_LR          : float = 0.003
    _PC_SIM_EMA_DECAY     : float = 0.90

    def __init__(self, env, train_cfg, log_dir=None, device="cpu", task_name="Unitree-Go2-StairAscend"):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self._task_name         = task_name
        self._current_pc_alpha: float = 0.0
        self._pc_sim_ema:       float = 0.0
        layout = _TERRAIN_LAYOUT[task_name]
        self._flat_cols   = layout["flat_cols"]
        self._ascend_cols = layout["ascend_cols"]

        self._success_buf = {
            "flat":    collections.deque(maxlen=self._SUCCESS_WINDOW),
            "ascend":  collections.deque(maxlen=self._SUCCESS_WINDOW),
            "descend": collections.deque(maxlen=self._SUCCESS_WINDOW),
        }
        self._energy_buf = collections.deque(maxlen=100)
        self._isaac_env  = self._resolve_isaac_env()
        self._ep_start_pos: torch.Tensor | None = None

        self._lambda_vel: float = float(train_cfg.get("lambda_vel", 0.5))
        self._aux_policy_buf: list[torch.Tensor] = []
        self._aux_linvel_buf: list[torch.Tensor] = []

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
            print(f"[SingleStairOnPolicyRunner] aux vel loss enabled  "
                  f"(lambda={self._lambda_vel})")
        else:
            self._aux_optimizer = None
            print("[SingleStairOnPolicyRunner] aux vel loss disabled")

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
        flat_mask    = tt < self._flat_cols
        ascend_mask  = (tt >= self._flat_cols) & (tt < self._flat_cols + self._ascend_cols)
        descend_mask = tt >= self._flat_cols + self._ascend_cols
        return flat_mask, ascend_mask, descend_mask

    def _record_episode_outcomes(self, dones, time_outs):
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) == 0:
            return
        flat_mask, ascend_mask, descend_mask = self._terrain_masks()

        fwd_dist = delta_z = cmd_dist = None
        if (self._isaac_env is not None
                and self._ep_start_pos is not None
                and hasattr(self._isaac_env.scene["robot"].data, "root_pos_w")):
            cur_root = self._isaac_env.scene["robot"].data.root_pos_w.to(self.device)
            fwd_dist = torch.norm(cur_root[:, :2] - self._ep_start_pos[:, :2], dim=1)
            delta_z  = cur_root[:, 2] - self._ep_start_pos[:, 2]
            commands = self._isaac_env.command_manager.get_command("base_velocity").to(self.device)
            cmd_dist = torch.norm(commands[:, :2], dim=1) * float(self._isaac_env.max_episode_length_s)
            self._ep_start_pos[done_ids] = cur_root[done_ids]

        for name, mask in [("flat", flat_mask), ("ascend", ascend_mask), ("descend", descend_mask)]:
            ids = done_ids[mask[done_ids]]
            if len(ids) == 0:
                continue
            if fwd_dist is not None and cmd_dist is not None:
                progress_ok = fwd_dist[ids] > torch.clamp(cmd_dist[ids] * 0.3, min=0.5)
                if name == "ascend" and delta_z is not None:
                    success = time_outs[ids].bool() & progress_ok & (delta_z[ids] > 0.05)
                elif name == "descend" and delta_z is not None:
                    success = time_outs[ids].bool() & progress_ok & (delta_z[ids] < -0.05)
                else:
                    success = time_outs[ids].bool() & progress_ok
                for s in success.float().tolist():
                    self._success_buf[name].append(s)
            else:
                for s in time_outs[ids].float().tolist():
                    self._success_buf[name].append(s)

        # Write success rate back to env for speed curriculum
        if self._isaac_env is not None:
            if (self._task_name == "Unitree-Go2-StairAscend"
                    and len(self._success_buf["ascend"]) > 0):
                self._isaac_env.ascend_success_rate = statistics.mean(self._success_buf["ascend"])
            elif (self._task_name == "Unitree-Go2-StairDescend"
                    and len(self._success_buf["descend"]) > 0):
                self._isaac_env.descend_success_rate = statistics.mean(self._success_buf["descend"])

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
        if self._isaac_env is None or self._ac is None:
            return None
        if not hasattr(self._ac, "pc_enc_a"):
            return None
        try:
            from isaaclab.utils.math import quat_apply_inverse

            robot       = self._isaac_env.scene["robot"]
            base_pos_w  = robot.data.root_pos_w
            base_quat_w = robot.data.root_quat_w

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
                z_ref   = self._ac.pc_enc_a(to_base_frame(ref_sensor))
                z_lidar = self._ac.pc_enc_a(to_base_frame(lidar_sensor))
                sim = F.cosine_similarity(z_ref, z_lidar, dim=-1).mean().item()
            return sim
        except Exception:
            return None

    def _update_pc_blend_alpha(self, iteration: int) -> float:
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

        if self._isaac_env is not None and hasattr(self._isaac_env.scene["robot"].data, "root_pos_w"):
            self._ep_start_pos = self._isaac_env.scene["robot"].data.root_pos_w.clone().to(self.device)

        ep_infos           = []
        rewbuffer          = collections.deque(maxlen=100)
        lenbuffer          = collections.deque(maxlen=100)
        cur_reward_sum     = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter   = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            t_start        = time.time()
            pc_blend_alpha = self._update_pc_blend_alpha(it)

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
                        cur_reward_sum     += rewards
                        cur_episode_length += 1
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
        if self._aux_optimizer is None or len(self._aux_policy_buf) == 0:
            return 0.0
        policy_seq   = torch.stack(self._aux_policy_buf, dim=0)
        true_lin_vel = self._aux_linvel_buf[-1]
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
        return (aux_loss / self._lambda_vel).item()

    def _lin_vel_cmd_level(self) -> float | None:
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
        # Two-axis curriculum diagnostics
        if self._isaac_env is not None:
            if self._task_name == "Unitree-Go2-StairAscend":
                tiny_sr   = getattr(self._isaac_env, "tiny_success_rate", None)
                medium_sr = getattr(self._isaac_env, "medium_success_rate", None)
                max_col   = getattr(self._isaac_env, "max_accessible_col", None)
            else:
                tiny_sr   = getattr(self._isaac_env, "descend_tiny_success_rate", None)
                medium_sr = getattr(self._isaac_env, "descend_medium_success_rate", None)
                max_col   = getattr(self._isaac_env, "descend_max_accessible_col", None)
            if tiny_sr   is not None: self.writer.add_scalar("Stair/tiny_success_rate",   tiny_sr,   it)
            if medium_sr is not None: self.writer.add_scalar("Stair/medium_success_rate", medium_sr, it)
            if max_col   is not None: self.writer.add_scalar("Stair/max_accessible_col",  max_col,   it)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs   = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations

    env_cfg.seed        = agent_cfg.seed
    env_cfg.sim.device  = args_cli.device if args_cli.device is not None else env_cfg.sim.device

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

    unsupported_alg_keys = ["optimizer", "share_cnn_encoders"]
    for k in unsupported_alg_keys:
        train_cfg_dict["algorithm"].pop(k, None)

    runner = SingleStairOnPolicyRunner(
        env,
        train_cfg_dict,
        log_dir=log_dir,
        device=agent_cfg.device,
        task_name=args_cli.task,
    )
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
    print(f"  Task     : {args_cli.task}")
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
