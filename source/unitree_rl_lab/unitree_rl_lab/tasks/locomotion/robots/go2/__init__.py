import gymnasium as gym

gym.register(
    id="Unitree-Go2-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Stairs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stairs_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.stairs_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

##
# Stair-Aware Locomotion Policy (ablation A~D)
##

_STAIR_AGENTS = "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_stair_ppo_cfg"

gym.register(
    id="Unitree-Go2-StairAscend",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":      f"{__name__}.stair_ascend_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.stair_ascend_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point":   f"{_STAIR_AGENTS}:StairAscendPPORunnerCfg",
        "rsl_rl_baseline_A":        f"{_STAIR_AGENTS}:StairAscendBaselineAPPORunnerCfg",
        "rsl_rl_baseline_E":        f"{_STAIR_AGENTS}:StairAscendPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-StairDescend",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":      f"{__name__}.stair_descend_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.stair_descend_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point":   f"{_STAIR_AGENTS}:StairDescendPPORunnerCfg",
        "rsl_rl_baseline_A":        f"{_STAIR_AGENTS}:StairDescendBaselineAPPORunnerCfg",
        "rsl_rl_baseline_E":        f"{_STAIR_AGENTS}:StairDescendPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-StairAware",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":    f"{__name__}.stair_aware_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.stair_aware_env_cfg:RobotPlayEnvCfg",
        # default → Baseline D (proposed)
        "rsl_rl_cfg_entry_point": f"{_STAIR_AGENTS}:StairAwarePPORunnerCfg",
        # ablation entry points selected via --baseline flag in train_stair.py
        "rsl_rl_baseline_A":      f"{_STAIR_AGENTS}:StairBaselineAPPORunnerCfg",
        "rsl_rl_baseline_B":      f"{_STAIR_AGENTS}:StairBaselineBPPORunnerCfg",
        "rsl_rl_baseline_C":      f"{_STAIR_AGENTS}:StairBaselineCPPORunnerCfg",
        "rsl_rl_baseline_D":      f"{_STAIR_AGENTS}:StairAwarePPORunnerCfg",
        "rsl_rl_baseline_E":      f"{_STAIR_AGENTS}:StairBaselineEPPORunnerCfg",
    },
)
