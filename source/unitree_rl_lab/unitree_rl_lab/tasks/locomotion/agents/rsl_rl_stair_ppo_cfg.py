"""PPO runner configs for the stair-aware locomotion task.

obs_groups convention (unitree_rl_lab):
  "policy"     → actor proprioception      (45 dim, no lin_vel)
  "height"     → projected height scan     (96 dim)
  "pointcloud" → raw 3-D hits, robot frame (288 dim = 96 pts × 3)
  "critic"     → privileged critic obs     (60 dim, includes lin_vel)

Baselines:
  A: prop-only MLP           policy(45)
  B: raw concat MLP          policy(45)+height(96)
  C: GRU + CNN               policy(45)+height(96)
  D: GRU + StepEdge          policy(45)+height(96)    ← explicit extractor
  E: GRU + PointCloudEncoder policy(45)+pointcloud(288) ← learned end-to-end
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

_BASE_ALGORITHM = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.01,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-3,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)


@configclass
class StairAwarePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Baseline D (proposed): GRU + StepEdgeExtractor, asymmetric actor-critic."""

    num_steps_per_env = 32
    max_iterations = 50000
    save_interval = 100
    experiment_name = "go2_stair_D_proposed"

    # Auxiliary supervised loss weight.
    # L_total = L_ppo + lambda_vel * MSE(predicted_lin_vel, true_lin_vel)
    # Set to 0.0 to disable aux loss.
    lambda_vel: float = 0.5

    obs_groups = {
        "policy": ["policy", "height"],  # Actor:  45 + 96 = 141 dim
        "critic": ["critic", "height"],  # Critic: 60 + 96 = 156 dim
    }

    policy = RslRlPpoActorCriticCfg(
        class_name="StairAwareActorCritic",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )
    algorithm = _BASE_ALGORITHM


@configclass
class StairBaselineAPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Baseline A: prop-only MLP (no height, asymmetric critic)."""

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "go2_stair_A_prop_mlp"

    obs_groups = {
        "policy": ["policy"],   # Actor:  45 dim
        "critic": ["critic"],   # Critic: 60 dim
    }

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = _BASE_ALGORITHM


@configclass
class StairBaselineBPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Baseline B: raw concat MLP (prop + height, asymmetric critic)."""

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "go2_stair_B_raw_concat"

    obs_groups = {
        "policy": ["policy", "height"],  # Actor:  45 + 96 = 141 dim
        "critic": ["critic", "height"],  # Critic: 60 + 96 = 156 dim
    }

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = _BASE_ALGORITHM


@configclass
class StairBaselineCPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Baseline C: GRU + CNN encoder, asymmetric actor-critic."""

    num_steps_per_env = 32
    max_iterations = 50000
    save_interval = 100
    experiment_name = "go2_stair_C_gru_cnn"

    obs_groups = {
        "policy": ["policy", "height"],  # Actor:  45 + 96 = 141 dim
        "critic": ["critic", "height"],  # Critic: 60 + 96 = 156 dim
    }

    policy = RslRlPpoActorCriticCfg(
        class_name="GRUCNNActorCritic",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )
    algorithm = _BASE_ALGORITHM


@configclass
class StairBaselineEPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Baseline E (proposed): GRU + PointCloudEncoder, end-to-end from raw 3-D hits.

    Key difference vs Baseline D:
      - Input: obs["pointcloud"] (288 = 96pts×3) instead of obs["height"] (96)
      - Encoder: PointCloudEncoder (learned PointNet) instead of StepEdgeExtractor
      - No manual grid projection at deployment — just ROI filter + frame transform.
    """

    num_steps_per_env = 32
    max_iterations    = 50000
    save_interval     = 100
    experiment_name   = "go2_stair_E_pointcloud"

    # lambda for auxiliary lin_vel supervised loss (0.0 = disabled)
    lambda_vel: float = 0.5

    obs_groups = {
        "policy": ["policy", "pointcloud"],  # Actor:  45 + 288 = 333 dim
        "critic": ["critic", "pointcloud"],  # Critic: 60 + 288 = 348 dim
    }

    policy = RslRlPpoActorCriticCfg(
        class_name="PointCloudActorCritic",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )
    algorithm = _BASE_ALGORITHM


@configclass
class StairAscendBaselineAPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Baseline A for StairAscend task: prop-only MLP, no exteroception.

    Identical to StairAscendPPORunnerCfg (E) in every training hyperparameter;
    only the policy architecture and obs_groups differ.  Use this for fair
    ablation: same env, same rewards, same terrain, same PPO settings.
    """

    num_steps_per_env = 32       # same as E
    max_iterations    = 50000
    save_interval     = 100
    experiment_name   = "go2_stair_ascend_A"

    obs_groups = {
        "policy": ["policy"],    # Actor:  47 dim  (prop only, no PC)
        "critic": ["critic"],    # Critic: 60 dim  (privileged, no PC)
    }

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = _BASE_ALGORITHM


@configclass
class StairAscendPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Baseline E for StairAscend task: GRU + PointCloudEncoder."""

    num_steps_per_env = 32
    max_iterations    = 50000
    save_interval     = 100
    experiment_name   = "go2_stair_ascend"

    lambda_vel: float = 0.5

    obs_groups = {
        "policy": ["policy", "pointcloud"],  # Actor:  47 + 288 = 335 dim
        "critic": ["critic", "pointcloud"],  # Critic: 60 + 288 = 348 dim
    }

    policy = RslRlPpoActorCriticCfg(
        class_name="PointCloudActorCritic",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )
    algorithm = _BASE_ALGORITHM


@configclass
class StairDescendBaselineAPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Baseline A for StairDescend task: prop-only MLP, no exteroception."""

    num_steps_per_env = 32
    max_iterations    = 50000
    save_interval     = 100
    experiment_name   = "go2_stair_descend_A"

    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = _BASE_ALGORITHM


@configclass
class StairDescendPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Baseline E for StairDescend task: GRU + PointCloudEncoder."""

    num_steps_per_env = 32
    max_iterations    = 50000
    save_interval     = 100
    experiment_name   = "go2_stair_descend"

    lambda_vel: float = 0.5

    obs_groups = {
        "policy": ["policy", "pointcloud"],  # Actor:  47 + 288 = 335 dim
        "critic": ["critic", "pointcloud"],  # Critic: 60 + 288 = 348 dim
    }

    policy = RslRlPpoActorCriticCfg(
        class_name="PointCloudActorCritic",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )
    algorithm = _BASE_ALGORITHM
