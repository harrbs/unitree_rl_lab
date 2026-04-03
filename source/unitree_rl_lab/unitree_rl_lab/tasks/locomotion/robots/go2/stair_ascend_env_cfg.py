"""Stair-Ascend locomotion environment for Unitree Go2.

Focused single-direction task: flat (20%) + ascending stairs (80%).
Use this instead of StairAware when you want to train ascending only.

Terrain layout (num_cols=10):
  flat          20%  → cols 0-1  (2 cols)
  ascend_tiny   30%  → cols 2-4  (3 cols)  step 0.02–0.06 m, width 0.45 m
  ascend_medium 30%  → cols 5-7  (3 cols)  step 0.05–0.15 m, width 0.35 m
  ascend_hard   20%  → cols 8-9  (2 cols)  step 0.10–0.25 m, width 0.30 m
"""

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp


_MID360_PITCH_DEG = 13.0
_MID360_PITCH_RAD = math.radians(_MID360_PITCH_DEG)
_MID360_ROT_WXYZ = (
    math.cos(_MID360_PITCH_RAD / 2.0),
    0.0,
    math.sin(_MID360_PITCH_RAD / 2.0),
    0.0,
)


STAIR_ASCEND_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        # ── Flat (20%) ─────────────────────────────────────────────────────
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.20),

        # ── Ascending stairs (80%) ─────────────────────────────────────────
        # InvertedPyramid: center is the lowest point, edges are high.
        # Robot spawns at center (bottom) and walks outward → climbs UP.
        "ascend_tiny": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=(0.02, 0.06),
            step_width=0.45,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "ascend_medium": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=(0.05, 0.15),
            step_width=0.35,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "ascend_hard": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.10, 0.25),
            step_width=0.30,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=STAIR_ASCEND_TERRAIN_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.1, 0.7], ordering="yx"),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    lidar_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.1870, 0.0, 0.0803),
            rot=_MID360_ROT_WXYZ,
        ),
        ray_alignment="base",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.1, 0.7], ordering="yx"),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (-1.0, 1.0)},
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    # Stair ascend: forward-only commands.
    # Lateral and angular velocity are fixed at 0 — side-stepping or turning
    # on stairs is unnatural and adds unnecessary exploration difficulty.
    # Speed curriculum (lin_vel_cmd_levels) can still raise the x upper bound.
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.10, 0.30), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.10, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.5})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class HeightCfg(ObsGroup):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            scale=1.0,
            noise=Unoise(n_min=-0.02, n_max=0.02),
            clip=(-0.5, 2.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PointCloudCfg(ObsGroup):
        point_cloud = ObsTerm(
            func=mdp.blended_lidar_pointcloud,
            params={
                "sensor_cfg": SceneEntityCfg("lidar_scanner"),
                "ref_sensor_cfg": SceneEntityCfg("height_scanner"),
            },
            noise=Unoise(n_min=-0.02, n_max=0.02),
            clip=(-2.0, 2.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy:     PolicyCfg     = PolicyCfg()
    height:     HeightCfg     = HeightCfg()
    pointcloud: PointCloudCfg = PointCloudCfg()
    critic:     CriticCfg     = CriticCfg()


@configclass
class RewardsCfg:
    # ── Task ──────────────────────────────────────────────────────────────
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # ── Base ──────────────────────────────────────────────────────────────
    base_linear_velocity = RewTerm(
        func=mdp.lin_vel_z_down,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    roll_pitch_angular_velocity = RewTerm(
        func=mdp.roll_pitch_ang_vel_l2,
        weight=-0.05,
        params={"pitch_weight": 0.25, "roll_weight": 1.0},
    )
    roll_pitch_orientation_l2 = RewTerm(
        func=mdp.roll_pitch_orientation_l2,
        weight=-2.0,
        params={"pitch_weight": 0.25, "roll_weight": 1.0},
    )

    # ── Joints ────────────────────────────────────────────────────────────
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty, weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )

    # ── Feet ──────────────────────────────────────────────────────────────
    feet_air_time = RewTerm(
        func=mdp.feet_air_time, weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty, weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide, weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    foot_clearance = RewTerm(
        func=mdp.foot_clearance_reward_terrain_rel, weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "target_clearance": 0.18,  # 18 cm — must clear max step height (25 cm) minus stance margin
            "std": 0.05,
            "tanh_mult": 2.0,
        },
    )
    feet_gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.2,  # reduced from 0.5 — trot pattern is helpful on flat/tiny but too rigid on medium/hard
        params={
            "period": 0.5,
            "offset": [0.0, 0.5, 0.5, 0.0],
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 0.5,
            "command_name": "base_velocity",
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble, weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    # Ascending-specific rewards
    ascend_forward_progress = RewTerm(
        func=mdp.ascend_forward_progress,
        weight=3.0,  # strong signal to push through stair resistance
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "flat_cols": 2,    # cols 0-1 = flat (20%)
            "ascend_cols": 8,  # cols 2-9 = ascend tiny/medium/hard (80%)
        },
    )
    stair_height_progress = RewTerm(
        func=mdp.stair_height_progress, weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # ── Safety ────────────────────────────────────────────────────────────
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]),
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    # Two-axis curriculum: row difficulty + column-type gating (flat→tiny→medium→hard)
    terrain_levels = CurrTerm(func=mdp.terrain_levels_two_axis_ascend)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 15.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.lidar_scanner.update_period = self.decimation * self.sim.dt

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50          # default: 5 rows × 10 cols = 50 tiles
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 10
        self.scene.terrain.max_init_terrain_level = 4  # spawn across all difficulty rows (0-4)
        self.observations.policy.enable_corruption = False
        self.observations.height.enable_corruption = False
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        # Play: disable two-axis column gating so all 10 columns are used
        from isaaclab.managers import CurriculumTermCfg as CurrTerm
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel_smooth)
