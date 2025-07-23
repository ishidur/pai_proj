import math
from itertools import chain

import genesis as gs
import torch
from genesis.utils.geom import (
    inv_quat,
    quat_to_xyz,
    transform_by_quat,
    xyz_to_quat,
)


def gs_rand_float(lower: float, upper: float, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def cartesian_to_spherical(cartesian_coords: torch.Tensor) -> torch.Tensor:
    """
    3次元直交座標系 (x, y, z) から極座標（球座標系） (r, theta, phi) に変換します。

    Args:
        cartesian_coords (torch.Tensor): (n, 3) の形状を持つTensor。各行が (x, y, z) 座標を表します。

    Returns:
        torch.Tensor: (n, 3) の形状を持つTensor。各行が (r, theta, phi) 座標を表します。
                      - r: 原点からの距離（半径）
                      - theta: z軸正方向からの角度（極角）。範囲は [0, pi] です。
                      - phi: xy平面上でのx軸正方向からの角度（方位角）。範囲は [-pi, pi] です。
    """
    x = cartesian_coords[:, 0]
    y = cartesian_coords[:, 1]
    z = cartesian_coords[:, 2]

    # ゼロ除算を避けるための微小な値
    eps = 1e-8

    # 半径 r の計算
    r = torch.norm(cartesian_coords, p=2, dim=1)

    # 極角 theta の計算
    # 浮動小数点精度の問題で acos の引数が [-1, 1] の範囲外になるのを防ぐために clamp を使用
    theta = torch.acos(torch.clamp(z / (r + eps), -1.0, 1.0))

    # 方位角 phi の計算
    phi = torch.atan2(y, x)

    # 結果を (r, theta, phi) の順でスタックして返す
    spherical_coords = torch.stack((r, theta, phi), dim=1)

    return spherical_coords


def spherical_diff(p1: torch.Tensor, p2: torch.Tensor):
    """
    p1, p2: [n, 3] tensor 各行が (r, theta, phi)
    returns: (r, theta, phi) tensors of shape [n]
    """

    # 半径差（符号付き）
    delta_r = p2[:, 0] - p1[:, 0]

    # 角度差（正規化して -π〜π に収める）
    raw_theta_diff = p2[:, 1] - p1[:, 1]
    raw_phi_diff = p2[:, 2] - p1[:, 2]
    delta_theta = torch.remainder(raw_theta_diff + math.pi, 2 * math.pi) - math.pi
    delta_phi = torch.remainder(raw_phi_diff + math.pi, 2 * math.pi) - math.pi
    normalized_diff = torch.stack((delta_r, delta_theta, delta_phi), dim=1)

    return normalized_diff


def normalize_angle(ang: torch.Tensor):
    return torch.remainder(ang + math.pi, 2 * math.pi) - math.pi


def calculate_distance_tensor_batched(line_points, line_angles, points):
    """
    線と点のペアごとの距離を計算する関数（角度を用いる）
    :param line_points: 線上の各点 -> Tensor型 [n, 2]
    :param line_angles: 各線の角度（ラジアン） -> Tensor型 [n]
    :param points: 各点 -> Tensor型 [n, 2]
    :return: 各線と点の距離 -> Tensor型 [n]
    """
    # ラジアンから線の方向ベクトルを作成
    directions = torch.stack(
        [torch.cos(line_angles), torch.sin(line_angles)], dim=1
    )  # [n, 2]

    # 各点と線上の点のベクトルを計算
    vectors = points - line_points  # [n, 2]

    # ベクトルを線の方向に投影
    projection = (
        torch.sum(vectors * directions, dim=1, keepdim=True)
    ) * directions  # [n, 2]

    # 点と線の距離を計算（ベクトルから投影を引いた残差のノルム）
    distances = torch.norm(vectors - projection, dim=1)  # [n]

    return distances


class MovePoseEnv:
    def __init__(
        self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False
    ):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = 0.01  # control frequency on real robot is 100hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(10.0, 10.0, 10.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )
        # add plane
        self.scene.add_entity(gs.morphs.Plane())

        # add target
        if self.env_cfg["visualize_target"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.URDF(
                    file="../assets/zx120/zx120-bucket.urdf",
                    fixed=False,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(1.0, 0.5, 0.5),
                    ),
                ),
            )
        else:
            self.target = None

        # add camera
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(10.0, 10.0, 10.0),
                lookat=(0.0, 0.0, 1.0),
                fov=40,
                GUI=False,
            )

        # add robot
        self.base_init_pos = torch.tensor(
            self.env_cfg["base_init_pos"], device=gs.device
        )
        self.base_init_quat = torch.tensor(
            self.env_cfg["base_init_quat"], device=gs.device
        )
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file="../assets/zx120/zx120.urdf"),
        )

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.crawlers_dof_idx = list(
            chain.from_iterable(
                [
                    self.robot.get_joint(name).dofs_idx_local
                    for name in self.env_cfg["crawler_joints"]
                ]
            )
        )
        self.motors_dof_idx = list(
            chain.from_iterable(
                [
                    self.robot.get_joint(name).dofs_idx_local
                    for name in self.env_cfg["joint_names"]
                ]
            )
        )
        self.bucket_end = self.robot.get_link("bucket_end_link")
        self.bucket_yaw = self.robot.get_link("body_link")

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(
                (self.num_envs,), device=gs.device, dtype=gs.tc_float
            )

        # initialize buffers
        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=gs.device, dtype=gs.tc_float
        )
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float
        )
        self.rew_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_float
        )
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_int
        )
        self.commands = torch.zeros(
            (self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float
        )
        self.commands_quat = torch.zeros(
            (self.num_envs, 4), device=gs.device, dtype=gs.tc_float
        )
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float
        )
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros(
            (self.num_envs, len(self.env_cfg["joint_names"])),
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.dof_vel = torch.zeros_like(self.dof_pos)
        self.bucket_end_pos = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.bucket_pose = torch.zeros(
            (self.num_envs, 2), device=gs.device, dtype=gs.tc_float
        )
        self.rel_pose = torch.zeros_like(self.bucket_pose)
        self.last_bucket_end_pos = torch.zeros_like(self.bucket_end_pos)
        self.default_dof_pos = torch.tensor(
            [
                self.env_cfg["default_joint_angles"][name]
                for name in self.env_cfg["joint_names"]
            ],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

    def _resample_commands(self, envs_idx):
        x = gs_rand_float(*self.command_cfg["x_range"], (len(envs_idx),), gs.device)
        y = gs_rand_float(*self.command_cfg["y_range"], (len(envs_idx),), gs.device)
        z = gs_rand_float(*self.command_cfg["z_range"], (len(envs_idx),), gs.device)
        bucket_pitch = gs_rand_float(
            *self.command_cfg["bucket_pitch_range"], (len(envs_idx),), gs.device
        )
        bucket_yaw = gs_rand_float(
            *self.command_cfg["bucket_yaw_range"], (len(envs_idx),), gs.device
        )
        self.commands[envs_idx, 0] = x
        self.commands[envs_idx, 1] = y
        self.commands[envs_idx, 2] = z
        self.commands[envs_idx, 3] = bucket_pitch
        self.commands[envs_idx, 4] = bucket_yaw
        self.commands_quat[envs_idx] = xyz_to_quat(
            torch.stack(
                (
                    torch.zeros_like(bucket_pitch),
                    bucket_pitch - math.pi * 0.5,
                    bucket_yaw,
                ),
                dim=1,
            ),
            rpy=True,
        )

    def _at_target(self):
        self.at_target = (
            (
                (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"])
                & (
                    torch.sum(torch.abs(self.rel_pose), dim=1)
                    < self.env_cfg["bucket_pose_threshold"]
                )
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        return self.at_target

    def step(self, actions):
        self.actions = torch.clip(
            actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]
        )
        exec_actions = (
            self.last_actions if self.simulate_action_latency else self.actions
        )
        target_crawler_vel = (
            exec_actions[:, :2].repeat(1, 3) * self.env_cfg["crawler_action_scale"]
        )
        self.robot.control_dofs_velocity(target_crawler_vel, self.crawlers_dof_idx)
        target_dof_vel = exec_actions[:, 2:] * self.env_cfg["action_scale"]
        self.robot.control_dofs_velocity(target_dof_vel, self.motors_dof_idx)
        # update target pos
        if self.target is not None:
            self.target.set_pos(self.commands[:, :3], zero_velocity=True)
            self.target.set_quat(self.commands_quat, zero_velocity=True)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        inv_base_quat = inv_quat(self.base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.last_bucket_end_pos[:] = self.bucket_end_pos[:]
        self.bucket_end_pos[:] = self.bucket_end.get_pos()
        self.rel_pos = self.commands[:, :3] - self.bucket_end_pos
        self.last_rel_pos = self.commands[:, :3] - self.last_bucket_end_pos

        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)
        self.bucket_pose[:, 0] = torch.sum(self.dof_pos[:, 1:], dim=1) + 2.44346095279
        self.bucket_pose[:, 1] = quat_to_xyz(self.bucket_yaw.get_quat(), rpy=True)[:, 2]
        self.rel_pose = normalize_angle(self.bucket_pose - self.commands[:, 3:])
        # resample commands
        envs_idx = self._at_target()
        self._resample_commands(envs_idx)
        self.rel_pose[envs_idx] = normalize_angle(
            self.bucket_pose[envs_idx] - self.commands[envs_idx, 3:]
        )

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length

        time_out_idx = (
            (self.episode_length_buf > self.max_episode_length)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self.extras["time_outs"] = torch.zeros_like(
            self.reset_buf, device=gs.device, dtype=gs.tc_float
        )
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())
        base2target = self.commands[:, :3] - self.base_pos

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_quat,  # 4
                base2target,  # 3
                self.rel_pos,  # 3
                self.rel_pose,  # 2
                self.bucket_pose,  # 2
                self.commands,  # 5
                self.dof_pos,  # 4
                self.dof_vel,  # 4
                self.actions,  # 6
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(
            self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx
        )
        self.robot.set_quat(
            self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx
        )
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.bucket_end_pos[envs_idx] = self.bucket_end.get_pos(envs_idx)
        self.last_bucket_end_pos[envs_idx] = self.bucket_end_pos[envs_idx]
        self.bucket_pose[envs_idx, 0] = (
            torch.sum(self.dof_pos[envs_idx, 1:], dim=1) + 2.44346095279
        )
        self.bucket_pose[envs_idx, 1] = quat_to_xyz(
            self.bucket_yaw.get_quat(envs_idx), rpy=True
        )[:, 2]

        self.rel_pos = self.commands[:, :3] - self.bucket_end_pos
        self.last_rel_pos = self.commands[:, :3] - self.last_bucket_end_pos

        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)
        self.rel_pose[envs_idx] = normalize_angle(
            self.bucket_pose[envs_idx] - self.commands[envs_idx, 3:]
        )

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_target(self):
        target_rew = torch.sum(torch.abs(self.last_rel_pos), dim=1) - torch.sum(
            torch.abs(self.rel_pos), dim=1
        )
        return target_rew

    def _reward_bucket_pose(self):
        pose_rew = torch.sum(torch.abs(self.rel_pose), dim=1)
        return pose_rew

    def _reward_base_pos(self):
        base_pos_rew = (
            torch.clamp(
                torch.norm(self.commands[:, :3] - self.base_pos, dim=1), min=5.0
            )
            - 5.0
        )
        return base_pos_rew

    def _reward_base_pose(self):
        base_pose_rew = calculate_distance_tensor_batched(
            self.commands[:, :2], self.commands[:, 4], self.base_pos[:, :2]
        )
        return base_pose_rew

    def _reward_smooth(self):
        smooth_rew = torch.sum(
            torch.abs(torch.sign(self.actions) - torch.sign(self.last_actions)), dim=1
        )  # signs of the velocity input consistency
        return smooth_rew

    def _reward_target_arrival(self):
        arrival_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        arrival_rew[self.at_target] = 1
        return arrival_rew

    def _reward_angular(self):
        angular_rew = torch.norm(self.base_ang_vel, dim=1)
        return angular_rew
