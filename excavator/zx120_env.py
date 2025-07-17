import torch
import math
import genesis as gs
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)
from itertools import chain
import numpy as np


def gs_rand_float(lower: float, upper: float, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Zx120Env:
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
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.05,
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
        self.motors_dof_idx = list(
            chain.from_iterable(
                [
                    self.robot.get_joint(name).dofs_idx_local
                    for name in self.env_cfg["joint_names"]
                ]
            )
        )
        self.bucket_end = self.robot.get_link("bucket_end_link")

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
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float
        )
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.bucket_end_pos = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.bucket_end_quat = torch.zeros(
            (self.num_envs, 4), device=gs.device, dtype=gs.tc_float
        )
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
        r = gs_rand_float(*self.command_cfg["r_range"], (len(envs_idx),), gs.device)
        az = gs_rand_float(
            *self.command_cfg["azimuth_range"], (len(envs_idx),), gs.device
        )
        alt = gs_rand_float(
            *self.command_cfg["altitude_range"], (len(envs_idx),), gs.device
        )
        self.commands[envs_idx, 0] = r * torch.cos(alt) * torch.cos(az)
        self.commands[envs_idx, 1] = r * torch.cos(alt) * torch.sin(az)
        self.commands[envs_idx, 2] = r * torch.sin(alt)

    def _at_target(self):
        self.at_target = (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"])
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
        target_dof_vel = exec_actions * self.env_cfg["action_scale"]
        self.robot.control_dofs_velocity(target_dof_vel, self.motors_dof_idx)
        # update target pos
        if self.target is not None:
            self.target.set_pos(self.commands, zero_velocity=True)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.last_bucket_end_pos[:] = self.bucket_end_pos[:]
        self.bucket_end_pos[:] = self.bucket_end.get_pos()
        self.rel_pos = self.commands - self.bucket_end_pos
        self.last_rel_pos = self.commands - self.last_bucket_end_pos
        self.bucket_end_quat[:] = self.bucket_end.get_quat()

        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # resample commands
        envs_idx = self._at_target()
        self._resample_commands(envs_idx)

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

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1),  # 3
                self.commands,  # 3
                self.dof_pos * self.obs_scales["dof_pos"],  # 4
                self.dof_vel * self.obs_scales["dof_vel"],  # 4
                self.actions,  # 4
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
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.bucket_end_pos[envs_idx] = self.bucket_end.get_pos(envs_idx)
        self.last_bucket_end_pos[envs_idx] = self.bucket_end_pos[envs_idx]
        self.bucket_end_quat[envs_idx] = self.bucket_end.get_quat(envs_idx)
        self.rel_pos = self.commands - self.bucket_end_pos
        self.last_rel_pos = self.commands - self.last_bucket_end_pos
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

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_target(self):
        target_rew = torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(
            torch.square(self.rel_pos), dim=1
        )
        return target_rew

    def _reward_smooth(self):
        smooth_rew = torch.sum(
            torch.abs(torch.sign(self.actions) - torch.sign(self.last_actions)), dim=1
        )  # signs of the velocity input consistency
        return smooth_rew

    def _reward_target_arrival(self):
        arrival_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        arrival_rew[self.at_target] = 1
        return arrival_rew
