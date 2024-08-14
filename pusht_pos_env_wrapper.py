#%%
import torch
import sapien
import numpy as np
import os.path as osp
import gymnasium as gym
from mani_skill.envs.tasks import PushTEnv
from mani_skill.utils.wrappers.record import RecordEpisode
from typing import Any, Dict, List, Sequence, Tuple, Union
from mani_skill.utils.structs.types import Array
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.utils import common, gym_utils, sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env


@register_env("PosPushT-v1", max_episode_steps=100)
class PosPushTEnv(PushTEnv):
    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # reward for overlap of the tees
        reward = self.pseudo_render_reward()
        return reward
    

obs_mode = "rgb"
render_mode = "human"
reward_mode = "dense"
shader = "default"
sim_backend = "auto"
record_dir = "demos"
traj_name = None
save_video = False
# env_id = "PushT-v1"
env_id = "PosPushT-v1"

def create_env(env_id):
    env = gym.make(
            env_id,
            obs_mode=obs_mode,
            control_mode="pd_joint_pos",
            render_mode=render_mode,
            reward_mode=reward_mode,
            shader_dir=shader,
            sim_backend=sim_backend
        )
    env = RecordEpisode(
        env,
        output_dir=osp.join(record_dir, env_id, "motionplanning"),
        trajectory_name=traj_name, save_video=save_video,
        source_type="motionplanning",
        source_desc="official motion planning solution from ManiSkill contributors",
        video_fps=30,
        save_on_reset=False
    )
    return env
env = create_env(env_id)
env.max_episode_steps = 99999
#%%
env.reset()
#%%
env.step(env.action_space.sample())
# %%    
# %%
