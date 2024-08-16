#%%
import os
import sys
import numpy as np
import os.path as osp
import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_PATH)
os.environ["CUSTOM_ASSET_DIR"] = os.path.join(PROJECT_PATH, "custom_assets")
from mani_skill.utils import common, gym_utils
from mani_skill.utils.visualization.misc import (
    images_to_video,
    put_info_on_image,
    tile_images,
)
from custom_tasks import PosPushTEnv

#%%
obs_mode = "rgb"
render_mode = "rgb_array"
reward_mode = "dense"
shader = "default"
sim_backend = "auto"
record_dir = "demos"
traj_name = None
save_video = True
env_id = "PosPushT-v1"

def create_env(env_id):
    env = gym.make(
            env_id,
            obs_mode=obs_mode,
            control_mode="pd_ee_delta_pose",
            render_mode=render_mode,
            reward_mode=reward_mode,
            shader_dir=shader,
            sim_backend=sim_backend
        )
    env = RecordEpisode(
        env,
        output_dir=osp.join(record_dir, env_id, "motionplanning"),
        trajectory_name=traj_name, 
        save_video=save_video,
        save_trajectory=True,
        source_type="motionplanning",
        source_desc="official motion planning solution from ManiSkill contributors",
        video_fps=30,
        save_on_reset=True,
        clean_on_close=False,
    )
    return env
env = create_env(env_id)
env.max_episode_steps = 1000
#%%
obs = env.reset()
viewer = env.render_human()
# %%
done = False
while not done:
    viewer = env.render_human()
    delta_ee_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    action_taken = False

    # Diagonal and single direction movement
    if viewer.window.key_down("i"):  # forward
        delta_ee_action[0] -= 0.1
        action_taken = True
    if viewer.window.key_down("k"):  # backward
        delta_ee_action[0] += 0.1
        action_taken = True
    if viewer.window.key_down("j"):  # left
        delta_ee_action[1] -= 0.1
        action_taken = True
    if viewer.window.key_down("l"):  # right
        delta_ee_action[1] += 0.1
        action_taken = True

    if viewer.window.key_down("e"):  # end episode
        env.reset()
        env.close()
        break

    if action_taken:  # Only step if an action was taken
        obs, reward, done, _, info = env.step(delta_ee_action)
        print(f"reward: {reward}")

# %%
