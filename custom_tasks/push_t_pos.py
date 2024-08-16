#%%
import sys
import os
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_PATH)

from typing import Any, Dict, Union
from mani_skill.envs.tasks import PushTEnv
from mani_skill.envs.tasks.tabletop.push_t import WhiteTableSceneBuilder
from mani_skill.utils.structs.types import Array
from mani_skill.utils.registration import register_env
from panda_stick_wrist_camera import WristCameraPandaStick
from mani_skill.agents.robots import PandaStick
import torch
import sapien
import numpy as np


class MyWhiteTableSceneBuilder(WhiteTableSceneBuilder):
    def initialize(self, env_idx: torch.Tensor):
        super().initialize(env_idx)
        b = len(env_idx)
        if self.env.robot_uids == "wrist_camera_panda_stick":
            qpos = np.array([0.662,0.212,0.086,-2.685,-.115,2.898,1.673,])
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
    

@register_env("PosPushT-v1", max_episode_steps=100)
class PosPushTEnv(PushTEnv):
    SUPPORTED_ROBOTS = ["wrist_camera_panda_stick"]
    agent: Union[WristCameraPandaStick]
    #T block design choices
    T_mass = 0.8
    T_dynamic_friction = 1
    T_static_friction = 1

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # reward for overlap of the tees
        reward = self.pseudo_render_intersection()
        return reward
    
    def __init__(self, *args, robot_uids="wrist_camera_panda_stick", robot_init_qpos_noise=0.02,**kwargs):
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)
    
    def _load_scene(self, options: dict):
        # have to put these parmaeters to device - defined before we had access to device
        # load scene is a convienent place for this one time operation
        self.ee_starting_pos2D = self.ee_starting_pos2D.to(self.device)
        self.ee_starting_pos3D = self.ee_starting_pos3D.to(self.device)

        # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = MyWhiteTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # returns 3d cad of create_tee - center of mass at (0,0,0)
        # cad Tee is upside down (both 3D tee and target)
        TARGET_RED = np.array([194, 19, 22, 255]) / 255 # same as mani_skill.utils.building.actors.common - goal target
        def create_tee(name="tee", target=False, base_color=TARGET_RED):
            # dimensions of boxes that make tee 
            # box2 is same as box1, except (3/4) the lenght, and rotated 90 degrees
            # these dimensions are an exact replica of the 3D tee model given by diffusion policy: https://cad.onshape.com/documents/f1140134e38f6ed6902648d5/w/a78cf81827600e4ff4058d03/e/f35f57fb7589f72e05c76caf
            box1_half_w = 0.2/2
            box1_half_h = 0.05/2
            half_thickness = 0.04/2 if not target else 1e-4

            # we have to center tee at its com so rotations are applied to com
            # vertical block is (3/4) size of horizontal block, so
            # center of mass is (1*com_horiz + (3/4)*com_vert) / (1+(3/4))
            # # center of mass is (1*(0,0)) + (3/4)*(0,(.025+.15)/2)) / (1+(3/4)) = (0,0.0375)
            com_y = 0.0375
            
            builder = self.scene.create_actor_builder()
            first_block_pose = sapien.Pose([0., 0.-com_y, 0.])
            first_block_size = [box1_half_w, box1_half_h, half_thickness]
            if not target:
                builder._mass = self.T_mass
                tee_material = sapien.pysapien.physx.PhysxMaterial(
                    static_friction=self.T_dynamic_friction, 
                    dynamic_friction=self.T_static_friction, 
                    restitution=0
                )
                builder.add_box_collision(pose=first_block_pose, half_size=first_block_size, material=tee_material)
                #builder.add_box_collision(pose=first_block_pose, half_size=first_block_size)
            builder.add_box_visual(pose=first_block_pose, half_size=first_block_size, material=sapien.render.RenderMaterial(
                base_color=base_color,
            ),)

            # for the second block (vertical part), we translate y by 4*(box1_half_h)-com_y to align flush with horizontal block
            # note that the cad model tee made here is upside down
            second_block_pose = sapien.Pose([0., 4*(box1_half_h)-com_y, 0.])
            second_block_size = [box1_half_h, (3/4)*(box1_half_w), half_thickness]
            if not target:
                builder.add_box_collision(pose=second_block_pose, half_size=second_block_size,material=tee_material)
                #builder.add_box_collision(pose=second_block_pose, half_size=second_block_size)
            builder.add_box_visual(pose=second_block_pose, half_size=second_block_size, material=sapien.render.RenderMaterial(
                base_color=base_color,
            ),)
            if not target:
                return builder.build(name=name)
            else: return builder.build_kinematic(name=name)

        self.tee = create_tee(name="Tee", target=False)
        self.goal_tee = create_tee(name="goal_Tee", target=True, base_color=np.array([128,128,128,255])/255)

        # adding end-effector end-episode goal position
        builder = self.scene.create_actor_builder()
        builder.add_cylinder_visual(
            radius=0.02,
            half_length=1e-4,
            material=sapien.render.RenderMaterial(base_color=np.array([128, 128, 128, 255]) / 255),
        )
        self.ee_goal_pos = builder.build_kinematic(name="goal_ee")

        # Rest of function is setting up for Custom 2D "Pseudo-Rendering" function below
        res = 64
        uv_half_width = 0.15
        self.uv_half_width = uv_half_width
        self.res = res
        oned_grid = (torch.arange(res, dtype=torch.float32).view(1,res).repeat(res,1) - (res/2))
        self.uv_grid = (torch.cat([oned_grid.unsqueeze(0), (-1*oned_grid.T).unsqueeze(0)], dim=0) + 0.5) / ((res/2)/uv_half_width)
        self.uv_grid = self.uv_grid.to(self.device)
        self.homo_uv = torch.cat([self.uv_grid, torch.ones_like(self.uv_grid[0]).unsqueeze(0)], dim=0)
        
        # tee render
        # tee is made of two different boxes, and then translated by center of mass
        self.center_of_mass = (0,0.0375) #in frame of upside tee with center of horizontal box (add cetner of mass to get to real tee frame)
        box1 = torch.tensor([[-0.1, 0.025], [0.1, 0.025], [-0.1, -0.025], [0.1, -0.025]]) 
        box2 = torch.tensor([[-0.025, 0.175], [0.025, 0.175], [-0.025, 0.025], [0.025, 0.025]])
        box1[:, 1] -= self.center_of_mass[1]
        box2[:, 1] -= self.center_of_mass[1]

        #convert tee boxes to indices
        box1 *= ((res/2)/uv_half_width)
        box1 += (res/2)

        box2 *= ((res/2)/uv_half_width)
        box2 += (res/2)

        box1 = box1.long()
        box2 = box2.long()

        self.tee_render = torch.zeros(res,res)
        # image map has flipped x and y, set values in transpose to undo
        self.tee_render.T[box1[0,0]:box1[1,0], box1[2,1]:box1[0,1]] = 1
        self.tee_render.T[box2[0,0]:box2[1,0], box2[2,1]:box2[0,1]] = 1
        # image map y is flipped of xy plane, flip to unflip
        self.tee_render = self.tee_render.flip(0).to(self.device)
        
        goal_fake_quat = torch.tensor([(torch.tensor([self.goal_z_rot])/2).cos(),0,0,0.0]).unsqueeze(0)
        zrot = self.quat_to_zrot(goal_fake_quat).squeeze(0) # 3x3 rot matrix for goal to world transform
        goal_trans = torch.eye(3)
        goal_trans[:2,:2] = zrot[:2,:2]
        goal_trans[0:2, 2] = self.goal_offset
        self.world_to_goal_trans = torch.linalg.inv(goal_trans).to(self.device) # this is just a 3x3 matrix (2d homogenious transform)

# %%
if __name__ == "__main__":
    import gymnasium as gym
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
                control_mode="pd_ee_delta_pose",
                render_mode=render_mode,
                reward_mode=reward_mode,
                shader_dir=shader,
                sim_backend=sim_backend
            )

        return env
    env = create_env(env_id)
    env.max_episode_steps = 99999
    #%%
    obs = env.reset()
    viewer = env.render()
    # %%
    done = False
    while not done:
        delta_ee_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Diagonal and single direction movement
        if viewer.window.key_down("i"):  # forward
            delta_ee_action[0] -= 0.1
        if viewer.window.key_down("k"):  # backward
            delta_ee_action[0] += 0.1
        if viewer.window.key_down("j"):  # left
            delta_ee_action[1] -= 0.1
        if viewer.window.key_down("l"):  # right
            delta_ee_action[1] += 0.1
        if viewer.window.key_down("e"):  # end episode
            env.reset()
            break
        obs,reward,done,_,info = env.step(delta_ee_action)
        print(f"reward:{reward}")
        viewer.render()
