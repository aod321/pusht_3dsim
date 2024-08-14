#%%
import torch
import sapien
import numpy as np
import os.path as osp
import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.envs.tasks import PushCubeEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver

    
class PandaStickMotionPlanningSolver(PandaArmMotionPlanningSolver):
    def follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][min(i, n_step - 1)]
                action = np.hstack([qpos, qvel])
            else:
                action = np.hstack([qpos])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info
#%%
obs_mode = "none"
render_mode = "human"
reward_mode = "dense"
shader = "default"
sim_backend = "auto"
record_dir = "demos"
traj_name = None
save_video = False
env_id = "PushT-v1"

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
# rollout and visualize the environment
obs, info = env.reset()
viewer = env.render()
planner = PandaStickMotionPlanningSolver(
    env,
    debug=False,
    vis=False,
    base_pose=env.unwrapped.agent.robot.pose,
    visualize_target_grasp_pose=False,
    print_env_info=False,
)
#%%
from uuid import uuid4

def create_maker_point(pose: sapien.Pose, env):
    scene = env.scene
    builder = scene.create_actor_builder()
    visual_width = 0.01
    color = [1.0, 1.0, 0.0]
    point = sapien.Pose(p=np.array([0.,0.,0.]), q=pose.q)
    point.set_p(p=np.array([pose.p[0],pose.p[1],0.]))
    builder.add_sphere_visual(
        pose=point,
        radius=visual_width,
        material=sapien.render.RenderMaterial(base_color=[*color[:3], 1])
    )
    builder.add_sphere_collision(
        pose=point,
        radius=visual_width
    )
    builder.disable_gravity = True
    random_name = str(uuid4())
    builder.build(name=random_name)
#%%
steps = 0
last_step = -4
uenv = env.unwrapped
goal_pose = sapien.Pose(p=env.agent.tcp.pose.sp.p, q=env.agent.tcp.pose.sp.q)
movement_speed = 0.01  # Movement speed variable
# movement_speed = 0.05  # Movement speed variable

#%%
while not viewer.closed:  # Press key q to quit
    # clear last terminal info
    new_goal_pose = sapien.Pose(p=env.agent.tcp.pose.sp.p, q=env.agent.tcp.pose.sp.q)
    if steps - last_step < 4:
        pass  # prevent multiple changes for one key press
    else:
        last_step = steps
        if viewer.window.key_down('i'):  # up
            new_goal_pose.p = new_goal_pose.p + np.array([-movement_speed, 0, 0])
            print(new_goal_pose)
            reach_pose = sapien.Pose(p=new_goal_pose.p, q=goal_pose.q)
            res = planner.move_to_pose_with_RRTConnect(reach_pose)
            if res != -1:
                #create_maker_point(reach_pose, env)
                obs, reward, terminated, truncated, info = res
                print(reward)
            print("up")
        elif viewer.window.key_down('k'):  # down
            new_goal_pose.p = new_goal_pose.p + np.array([movement_speed, 0, 0])
            print(new_goal_pose)
            reach_pose = sapien.Pose(p=new_goal_pose.p, q=goal_pose.q)
            res = planner.move_to_pose_with_RRTConnect(reach_pose)
            if res != -1:
                #create_maker_point(reach_pose, env)
                obs, reward, terminated, truncated, info = res
                print(reward)
            print("down")
        elif viewer.window.key_down('j'):  # left turn
            new_goal_pose.p = new_goal_pose.p + np.array([0, -movement_speed, 0])
            print(new_goal_pose)
            reach_pose = sapien.Pose(p=new_goal_pose.p, q=goal_pose.q)
            res = planner.move_to_pose_with_RRTConnect(reach_pose)
            if res != -1:
                #create_maker_point(reach_pose, env)
                obs, reward, terminated, truncated, info = res
                print(reward)
            try:
                obs, reward, terminated, truncated, info = res
                print(reward)
            except:
                print("Failed to reach the goal")
                print(res)
            print("left")
        elif viewer.window.key_down('l'):  # right turn
            new_goal_pose.p = new_goal_pose.p + np.array([0, movement_speed, 0])
            print(new_goal_pose)
            reach_pose = sapien.Pose(p=new_goal_pose.p, q=goal_pose.q)
            res = planner.move_to_pose_with_RRTConnect(reach_pose)
            if res != -1:
                #create_maker_point(reach_pose, env)
                obs, reward, terminated, truncated, info = res
                print(reward)
            print("right")
        elif viewer.window.key_down('u'):  # up-left
            new_goal_pose.p = new_goal_pose.p + np.array([-movement_speed, -movement_speed, 0])
            print(new_goal_pose)
            reach_pose = sapien.Pose(p=new_goal_pose.p, q=goal_pose.q)
            res = planner.move_to_pose_with_RRTConnect(reach_pose)
            if res != -1:
                #create_maker_point(reach_pose, env)
                obs, reward, terminated, truncated, info = res
                print(reward)
            print("up-left")
        elif viewer.window.key_down('o'):  # up-right
            new_goal_pose.p = new_goal_pose.p + np.array([-movement_speed, movement_speed, 0])
            print(new_goal_pose)
            reach_pose = sapien.Pose(p=new_goal_pose.p, q=goal_pose.q)
            res = planner.move_to_pose_with_RRTConnect(reach_pose)
            if res != -1:
                #create_maker_point(reach_pose, env)
                obs, reward, terminated, truncated, info = res
                print(reward)
            print("up-right")
        elif viewer.window.key_down('n'):  # down-left
            new_goal_pose.p = new_goal_pose.p + np.array([movement_speed, -movement_speed, 0])
            print(new_goal_pose)
            reach_pose = sapien.Pose(p=new_goal_pose.p, q=goal_pose.q)
            res = planner.move_to_pose_with_RRTConnect(reach_pose)
            if res != -1:
                #create_maker_point(reach_pose, env)
                obs, reward, terminated, truncated, info = res
                print(reward)
            print("down-left")
        elif viewer.window.key_down('m'):  # down-right
            new_goal_pose.p = new_goal_pose.p + np.array([movement_speed, movement_speed, 0])
            print(new_goal_pose)
            reach_pose = sapien.Pose(p=new_goal_pose.p, q=goal_pose.q)
            res = planner.move_to_pose_with_RRTConnect(reach_pose)
            if res != -1:
                #create_maker_point(reach_pose, env)
                obs, reward, terminated, truncated, info = res
                print(reward)
            print("down-right")
        elif viewer.window.key_down('r'):  # reset
            env.reset()
            print("reset")
    steps += 1      
    viewer.render()
# %%
