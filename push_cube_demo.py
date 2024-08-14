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

obs_mode = "none"
render_mode = "human"
reward_mode = "dense"
shader = "default"
sim_backend = "auto"
record_dir = "demos"
traj_name = None
save_video = False
env_id = "PushCube-v1"

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
# rollout and visualize the environment
obs, info = env.reset()
viewer = env.render()
planner = PandaArmMotionPlanningSolver(
    env,
    debug=False,
    vis=False,
    base_pose=env.unwrapped.agent.robot.pose,
    visualize_target_grasp_pose=False,
    print_env_info=False,
)
#%%
steps = 0
last_step = -4
# create_maker_point(env.agent.tcp.pose.sp, env)
# maker = env.scene.actors['maker']
# maker.disable_gravity = True
goal_pose = sapien.Pose(p=env.agent.tcp.pose.sp.p, q=env.agent.tcp.pose.sp.q)
uenv = env.unwrapped
# maker_pose = sapien.Pose(p=np.array([goal_pose.p[0], goal_pose.p[1], 0.]), q=goal_pose.q)
# %%
# def move_actor(actor, pose):
    # actor.set_pose(pose)
    # actor.set_linear_velocity(np.zeros(3))
    # actor.set_angular_velocity(np.zeros(3))
#%%
while not viewer.closed:  # Press key q to quit
    if steps - last_step < 4:
        pass  # prevent multiple changes for one key press
    else:
        last_step = steps
        if viewer.window.key_down('i'):  # up
            # goal_pose.p = goal_pose.p + np.array([0, -0.1, 0])
            # print(goal_pose)
            # create_maker_point(goal_pose, env)
            planner.close_gripper()
            reach_pose = sapien.Pose(p=uenv.obj.pose.sp.p + np.array([0, -0.05, 0]), q=uenv.agent.tcp.pose.sp.q)
            # reach_pose = sapien.Pose(p=env.goal_tee.pose.sp.p + np.array([-0.05, 0, 0]), q=env.agent.tcp.pose.sp.q)
            res = planner.move_to_pose_with_screw(reach_pose)
            print("up")
        elif viewer.window.key_down('k'):  # down
            goal_pose.p = goal_pose.p + np.array([0, 0.05, 0])
            print(goal_pose)
            res = planner.move_to_pose_with_screw(goal_pose)
            print("down")
        elif viewer.window.key_down('j'):  # left turn
            goal_pose.p = goal_pose.p + np.array([-0.05, 0, 0])
            print(goal_pose)
            res = planner.move_to_pose_with_screw(goal_pose)
            print("left")
        elif viewer.window.key_down('l'):  # right turn
            goal_pose.p = goal_pose.p + np.array([0.05, 0, 0])
            print(goal_pose)
            res = planner.move_to_pose_with_screw(goal_pose)
            print("right")
        elif viewer.window.key_down('r'):  # reset
            env.reset()
            print("r")
    steps += 1      
    viewer.render()

    

# %%
