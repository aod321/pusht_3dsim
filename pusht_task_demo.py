#%%
import torch
import sapien
import numpy as np
from mani_skill.envs.tasks import PushTEnv
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
#---
# %%
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
    builder.disable_gravity = True
    random_name = str(uuid4())
    builder.build(name=random_name)
#%%

env = PushTEnv(render_mode='human')
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
steps = 0
last_step = -4
# create_maker_point(env.agent.tcp.pose.sp, env)
# maker = env.scene.actors['maker']
# maker.disable_gravity = True
goal_pose = sapien.Pose(p=env.agent.tcp.pose.sp.p, q=env.agent.tcp.pose.sp.q)
create_maker_point(goal_pose, env)
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
            reach_pose = sapien.Pose(p=env.goal_tee.pose.sp.p + np.array([-0.05, 0, 0]), q=env.agent.tcp.pose.sp.q)
            res = planner.move_to_pose_with_screw(reach_pose)
            print("up")
        elif viewer.window.key_down('k'):  # down
            goal_pose.p = goal_pose.p + np.array([0, 0.1, 0])
            print(goal_pose)
            create_maker_point(goal_pose, env)
            res = planner.move_to_pose_with_screw(goal_pose)
            print("down")
        elif viewer.window.key_down('j'):  # left turn
            goal_pose.p = goal_pose.p + np.array([-0.1, 0, 0])
            print(goal_pose)
            create_maker_point(goal_pose, env)
            res = planner.move_to_pose_with_screw(goal_pose)
            print("left")
        elif viewer.window.key_down('l'):  # right turn
            goal_pose.p = goal_pose.p + np.array([0.1, 0, 0])
            print(goal_pose)
            create_maker_point(goal_pose, env)
            res = planner.move_to_pose_with_screw(goal_pose)
            print("right")
        elif viewer.window.key_down('r'):  # reset
            env.reset()
            print("r")
    steps += 1      
    viewer.render()

    

# %%
