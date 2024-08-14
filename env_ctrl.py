# %%
import sapien
import numpy as np
import os.path as osp
import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode
#---
import numpy as np
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

record_di = "demos"
traj_name = None
save_video = False

env = gym.make(
        "PushT",
        obs_mode=None,
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
        shader_dir="default",
        sim_backend="auto"
    )
env = RecordEpisode(
        env,
        output_dir=osp.join(record_di, "PushT", "motionplanning"),
        trajectory_name=traj_name, save_video=save_video,
        source_type="motionplanning",
        source_desc="official motion planning solution from ManiSkill contributors",
        video_fps=30,
        save_on_reset=False
    )

# %%
import cv2

def map_to_2d_plane(tcp_position, img_size):
    normalized_position = ((tcp_position + 1)/2 * img_size).astype(int)
    return normalized_position

def visualize_agent_position(tcp_position, img_size=512):
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    tcp_position = tcp_position.flatten()
    pos_2d = map_to_2d_plane(tcp_position[:2], img_size)
    print(pos_2d)
    cv2.circle(img, tuple(pos_2d), radius=5, color=(0, 0, 255), thickness=-1)
    cv2.imshow("Agent Position", img)
    cv2.waitKey(1)
# %%
while True:
    env.reset()
    planner = PandaStickMotionPlanningSolver(
        env,
        debug=False,
        vis=True,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )
    uenv = env.unwrapped
    target_pose = sapien.Pose(p=uenv.agent.tcp.pose.sp.p + np.array([-0.05, 0, 0]), q=uenv.agent.tcp.pose.sp.q)
    res = planner.move_to_pose_with_screw(target_pose)
    print(res)
    # Visualize agent's TCP position
    tcp_position = uenv.agent.tcp.pose.p
    tcp_position = tcp_position.cpu().numpy()  # Convert to numpy array

    visualize_agent_position(tcp_position)
    planner.close()
    # uenv.agent.tcp.pose.raw_pose
    # tensor([[-0.3277,  0.3033,  0.0240,  0.0055,  0.9997,  0.0208,  0.0092]])

env.close()