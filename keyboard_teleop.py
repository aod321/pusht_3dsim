#%% 
import os
os.environ["DISPLAY"] = ":0"
#%%
import sapien as sapien
import numpy as np
# %% Funcitons
# %%% create a box
def create_box(
    scene: sapien.Scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    name="",
) -> sapien.Entity:
	half_size = np.array(half_size)
	builder = scene.create_actor_builder()
	builder.add_box_collision(half_size=half_size)
	builder.add_box_visual(half_size=half_size,  material=color) 
	box = builder.build(name=name)
	box.set_pose(pose)
	return box
# %%% create a sphere
def create_sphere(
    scene: sapien.Scene,
    pose: sapien.Pose,
    radius,
    color=None,
    name="",
) -> sapien.Entity:
    builder = scene.create_actor_builder()
    builder.add_sphere_collision(radius=radius)
    builder.add_sphere_visual(radius=radius,  material=color) 
    sphere = builder.build(name=name)
    sphere.set_pose(pose)
    return sphere
# %% create a capsule
def create_capsule(
    scene: sapien.Scene,
    pose: sapien.Pose,
    radius,
    half_length,
    color=None,
    name="",
) -> sapien.Entity:
	builder = scene.create_actor_builder()
	builder.add_capsule_collision(radius=radius, half_length=half_length)
	builder.add_capsule_visual(radius=radius,  half_length=half_length, material=color) 
	capsule = builder.build(name=name)
	capsule.set_pose(pose)
	return capsule
# %%
def create_table(
    scene: sapien.Scene,
    pose: sapien.Pose,
    size,
    height,
    thickness=0.1,
    color=(0.8, 0.6, 0.4),
    name="table",
) -> sapien.Entity:
    """Create a table (a collection of collision and visual shapes)."""
    builder = scene.create_actor_builder()

    # Tabletop
    tabletop_pose = sapien.Pose(
        [0.0, 0.0, -thickness / 2]
    )  # Make the top surface's z equal to 0
    tabletop_half_size = [size / 2, size / 2, thickness / 2]
    builder.add_box_collision(pose=tabletop_pose, half_size=tabletop_half_size)
    builder.add_box_visual(
        pose=tabletop_pose, half_size=tabletop_half_size, material=color
    )

    # Table legs (x4)
    for i in [-1, 1]:
        for j in [-1, 1]:
            x = i * (size - thickness) / 2
            y = j * (size - thickness) / 2
            table_leg_pose = sapien.Pose([x, y, -height / 2])
            table_leg_half_size = [thickness / 2, thickness / 2, height / 2]
            builder.add_box_collision(
                pose=table_leg_pose, half_size=table_leg_half_size
            )
            builder.add_box_visual(
                pose=table_leg_pose, half_size=table_leg_half_size, material=color
            )

    table = builder.build(name=name)
    table.set_pose(pose)
    return table
# %% 
def main():
    scene = sapien.Scene()
    scene.set_timestep(1 / 100.0)  # Set the simulation frequency
    scene.add_ground(altitude=0)  # Add a ground
    box = create_box(
        scene=scene,
        pose=sapien.Pose([0, 0, 1+0.5]),
        half_size=[0.05, 0.05, 0.05],
        color=[1.0, 0, 0],
        name="box",
    )
    sphere = create_sphere(
        scene=scene,
        pose=sapien.Pose([0, -0.2, 1+0.5]),
        radius=0.05,
        color=[0, 1.0, 0],
        name="sphere",
    )
    capsule = create_capsule(
        scene=scene,
        pose=sapien.Pose([0, 0.2, 1+0.5]),
        radius=0.05,
        half_length=0.1,
        color=[0, 0, 1.0],
        name="capsule",
    )
    table = create_table(
        scene=scene,
        pose=sapien.Pose([0, 0, 1]),
        size=1,
        height=1,
        thickness=0.1,
        color=[0.8, 0.6, 0.4],
        name="table",
    )
    
    # create a banana
    builder = scene.create_actor_builder()
    builder.add_convex_collision_from_file(
        filename="/home/yinzi/collision.obj"
    )
    builder.add_visual_from_file(filename="/home/yinzi/visual.glb")
    mesh = builder.build(name="mesh")
    mesh.set_pose(sapien.Pose(p=[-0.2, 0, 1.0 + 0.05]))

    # create a robot, panda arm
     # robot
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load('/home/yinzi/custom_libs/SAPIEN/assets/robot/panda/panda.urdf')
    robot.set_name('panda')
    robot.set_root_pose(sapien.Pose([-0.5, 0, 1.5]))
    robot.set_qpos(np.array([0, 0.19634954084936207, 0.0, -2.617993877991494,
                          0.0, 2.941592653589793, 0.7853981633974483, 0, 0]))
    for l in robot.links:
        l.disable_gravity = True

    # Add some lights so that you can observe the scene
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = scene.create_viewer()  # Create a viewer (window)
    # The coordinate frame in Sapien is: x(forward), y(left), z(upward)
    # The principle axis of the camera is the x-axis
    viewer.set_camera_xyz(x=-2, y=0, z=2.5)
    # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
    # The camera now looks at the origin
    viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 2), y=0)
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

    position = 0.0  # position target of joints
    velocity = 0.0  # velocity target of joints
    steps = 0
    last_step = -4
    while not viewer.closed:  # Press key q to quit
        if steps - last_step < 4:
            pass  # prevent multiple changes for one key press
        if viewer.window.key_down('i'):  # accelerate
            print("i")
        elif viewer.window.key_down('k'):  # brake
            print("k")
        elif viewer.window.key_down('j'):  # left turn
            print("j")
        elif viewer.window.key_down('l'):  # right turn
            print("l")
        elif viewer.window.key_down('r'):  # reset
            print("r")
        steps += 1      
        scene.step()  # Simulate the world
        scene.update_render()  # Update the world to the renderer
        viewer.render()

if __name__ == "__main__":
    main()
# %%
