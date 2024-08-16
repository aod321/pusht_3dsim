# %%
import h5py
traj_path = "/nvmessd/yinzi/pusht_3dsim/demos/PosPushT-v1/motionplanning/20240816_124843.h5"
ori_h5_file = h5py.File(traj_path, "r")
# %%
a =dict(ori_h5_file)

# %%
