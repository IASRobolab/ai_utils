import pdb
from camera_utils.cameras.IntelRealsense import IntelRealsense
from ai_utils.MidasInference import MidasInference

import numpy as np
import open3d as o3d

import select
import sys


camera = IntelRealsense()
midas = MidasInference("/home/frollo/Desktop/weights/dpt_beit_large_512.pt", display_img=True, concatenate=True)

intr = camera.get_intrinsics()
fx = intr["fx"]
fy = intr["fy"]

f = (fx+fy)/2

pcd_list = []


# function used to stop loop functions
def stop_loop(stop_entry):
    '''
    Used to quit an infinite loop with a char/string entry
    '''
    rlist = select.select([sys.stdin], [], [], 0.001)[0]
    if rlist and sys.stdin.readline().find(stop_entry) != -1:
        return True
    return False

# while not stop_loop('q'):
# for i in range(5):

frame, depth_real = camera.get_aligned_frames()

idepth = midas.get_idepth_image(frame)

idepth = idepth - np.amin(idepth)
idepth /= np.amax(idepth)

depth = f / (idepth)
depth[depth >= 15 * f] = np.inf

depth = (depth / 3.5)
depth = depth + depth_real[int(depth_real.shape[0]/2), int(depth_real.shape[1]/2)] - depth[int(depth.shape[0]/2), int(depth.shape[1]/2)]

print(depth[int(depth.shape[0]/2), int(depth.shape[1]/2)])
print(depth_real[int(depth_real.shape[0]/2), int(depth_real.shape[1]/2)])
print(depth_real[int(depth_real.shape[0]/2), int(depth_real.shape[1]/2)]-depth[int(depth.shape[0]/2), int(depth.shape[1]/2)])
print()

rgb = o3d.geometry.Image(frame)
depth = o3d.geometry.Image(depth)
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_trunc=3.0)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera.o3d_intr)

depth_real = o3d.geometry.Image(depth_real)
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth_real, depth_trunc=3.0)
pcd_real = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera.o3d_intr)

# pcd_display = o3d.geometry.PointCloud(pcd)
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
pcd_real.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd, pcd_real])


pcd.paint_uniform_color([1, 0, 0])
pcd_real.paint_uniform_color([0, 0, 1])

pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.2)
pcd_real, ind_real = pcd_real.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.0)

pcd_list.append(pcd_real)
pcd_list.append(pcd)
# print(depth[int(depth.shape[0]/2), int(depth.shape[1]/2)], end="\r")

o3d.visualization.draw_geometries([pcd for pcd in pcd_list])


