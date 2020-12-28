import open3d as o3d
import numpy as np
from os.path import join, exists, isdir, exists
import math
from src.utils.io import export_pointcloud
import matplotlib.pyplot as plt

print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("data/Matterport3D_processed/17DRP5sb8fy/pointcloud.ply")
pcd = o3d.io.read_point_cloud("data/bedroom_fragment/fragment_bedroom/bedroom/mesh_96.ply")
pcd = o3d.io.read_point_cloud("data/ours_bedroom/bedroom.ply")


print(pcd)
print(np.asarray(pcd.points))

pcd_np = np.load("data/Matterport3D_processed/17DRP5sb8fy/pointcloud.npz")

pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
theta = -26.6
cos_theta = math.cos(math.radians(theta))
sin_theta = math.sin(math.radians(theta))
pcd.transform([[1, 0, 0, 0],
               [0, cos_theta, -sin_theta, 0],
               [0, sin_theta, cos_theta, 0],
               [0, 0, 0, 1]])

#z-axis
alpha = -10
cos_alpha = math.cos(math.radians(alpha))
sin_alpha = math.sin(math.radians(alpha))
pcd.transform([[cos_alpha, 0, sin_alpha, 0],
               [0, 1, 0, 0],
               [-sin_alpha, 0, cos_alpha, 0],
               [0, 0, 0, 1]])

beta = -2
cos_beta = math.cos(math.radians(beta))
sin_beta = math.sin(math.radians(beta))
pcd.transform([[cos_beta, -sin_beta, 0, 0],
               [sin_beta, cos_beta, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])

#shift
pcd.transform([[1, 0, 0, 0],
               [0, 1, 0, 3],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])

pcl = np.asarray(pcd.points)
o3d.visualization.draw_geometries([pcd])

pcl_filter = pcl[pcl[:,1]> -0.05]
pcl_filter = pcl_filter[pcl_filter[:,0] > -1.85]
pcl_filter = pcl_filter[pcl_filter[:,2] > -0.93]
pcl_filter = pcl_filter[pcl_filter[:,2] < 4.85]
plt.hist(pcl_filter[:,2], bins = 90)
plt.show()

#pcd.transform([[1, 0, 0, 0], [0, 0.5, 0.5, 0], [0, -0.5, 0.5, 0], [0, 0, 0, 1]])


from os.path import join, exists, isdir, exists



out_path = 'data/test'
scene_name = 'a'
outfile = join(out_path, scene_name)
out_file = join(outfile, 'pointcloud.npz')
dtype = np.float16
np.savez('pointcloud.npz', points=pcl_filter.astype(dtype), normals=pcl_filter.astype(dtype))
np.savez('pointcloud.npz', points=pcl_filter.astype(dtype))

# save surface points
out_file = join(outfile, 'pointcloud.npz')
np.savez(out_file, points=pcl.astype(dtype), normals=normals.astype(dtype))
export_pointcloud(pcl_filter, 'pointcloud_align.ply')
export_pointcloud(pcl_filter, 'pointcloud_align_filter.ply')



####### Try create occlusion ######
