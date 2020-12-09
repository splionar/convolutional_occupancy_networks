import open3d as o3d
import numpy as np
from os.path import join, exists, isdir, exists

print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("data/Matterport3D_processed/17DRP5sb8fy/pointcloud.ply")
pcd = o3d.io.read_point_cloud("data/bedroom_fragment/fragment_bedroom/bedroom/mesh_96.ply")

print(pcd)
print(np.asarray(pcd.points))

pcd_np = np.load("data/Matterport3D_processed/17DRP5sb8fy/pointcloud.npz")

pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
pcd.transform([[1, 0, 0, 0], [0, 0.5, 0.5, 0], [0, -0.5, 0.5, 0], [0, 0, 0, 1]])
pcl = np.asarray(pcd.points)
o3d.visualization.draw_geometries([pcd])


from src.utils.io import export_pointcloud
from os.path import join, exists, isdir, exists



out_path = 'data/test'
scene_name = 'a'
outfile = join(out_path, scene_name)
out_file = join(outfile, 'pointcloud.npz')
dtype = np.float16
np.savez(out_file, points=pcl.astype(dtype), normals=pcl.astype(dtype))

# save surface points
out_file = join(outfile, 'pointcloud.npz')
np.savez(out_file, points=pcl.astype(dtype), normals=normals.astype(dtype))
export_pointcloud(pcl, join(outfile, 'pointcloud.ply'))