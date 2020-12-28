import open3d as o3d
from open3d.open3d.geometry import create_rgbd_image_from_color_and_depth
import matplotlib.pyplot as plt
from src.utils.io import export_pointcloud
import numpy as np
import math

rgb_path = "/home/slionar/00_eth/mt/rgbd_bedroom/bedroom/image/"
depth_path = "/home/slionar/00_eth/mt/rgbd_bedroom/bedroom/depth/"
pose_path = "/home/slionar/00_eth/mt/pose_bedroom/bedroom.log"

color_raw = o3d.io.read_image(rgb_path+"011000.jpg")
depth_raw = o3d.io.read_image(depth_path+"011000.png")
rgbd_image = create_rgbd_image_from_color_and_depth(color_raw, depth_raw)

"""
plt.subplot(1, 2, 1)
plt.title('Redwood grayscale image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('Redwood depth image')
plt.imshow(rgbd_image.depth)
plt.show()
"""

pcd = o3d.geometry.create_point_cloud_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

#pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
#o3d.visualization.draw_geometries([pcd])

# Camera to world frame
pose_path = "/home/slionar/00_eth/mt/pose_bedroom/bedroom.log"
f = open(pose_path, "r")
log = f.read()
f.close()
log_list = log.split("\n")

n = 11000
begin = 5*n + 1
end = begin + 4

camera_pose = log_list[begin:end]
camera_pose[0] = camera_pose[0].split(" ")
camera_pose[1] = camera_pose[1].split(" ")
camera_pose[2] = camera_pose[2].split(" ")
camera_pose[3] = camera_pose[3].split(" ")

camera_pose_np = np.array(camera_pose,dtype=float)

pcl = np.asarray(pcd.points).T # 3 x N
N = pcl.shape[1]
one = np.ones((1, N))

pcl_h = np.vstack((pcl, one))

pcl_h_world = np.matmul(camera_pose_np, pcl_h)

# Align
theta = -26.6
cos_theta = math.cos(math.radians(theta))
sin_theta = math.sin(math.radians(theta))

alpha = -10
cos_alpha = math.cos(math.radians(alpha))
sin_alpha = math.sin(math.radians(alpha))

beta = -2
cos_beta = math.cos(math.radians(beta))
sin_beta = math.sin(math.radians(beta))

flip = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

rot_1 = np.array([[1, 0, 0, 0],
               [0, cos_theta, -sin_theta, 0],
               [0, sin_theta, cos_theta, 0],
               [0, 0, 0, 1]])

rot_2 = np.array([[cos_alpha, 0, sin_alpha, 0],
               [0, 1, 0, 0],
               [-sin_alpha, 0, cos_alpha, 0],
               [0, 0, 0, 1]])

rot_3 = np.array([[cos_beta, -sin_beta, 0, 0],
               [sin_beta, cos_beta, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])

shift = np.array([[1, 0, 0, 0],
               [0, 1, 0, 3],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])

align_matrix = np.matmul(rot_1, flip)
align_matrix = np.matmul(rot_2, align_matrix)
align_matrix = np.matmul(rot_3, align_matrix)
align_matrix = np.matmul(shift, align_matrix)

pcl_h_world = np.matmul(align_matrix, pcl_h_world)
pcl_world = pcl_h_world[:3].T

# Export Point cloud
export_pointcloud(pcl_world, 'pointcloud1.ply')
dtype = np.float16
np.savez('pointcloud.npz', points=pcl.astype(dtype), normals=pcl.astype(dtype))



