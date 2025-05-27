import os
from graspclutter6dAPI import GraspClutter6D
from graspnetAPI import GraspNet, Grasp, GraspGroup # Compatible with GraspNet
import open3d as o3d
import cv2
import numpy as np

# GraspClutter6D example for loading grasp for a scene.
####################################################################
####################################################################

sceneId = 1
annId = 3
camera = 'realsense-d415' # 'realsense-d415', 'realsense-d435', 'azure-kinect', 'zivid'

if 'GC6D_ROOT' not in os.environ:
    print('Please set the environment variable GC6D_ROOT (e.g. export GC6D_ROOT=/path/to/GraspClutter6D)')
    exit(0)
else:
    gc6d_root = os.environ['GC6D_ROOT']

# initialize a GraspNet instance  
g = GraspClutter6D(gc6d_root, camera=camera, split='train')

# load grasps
_6d_grasp = g.loadGrasp(sceneId = sceneId, annId = annId, format = '6d', camera = camera, fric_coef_thresh = 0.2)
print('6d grasp:\n{}'.format(_6d_grasp))

# _6d_grasp is an GraspGroup instance defined in grasp.py
print('_6d_grasp:\n{}'.format(_6d_grasp))

# index
grasp = _6d_grasp[0]
print('_6d_grasp[0](grasp):\n{}'.format(grasp))

# slice
print('_6d_grasp[0:2]:\n{}'.format(_6d_grasp[0:2]))
print('_6d_grasp[[0,1]]:\n{}'.format(_6d_grasp[[0,1]]))

# grasp is a Grasp instance defined in grasp.py
# access and set properties
print('grasp.translation={}'.format(grasp.translation))
grasp.translation = np.array([1.0, 2.0, 3.0])
print('After modification, grasp.translation={}'.format(grasp.translation))
print('grasp.rotation_matrix={}'.format(grasp.rotation_matrix))
grasp.rotation_matrix = np.eye(3).reshape((9))
print('After modification, grasp.rotation_matrix={}'.format(grasp.rotation_matrix))
print('grasp.width={}, height:{}, depth:{}, score:{}'.format(grasp.width, grasp.height, grasp.depth, grasp.score))
print('More operation on Grasp and GraspGroup can be seen in the API document')

# transform grasp
g = Grasp() # simple Grasp
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)

# Grasp before transformation
o3d.visualization.draw_geometries([g.to_open3d_geometry(), frame])
g.translation = np.array((0,0,0.01))

# setup a transformation matrix
T = np.eye(4)
T[:3,3] = np.array((0.01, 0.02, 0.03))
T[:3,:3] = np.array([[0,0,1.0],[1,0,0],[0,1,0]])
g.transform(T)

# Grasp after transformation
o3d.visualization.draw_geometries([g.to_open3d_geometry(), frame])

g1 = Grasp()
gg = GraspGroup()
gg.add(g)
gg.add(g1)

# GraspGroup before transformation
o3d.visualization.draw_geometries([*gg.to_open3d_geometry_list(), frame])

gg.transform(T)

# GraspGroup after transformation
o3d.visualization.draw_geometries([*gg.to_open3d_geometry_list(), frame])