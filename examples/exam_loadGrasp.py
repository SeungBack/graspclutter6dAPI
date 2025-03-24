import os
from graspclutter6dAPI import GraspClutter6D
import open3d as o3d
import cv2

sceneId = 1
annId = 3 # 1~13
camera = 'realsense-d415' # 'realsense-d415', 'realsense-d435', 'azure-kinect', 'zivid'

# GraspClutter6D example for loading grasp for a scene.

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

# visualize the grasps using open3d
geometries = []
geometries.append(g.loadScenePointCloud(sceneId = sceneId, annId = annId, camera = camera))
geometries += _6d_grasp.random_sample(numGrasp = 20).to_open3d_geometry_list()
o3d.visualization.draw_geometries(geometries)
