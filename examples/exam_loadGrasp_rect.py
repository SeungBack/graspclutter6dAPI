import os
from graspclutter6dAPI import GraspClutter6D
import cv2

sceneId = 1
annId = 3 # 1~13
camera = 'realsense-d415' # 'realsense-d415', 'realsense-d435', 'azure-kinect', 'zivid'

# GraspClutter6D example for loading rectangle grasp for a scene.
# Rectangle grasps are converted on-the-fly from 6d grasps using the per-scene intrinsics.

if 'GC6D_ROOT' not in os.environ:
    print('Please set the environment variable GC6D_ROOT (e.g. export GC6D_ROOT=/path/to/GraspClutter6D)')
    exit(0)
else:
    gc6d_root = os.environ['GC6D_ROOT']

# initialize a GraspClutter6D instance
g = GraspClutter6D(gc6d_root, camera=camera, split='train')

# load rectangle grasps
rect_grasp = g.loadGrasp(sceneId = sceneId, annId = annId, format = 'rect', camera = camera, fric_coef_thresh = 0.2)
print('rectangle grasp:\n{}'.format(rect_grasp))

# visualize the rectangle grasps using opencv
bgr = g.loadBGR(sceneId = sceneId, annId = annId, camera = camera)
img = rect_grasp.to_opencv_image(bgr, numGrasp = 20)
cv2.imwrite('rectangle_grasps.png', img)

