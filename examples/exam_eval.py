import os
from graspclutter6dAPI import GraspClutter6DEval
import open3d as o3d
import cv2
import numpy as np


if 'GC6D_ROOT' not in os.environ:
    print('Please set the environment variable GC6D_ROOT (e.g. export GC6D_ROOT=/path/to/GraspClutter6D)')
    exit(0)
else:
    gc6d_root = os.environ['GC6D_ROOT']

sceneId = 121
camera = 'realsense-d415' # 'realsense-d415', 'realsense-d435', 'azure-kinect', 'zivid'
dump_folder = './dump' # folder that stores the predicted grasps

ge = GraspClutter6DEval(root = gc6d_root, camera = camera, split = 'test')

# eval a single scene
print('Evaluating scene:{}, camera:{}'.format(sceneId, camera))
acc = ge.eval_scene(scene_id = sceneId, dump_folder = dump_folder)
np_acc = np.array(acc)
print('mean accuracy:{}'.format(np.mean(np_acc)))

# # eval all data
res, ap = ge.eval_all(dump_folder, proc = 4)
