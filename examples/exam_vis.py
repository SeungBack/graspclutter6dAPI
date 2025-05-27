import os
from graspclutter6dAPI import GraspClutter6D

# GraspClutter6D example for visualization.

if 'GC6D_ROOT' not in os.environ:
    print('Please set the environment variable GC6D_ROOT (e.g. export GC6D_ROOT=/path/to/GraspClutter6D)')
    exit(0)
else:
    gc6d_root = os.environ['GC6D_ROOT']


# initialize a GraspNet instance  
g = GraspClutter6D(gc6d_root, camera='realsense-d415', split='train')

# show object grasps (image will be saved in save_fig folder). object id 1~200
g.showObjGrasp(objIds = 1, numGrasp=50, th=0.5, maxWidth=0.10, saveFolder='save_fig', show=True) 

# show 6d poses (scene id 0~999)
g.show6DPose(sceneIds = 1, show = True)

# show scene 6d grasps(You may need to wait several minutes)
g.showSceneGrasp(sceneId = 0, camera = 'zivid', annId = 2, format = '6d', numGrasp = 50, width=0.14)