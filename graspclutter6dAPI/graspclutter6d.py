# Author: Seunghyeok Back (shback@kimm.re.kr)
# KIMM & GIST, Republic of Korea
# Base codes are from GraspNetAPI (hsfang, mhgou, cxwang), https://github.com/graspnet/graspnetAPI


# The following API functions are defined:
#  GraspNet             - GraspNet api class that loads GraspNet annotation file and prepare data structures.
#  checkDataCompleteness- Check the file completeness of the dataset.
#  getSceneIds          - Get scene ids that satisfy given filter conditions.
#  getObjIds            - Get obj ids that satisfy given filter conditions.
#  getDataIds           - Get data ids that satisfy given filter conditions.
#  loadBGR              - Load image in BGR format.
#  loadRGB              - Load image in RGB format.
#  loadDepth            - Load depth image.
#  loadMask             - Load the segmentation masks.
#  loadSceneModels      - Load object models in a scene.
#  loadScenePointCloud  - Load point cloud constructed by the depth and color image.
#  loadWorkSpace        - Load the workspace bounding box. (deprecated)
#  loadGraspLabels      - Load grasp labels with the specified object ids.
#  loadObjModels        - Load object 3d mesh model with the specified object ids.
#  loadObjTrimesh       - Load object 3d mesh in Trimesh format.
#  loadCollisionLabels  - Load collision labels with the specified scene ids.
#  loadGrasp            - Load grasp labels with the specified scene and annotation id.
#  loadData             - Load data path with the specified data ids.
#  showObjGrasp         - Save visualization of the grasp pose of specified object ids.
#  showSceneGrasp       - Save visualization of the grasp pose of specified scene ids.
#  show6DPose           - Save visualization of the 6d pose of specified scene ids, project obj models onto pointcloud
# Throughout the API "ann"=annotation, "obj"=object, and "img"=image.

import os
import numpy as np
from tqdm import tqdm
import open3d as o3d
import cv2
import trimesh
import json 

from .grasp import Grasp, GraspGroup, RectGraspGroup, RECT_GRASP_ARRAY_LEN
from .utils.utils import transform_points, parse_posevector
from .utils.xmlhandler import xmlReader

GRASP_HEIGHT = 0.02

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class GraspClutter6D():
    def __init__(self, root, camera='zivid', split='train'):
        '''

        GraspClutter6D main class.

        **input**:

        - camera: string of type of camera: "kinect" or "realsense"

        - split: string of type of split of dataset: "all", "train", "test"
        '''
        assert camera in ['realsense-d415', 'realsense-d435', 'azure-kinect', 'zivid', 'realsense', 'kinect'], 'camera should be realsense-d415/realsense-d435/azure-kinect/zivid'
        assert split in ['all', 'train', 'test'], 'split should be all/train/test'
        if camera == 'realsense':
            camera = 'realsense-d435' # for GraspNet-1Billion compatibility
        elif camera == 'kinect':
            camera = 'azure-kinect'
        self.root = root
        self.camera = camera
        self.split = split
        self.collisionLabels = {}

        if split == 'all':
            with open(os.path.join(self.root, 'split_info', 'grasp_train_scene_ids.json')) as f:
                self.sceneIds = [int(x) for x in json.load(f)]
            with open(os.path.join(self.root, 'split_info', 'grasp_test_scene_ids.json')) as f:
                self.sceneIds += [int(x) for x in json.load(f)]
        elif split == 'train':
            with open(os.path.join(self.root, 'split_info', 'grasp_train_scene_ids.json')) as f:
                self.sceneIds = [int(x) for x in json.load(f)]
        elif split == 'test':
            with open(os.path.join(self.root, 'split_info', 'grasp_test_scene_ids.json')) as f:
                self.sceneIds = [int(x) for x in json.load(f)]

        self.rgbPath = []
        self.depthPath = []
        self.segLabelPath = []
        self.metaPath = []
        self.rectLabelPath = []
        self.sceneName = []
        self.annId = []

        for i in tqdm(self.sceneIds, desc='Loading data path...'):
            for ann_id in range(13):
                img_num = 4*ann_id
                if self.camera == 'realsense-d415':
                    img_num += 1
                elif self.camera == 'realsense-d435':
                    img_num += 2
                elif self.camera == 'azure-kinect':
                    img_num += 3
                elif self.camera == 'zivid':
                    img_num += 4
                self.rgbPath.append(os.path.join(
                    root, 'scenes', str(i).zfill(6), 'rgb', str(img_num).zfill(6)+'.png'))
                self.depthPath.append(os.path.join(
                    root, 'scenes', str(i).zfill(6), 'depth', str(img_num).zfill(6)+'.png'))
                self.segLabelPath.append(os.path.join(
                    root, 'scenes', 'scene_'+str(i).zfill(4), camera, 'label', str(img_num).zfill(4)+'.png'))
                self.metaPath.append(os.path.join(
                    root, 'scenes', 'scene_'+str(i).zfill(4), camera, 'meta', str(img_num).zfill(4)+'.mat'))
                self.rectLabelPath.append(os.path.join(
                    root, 'scenes', 'scene_'+str(i).zfill(4), camera, 'rect', str(img_num).zfill(4)+'.npy'))
                self.sceneName.append('scene_'+str(i).zfill(4))
                self.annId.append(ann_id)
        self.objIds = self.getObjIds(self.sceneIds)


    def __len__(self):
        return len(self.depthPath)

    def checkDataCompleteness(self):
        '''
        Check whether the dataset files are complete.

        **Output:**

        - bool, True for complete, False for not complete.
        '''
        error_flag = False
        for objId in tqdm(range(1, 201), 'Checking Models'):
            if not os.path.exists(os.path.join(self.root, 'models_m', 'obj_%06d.ply' % objId)):
                error_flag = True
                print('No obj_%06d.ply For Object {}'.format(objId))
            if not os.path.exists(os.path.join(self.root, 'models_obj_m', 'obj_%06d.obj' % objId)):
                error_flag = True
                print('No obj_%06d.obj For Object {}'.format(objId))

        for objId in tqdm(range(1, 201), 'Checking Grasp Labels'):
            if not os.path.exists(os.path.join(self.root, 'grasp_label', 'obj_%06d_labels.npz' % objId)):
                error_flag = True
                print('No Grasp Label For Object {}'.format(objId))

        for sceneId in tqdm(self.sceneIds, 'Checking Collosion Labels'):
            if not os.path.exists(os.path.join(self.root, 'collision_label', '%06d.npz' % sceneId)):
                error_flag = True
                print('No Collision Labels For Scene {}'.format(sceneId))

        for split_file in ['grasp_train_scene_ids.json', 'grasp_test_scene_ids.json']:
            if not os.path.exists(os.path.join(self.root, 'split_info', split_file)):
                error_flag = True
                print('No {} For Split {}'.format(split_file, split_file.split('_')[1].split('.')[0]))

        for sceneId in tqdm(self.sceneIds, 'Checking Scene Datas'):
            sceneDir = os.path.join(self.root, 'scenes', '%06d' % sceneId)
            if not os.path.exists(os.path.join(sceneDir, 'scene_camera.json')):
                error_flag = True
                print('No scene_camera.json For Scene {}'.format(sceneId))
            if not os.path.exists(os.path.join(sceneDir, 'scene_gt.json')):
                error_flag = True
                print('No scene_gt.json For Scene {}'.format(sceneId))

            for imgId in range(1, 53):
                if not os.path.exists(os.path.join(sceneDir, 'rgb','%06d.png' % imgId)):
                    error_flag = True
                    print('No RGB Image For Scene {}, Image:{}'.format(sceneId, imgId))
                if not os.path.exists(os.path.join(sceneDir,'depth','%06d.png' % imgId)):
                    error_flag = True
                    print('No Depth Image For Scene {}, Image:{}'.format(sceneId, imgId))
                if not os.path.exists(os.path.join(sceneDir,'label','%06d.png' % imgId)):
                    error_flag = True
                    print('No Label Image For Scene {}, Image:{}'.format(sceneId, imgId))
        return not error_flag

    def getSceneIds(self, objIds=None):
        '''
        **Input:**

        - objIds: int or list of int of the object ids.

        **Output:**

        - a list of int of the scene ids that contains **all** the objects.
        '''
        if objIds is None:
            return self.sceneIds
        assert _isArrayLike(objIds) or isinstance(objIds, int), 'objIds must be integer or a list/numpy array of integers'
        objIds = objIds if _isArrayLike(objIds) else [objIds]
        sceneIds = []
        for i in self.sceneIds:
            f = open(os.path.join(self.root, 'scenes', 'scene_' + str(i).zfill(4), 'object_id_list.txt'))
            idxs = [int(line.strip()) for line in f.readlines()]
            check = all(item in idxs for item in objIds)
            if check:
                sceneIds.append(i)
        return sceneIds

    def getObjIds(self, sceneIds=None):
        '''
        **Input:**

        - sceneIds: int or list of int of the scene ids.

        **Output:**

        - a list of int of the object ids in the given scenes.
        '''
        # get object ids in the given scenes
        if sceneIds is None:
            return self.objIds
        assert _isArrayLike(sceneIds) or isinstance(sceneIds, int), 'sceneIds must be an integer or a list/numpy array of integers'
        sceneIds = sceneIds if _isArrayLike(sceneIds) else [sceneIds]
        objIds = []
        for i in sceneIds:
            f = open(os.path.join(self.root, 'split_info', 'obj_ids_per_scene.json'))
            objIds += json.load(f)[str(i)]
        return objIds

    def getDataIds(self, sceneIds=None):
        '''
        **Input:**

        - sceneIds:int or list of int of the scenes ids.

        **Output:**

        - a list of int of the data ids. Data could be accessed by calling self.loadData(ids).
        '''
        # get index for datapath that contains the given scenes
        if sceneIds is None:
            return list(range(len(self.sceneName)))
        ids = []
        indexPosList = []
        for i in sceneIds:
            indexPosList += [ j for j in range(0,len(self.sceneName),256) if self.sceneName[j] == 'scene_'+str(i).zfill(4) ]
        for idx in indexPosList:
            ids += list(range(idx, idx+256))
        return ids

    def loadGraspLabels(self, objIds=None):
        '''
        **Input:**

        - objIds: int or list of int of the object ids.

        **Output:**

        - a dict of grasplabels of each object. 
        '''
        # load object-level grasp labels of the given obj ids
        objIds = self.objIds if objIds is None else objIds
        assert _isArrayLike(objIds) or isinstance(objIds, int), 'objIds must be an integer or a list/numpy array of integers'
        objIds = objIds if _isArrayLike(objIds) else [objIds]
        graspLabels = {}
        for i in tqdm(objIds, desc='Loading grasping labels...'):
            file = np.load(os.path.join(self.root, 'grasp_label', 'obj_{}_labels.npz'.format(str(i).zfill(6))))
            graspLabels[i] = (file['points'].astype(np.float32), file['offsets'].astype(np.float32), file['scores'].astype(np.float32))
        return graspLabels

    def loadObjModels(self, objIds=None):
        '''
        **Function:**

        - load object 3D models of the given obj ids

        **Input:**

        - objIDs: int or list of int of the object ids

        **Output:**

        - a list of open3d.geometry.PointCloud of the models
        '''
        objIds = self.objIds if objIds is None else objIds
        assert _isArrayLike(objIds) or isinstance(objIds, int), 'objIds must be an integer or a list/numpy array of integers'
        objIds = objIds if _isArrayLike(objIds) else [objIds]
        models = []
        for i in tqdm(objIds, desc='Loading objects...'):
            plyfile = os.path.join(self.root, 'models_m','obj_%06d.ply' % i)
            models.append(o3d.io.read_point_cloud(plyfile))
        return models

    def loadObjTrimesh(self, objIds=None):
        '''
        **Function:**

        - load object 3D trimesh of the given obj ids

        **Input:**

        - objIDs: int or list of int of the object ids

        **Output:**

        - a list of trimesh.Trimesh of the models
        '''
        objIds = self.objIds if objIds is None else objIds
        assert _isArrayLike(objIds) or isinstance(objIds, int), 'objIds must be an integer or a list/numpy array of integers'
        objIds = objIds if _isArrayLike(objIds) else [objIds]
        models = []
        for i in tqdm(objIds, desc='Loading objects...'):
            plyfile = os.path.join(self.root, 'models_m','obj_%06d.ply' % i)
            models.append(trimesh.load(plyfile))
        return models

    def loadCollisionLabels(self, sceneIds=None):
        '''
        **Input:**
        
        - sceneIds: int or list of int of the scene ids.

        **Output:**

        - dict of the collision labels.
        '''
        sceneIds = self.sceneIds if sceneIds is None else sceneIds
        assert _isArrayLike(sceneIds) or isinstance(sceneIds, int), 'sceneIds must be an integer or a list/numpy array of integers'
        sceneIds = sceneIds if _isArrayLike(sceneIds) else [sceneIds]
        collisionLabels = {}
        for sid in tqdm(sceneIds, desc='Loading collision labels...'):
            labels = np.load(os.path.join(self.root, 'collision_label', str(sid).zfill(6) + '.npz'))
            collisionLabel = []
            for j in range(len(labels)):
                collisionLabel.append(labels['arr_{}'.format(j)])
            collisionLabels[str(sid).zfill(6)] = collisionLabel
        return collisionLabels
    
    def annId2ImgId(self, annId, camera = None):
        '''
        **Input:**

        - annId: int of the annotation index.

        **Output:**

        - int of the image index.
        '''
        imgId = annId * 4
        if camera is None:
            camera = self.camera
        if camera == 'realsense-d415':
            imgId += 1
        elif camera == 'realsense-d435':
            imgId += 2
        elif camera == 'azure-kinect':
            imgId += 3
        elif camera == 'zivid':
            imgId += 4
        return imgId

    def loadRGB(self, sceneId, camera, annId):
        '''
        **Input:**

        - sceneId: int of the scene index.
        
        - camera: string of type of camera, 'realsense' or 'zivid'

        - annId: int of the annotation index.

        **Output:**

        - numpy array of the rgb in RGB order.
        '''
        return cv2.cvtColor(cv2.imread(os.path.join(self.root, 'scenes', '%06d' % sceneId, 'rgb', '%06d.png' % self.annId2ImgId(annId, camera))), cv2.COLOR_BGR2RGB)

    def loadBGR(self, sceneId, camera, annId):
        '''
        **Input:**

        - sceneId: int of the scene index.
        
        - camera: string of type of camera, 'realsense' or 'zivid'

        - annId: int of the annotation index.

        **Output:**

        - numpy array of the rgb in BGR order.
        '''
        return cv2.imread(os.path.join(self.root, 'scenes', '%06d' % sceneId, 'rgb', '%06d.png' % self.annId2ImgId(annId, camera)))

    def loadDepth(self, sceneId, camera, annId):
        '''
        **Input:**

        - sceneId: int of the scene index.
        
        - camera: string of type of camera, 'realsense' or 'zivid'

        - annId: int of the annotation index.

        **Output:**

        - numpy array of the depth with dtype = np.uint16
        '''
        return cv2.imread(os.path.join(self.root, 'scenes', '%06d' % sceneId, 'depth', '%06d.png' % self.annId2ImgId(annId, camera)), cv2.IMREAD_UNCHANGED)
 
    def loadMask(self, sceneId, camera, annId):
        '''
        **Input:**

        - sceneId: int of the scene index.
        
        - camera: string of type of camera, 'realsense' or 'zivid'

        - annId: int of the annotation index.

        **Output:**

        - numpy array of the mask with dtype = np.uint16
        '''
        return cv2.imread(os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId, camera, 'label', '%04d.png' % annId), cv2.IMREAD_UNCHANGED)
   
    def loadScenePointCloud(self, sceneId, camera, annId, align=False, format = 'open3d', use_mask = True, use_inpainting = False):
        '''
        **Input:**

        - sceneId: int of the scene index.
        
        - camera: string of type of camera, 'realsense' or 'zivid'

        - annId: int of the annotation index.

        - aligh: bool of whether align to the table frame.

        - format: string of the returned type. 'open3d' or 'numpy'

        - use_mask: bool of whether crop the point cloud use mask(z>0), only open3d 0.9.0 is supported for False option.
                    Only turn to False if you know what you are doing.

        - use_inpainting: bool of whether inpaint the depth image for the missing information.

        **Output:**

        - open3d.geometry.PointCloud instance of the scene point cloud.

        - or tuple of numpy array of point locations and colors.
        '''
        assert camera in ['realsense-d415', 'realsense-d435', 'azure-kinect', 'zivid'], 'camera should be realsense-d415/realsense-d435/azure-kinect/zivid'
        colors = self.loadRGB(sceneId = sceneId, camera = camera, annId = annId).astype(np.float32) / 255.0
        depths = self.loadDepth(sceneId = sceneId, camera = camera, annId = annId)
        if use_inpainting:
            fault_mask = depths < 200
            depths[fault_mask] = 0
            inpainting_mask = (np.abs(depths) < 10).astype(np.uint8)
            depths = cv2.inpaint(depths, inpainting_mask, 5, cv2.INPAINT_NS)

        scene_camera_path = os.path.join(self.root, 'scenes', '%06d' % sceneId, 'scene_camera.json')
        with open(scene_camera_path) as f:
            scene_camera = json.load(f)
        imgId = self.annId2ImgId(annId, camera)
        intrinsics = np.array(scene_camera[str(imgId)]['cam_K']).reshape((3,3))
        fx, fy = intrinsics[0,0], intrinsics[1,1]
        cx, cy = intrinsics[0,2], intrinsics[1,2]
        if camera in ['realsense-d415', 'realsense-d435']:
            s = 1000.0
        elif camera in ['azure-kinect', 'zivid']:
            s = 10000.0
        
        if align:
            camera_poses = np.load(os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId, camera, 'camera_poses.npy'))
            camera_pose = camera_poses[imgId]
            align_mat = np.load(os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId, camera, 'cam0_wrt_table.npy'))
            camera_pose = align_mat.dot(camera_pose)

        xmap, ymap = np.arange(colors.shape[1]), np.arange(colors.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depths / s
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z

        mask = (points_z > 0)
        points = np.stack([points_x, points_y, points_z], axis=-1)
        # print(f'points.shape:{points.shape}')
        if use_mask:
            points = points[mask]
            colors = colors[mask]
        else:
            points = points.reshape((-1, 3))
            colors = colors.reshape((-1, 3))
        if align:
            points = transform_points(points, camera_pose)
        if format == 'open3d':
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            cloud.colors = o3d.utility.Vector3dVector(colors)
            return cloud
        elif format == 'numpy':
            return points, colors
        elif format == 'image_space':
            return np.stack([points_x, points_y, points_z], axis=-1)
        else:
            raise ValueError('Format must be either "open3d" or "numpy".')

    def loadSceneModel(self, sceneId, camera = 'zivid', annId = 0, align = False, return_pose = False):
        '''
        **Input:**

        - sceneId: int of the scene index.
        
        - camera: string of type of camera, 'realsense' or 'zivid'

        - annId: int of the annotation index.

        - align: bool of whether align to the table frame.

        **Output:**

        - open3d.geometry.PointCloud list of the scene models.
        '''
        imgId = self.annId2ImgId(annId, camera)
        if align:
            scene_camera_path = os.path.join(self.root, 'scenes', '%06d'%sceneId, 'scene_camera.json')
            with open(scene_camera_path) as f:
                scene_camera = json.load(f)
            cam_R_w2c = np.array(scene_camera[str(imgId)]['cam_R_w2c']).reshape((3,3))
            cam_t_w2c = np.array(scene_camera[str(imgId)]['cam_t_w2c']).reshape((3,1))
            camera_pose = np.eye(4)
            camera_pose[:3,:3] = cam_R_w2c
            camera_pose[:3,3] = cam_t_w2c.squeeze() / 1000.0


        scene_gt_path = os.path.join(self.root, 'scenes', '%06d'%sceneId, 'scene_gt.json')
        with open(scene_gt_path) as f:
            scene_gt = json.load(f)
        obj_list = []
        pose_list = []
        for obj in scene_gt[str(self.annId2ImgId(annId, camera))]:
            obj_list.append(obj['obj_id'])
            pose = np.eye(4)
            pose[:3,:3] = np.array(obj['cam_R_m2c']).reshape((3,3))
            pose[:3,3] = np.array(obj['cam_t_m2c']).reshape((3,)) / 1000.0
            pose_list.append(pose)

        model_list = []
        for objIdx, pose in zip(obj_list, pose_list):
            plyfile = os.path.join(self.root, 'models_m', 'obj_%06d.ply'%objIdx)
            model = o3d.io.read_point_cloud(plyfile)
            if align:
                pose = np.dot(camera_pose, pose)
            points = np.array(model.points)
            points = transform_points(points, pose)
            model.points = o3d.utility.Vector3dVector(points)
            model_list.append(model)
            pose_list.append(pose)
        
        if return_pose:
            return model_list, pose_list
        else:
            return model_list

    def loadGrasp(self, sceneId, annId=0, format = '6d', camera='zivid', grasp_labels = None, collision_labels = None, fric_coef_thresh=0.4, remove_invisible = True):
        '''
        **Input:**

        - sceneId: int of scene id.

        - annId: int of annotation id.

        - format: string of grasp format, '6d' or 'rect'.

        - camera: string of camera type, 'zivid' or 'realsense'.

        - grasp_labels: dict of grasp labels. Call self.loadGraspLabels if not given.

        - collision_labels: dict of collision labels. Call self.loadCollisionLabels if not given.

        - fric_coef_thresh: float of the frcition coefficient threshold of the grasp. 

        **ATTENTION**

        the LOWER the friction coefficient is, the better the grasp is.

        **Output:**

        - If format == '6d', return a GraspGroup instance.

        - If format == 'rect', return a RectGraspGroup instance.
        '''
        import numpy as np
        assert camera in ['realsense-d415', 'realsense-d435', 'azure-kinect', 'zivid'], 'camera should be realsense-d415/realsense-d435/azure-kinect/zivid'
        assert format == '6d' or format == 'rect', 'format must be "6d" or "rect"'
        if format == '6d':
            from .utils.utils import get_obj_pose_list, generate_views, get_model_grasps, transform_points
            from .utils.rotation import batch_viewpoint_params_to_matrix
            
            imgId = self.annId2ImgId(annId, camera)
            scene_camera_path = os.path.join(self.root, 'scenes', '%06d' % sceneId, 'scene_camera.json')
            with open(scene_camera_path) as f:
                scene_camera = json.load(f)
            cam_R_w2c = np.array(scene_camera[str(imgId)]['cam_R_w2c']).reshape((3,3))
            cam_t_w2c = np.array(scene_camera[str(imgId)]['cam_t_w2c']).reshape((3,1))
            camera_pose = np.eye(4)
            camera_pose[:3,:3] = cam_R_w2c
            camera_pose[:3,3] = cam_t_w2c.squeeze() / 1000.0

            scene_gt_path = os.path.join(self.root, 'scenes', '%06d' % sceneId, 'scene_gt.json')
            with open(scene_gt_path) as f:
                scene_gt = json.load(f)
            obj_list = []
            pose_list = []
            for obj in scene_gt[str(imgId)]:
                pose = np.eye(4)
                pose[:3,:3] = np.array(obj['cam_R_m2c']).reshape((3,3))
                pose[:3,3] = np.array(obj['cam_t_m2c']).reshape((3,)) / 1000
                pose_list.append(pose)
                obj_list.append(obj['obj_id'])
                
                
            if grasp_labels is None:
                print('warning: grasp_labels are not given, calling self.loadGraspLabels to retrieve them')
                grasp_labels = self.loadGraspLabels(objIds = obj_list)
            if collision_labels is None:
                print('warning: collision_labels are not given, calling self.loadCollisionLabels to retrieve them')
                collision_labels = self.loadCollisionLabels(sceneId)

            num_views, num_angles, num_depths = 300, 12, 4
            template_views = generate_views(num_views)
            template_views = template_views[np.newaxis, :, np.newaxis, np.newaxis, :]
            template_views = np.tile(template_views, [1, 1, num_angles, num_depths, 1])

            collision_dump = collision_labels[str(sceneId).zfill(6)]

            if remove_invisible:
                scene_cloud = self.loadScenePointCloud(sceneId, camera, annId, align=False, format='image_space')

            # grasp = dict()
            grasp_group = GraspGroup()
            for i, (objIdx, trans) in enumerate(zip(obj_list, pose_list)):
                
                # skip invisible objects
                if remove_invisible:
                    visible_mask = cv2.imread(os.path.join(self.root, 'scenes', '%06d' % sceneId, 'mask_visib', '{:06d}_{:06d}.png'.format(imgId, i)))
                    amodal_mask = cv2.imread(os.path.join(self.root, 'scenes', '%06d' % sceneId, 'mask', '{:06d}_{:06d}.png'.format(imgId, i)))
                    visibility = np.sum(visible_mask) / np.sum(amodal_mask)
                    if visibility < 0.05 or np.isnan(visibility):
                        continue

                sampled_points, offsets, fric_coefs = grasp_labels[objIdx]
                collision = collision_dump[i]

                # remove invisible grasp points
                if remove_invisible:
                    sampled_points, offsets, fric_coefs, collision = self.removeInvisibleGrasp(
                        scene_cloud, visible_mask, sampled_points, offsets, fric_coefs, collision, trans, th=0.03)

                point_inds = np.arange(sampled_points.shape[0])
                num_points = len(point_inds)
                target_points = sampled_points[:, np.newaxis, np.newaxis, np.newaxis, :]
                target_points = np.tile(target_points, [1, num_views, num_angles, num_depths, 1])
                views = np.tile(template_views, [num_points, 1, 1, 1, 1])
                angles = offsets[:, :, :, :, 0]
                depths = offsets[:, :, :, :, 1]
                widths = offsets[:, :, :, :, 2]
                
                mask1 = ((fric_coefs <= fric_coef_thresh) & (fric_coefs > 0) & ~collision)
                target_points = target_points[mask1]
                target_points = transform_points(target_points, trans)
                #target_points = transform_points(target_points, np.linalg.inv(camera_pose))
                views = views[mask1]
                angles = angles[mask1]
                depths = depths[mask1]
                widths = widths[mask1]
                fric_coefs = fric_coefs[mask1]

                Rs = batch_viewpoint_params_to_matrix(-views, angles)
                Rs = np.matmul(trans[np.newaxis, :3, :3], Rs)
                #Rs = np.matmul(np.linalg.inv(camera_pose)[np.newaxis,:3,:3], Rs)

                num_grasp = widths.shape[0]
                scores = (1.1 - fric_coefs).reshape(-1,1)
                widths = widths.reshape(-1,1)
                heights = GRASP_HEIGHT * np.ones((num_grasp,1))
                depths = depths.reshape(-1,1)
                rotations = Rs.reshape((-1,9))
                object_ids = objIdx * np.ones((num_grasp,1), dtype=np.int32)

                obj_grasp_array = np.hstack([scores, widths, heights, depths, rotations, target_points, object_ids]).astype(np.float32)
                grasp_group.grasp_group_array = np.concatenate((grasp_group.grasp_group_array, obj_grasp_array))
            return grasp_group
        else:
            # 'rect'
            rect_grasps = RectGraspGroup(os.path.join(self.root,'scenes','scene_%04d' % sceneId,camera,'rect','%04d.npy' % annId))
            return rect_grasps

    def loadData(self, ids=None, *extargs):
        '''
        **Input:**

        - ids: int or list of int of the the data ids.

        - extargs: extra arguments. This function can also be called with loadData(sceneId, camera, annId)

        **Output:**

        - if ids is int, returns a tuple of data path

        - if ids is not specified or is a list, returns a tuple of data path lists
        '''
        if ids is None:
            return (self.rgbPath, self.depthPath, self.segLabelPath, self.metaPath, self.rectLabelPath, self.sceneName, self.annId)
        
        if len(extargs) == 0:
            if isinstance(ids, int):
                return (self.rgbPath[ids], self.depthPath[ids], self.segLabelPath[ids], self.metaPath[ids], self.rectLabelPath[ids], self.sceneName[ids], self.annId[ids])
            else:
                return ([self.rgbPath[id] for id in ids],
                    [self.depthPath[id] for id in ids],
                    [self.segLabelPath[id] for id in ids],
                    [self.metaPath[id] for id in ids],
                    [self.rectLabelPath[id] for id in ids],
                    [self.sceneName[id] for id in ids],
                    [self.annId[id] for id in ids])
        if len(extargs) == 2:
            sceneId = ids
            camera, annId = extargs
            rgbPath = os.path.join(self.root, 'scenes', 'scene_'+str(sceneId).zfill(4), camera, 'rgb', str(annId).zfill(4)+'.png')
            depthPath = os.path.join(self.root, 'scenes', 'scene_'+str(sceneId).zfill(4), camera, 'depth', str(annId).zfill(4)+'.png')
            segLabelPath = os.path.join(self.root, 'scenes', 'scene_'+str(sceneId).zfill(4), camera, 'label', str(annId).zfill(4)+'.png')
            metaPath = os.path.join(self.root, 'scenes', 'scene_'+str(sceneId).zfill(4), camera, 'meta', str(annId).zfill(4)+'.mat')
            rectLabelPath = os.path.join(self.root, 'scenes', 'scene_'+str(sceneId).zfill(4), camera, 'rect', str(annId).zfill(4)+'.npy')
            scene_name = 'scene_'+str(sceneId).zfill(4)
            return (rgbPath, depthPath, segLabelPath, metaPath, rectLabelPath, scene_name,annId)

    def showObjGrasp(self, objIds=[], numGrasp=10, th=0.5, maxWidth=0.08, saveFolder='save_fig', show=False):
        '''
        **Input:**

        - objIds: int of list of objects ids.

        - numGrasp: how many grasps to show in the image.

        - th: threshold of the coefficient of friction.

        - maxWidth: float, only visualize grasps with width<=maxWidth

        - saveFolder: string of the path to save the rendered image.

        - show: bool of whether to show the image.

        **Output:**

        - No output but save the rendered image and maybe show it.
        '''
        from .utils.vis import visObjGrasp
        objIds = objIds if _isArrayLike(objIds) else [objIds]
        if len(objIds) == 0:
            print('You need to specify object ids.')
            return 0
        if not os.path.exists(saveFolder):
            os.mkdir(saveFolder)
        for objId in objIds:
            visObjGrasp(self.root, objId, num_grasp=numGrasp, th=th, max_width=maxWidth, save_folder=saveFolder, show=show)

    def showSceneGrasp(self, sceneId, camera = 'zivid', annId = 0, format = '6d', numGrasp = 20, show_object = True, coef_fric_thresh = 0.5, width=0.14):
        '''
        **Input:**

        - sceneId: int of the scene index.

        - camera: string of the camera type, 'realsense' or 'zivid'.

        - annId: int of the annotation index.

        - format: int of the annotation type, 'rect' or '6d'.

        - numGrasp: int of the displayed grasp number, grasps will be randomly sampled.

        - coef_fric_thresh: float of the friction coefficient of grasps.
        '''
        if format == '6d':
            # coef_fric_thresh = 1.0
            geometries = []
            collision_labels = self.loadCollisionLabels(sceneIds = sceneId)
            sceneGrasp = self.loadGrasp(sceneId = sceneId, annId = annId, camera = camera, format = '6d', fric_coef_thresh = coef_fric_thresh, collision_labels=collision_labels)

            sceneGrasp = sceneGrasp.nms(translation_thresh=0.05, rotation_thresh = 30.0 / 180.0 * np.pi)

            widths = sceneGrasp.widths
            sceneGrasp.grasp_group_array = sceneGrasp.grasp_group_array[widths <= width]
            depths = sceneGrasp.depths
            sceneGrasp.grasp_group_array = sceneGrasp.grasp_group_array[np.isclose(depths, 0.02)]
            sceneGrasp = sceneGrasp.nms(translation_thresh=0.03, rotation_thresh = 15.0 / 180.0 * np.pi)

            scenePCD = self.loadScenePointCloud(sceneId = sceneId, camera = camera, annId = annId, align = False)
            geometries.append(scenePCD)
            geometries += sceneGrasp.to_open3d_geometry_list()
            if show_object:
                objectPCD = self.loadSceneModel(sceneId = sceneId, camera = camera, annId = annId, align = False)
                geometries += objectPCD
            o3d.visualization.draw_geometries(geometries)
        elif format == 'rect':
            bgr = self.loadBGR(sceneId = sceneId, camera = camera, annId = annId)
            sceneGrasp = self.loadGrasp(sceneId = sceneId, camera = camera, annId = annId, format = 'rect', fric_coef_thresh = coef_fric_thresh)
            sceneGrasp = sceneGrasp.random_sample(numGrasp = numGrasp)
            img = sceneGrasp.to_opencv_image(bgr, numGrasp = numGrasp)
            cv2.imshow('Rectangle Grasps',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def show6DPose(self, sceneIds, saveFolder='save_fig', show=False, perObj=False):
        '''
        **Input:**

        - sceneIds: int or list of scene ids. 

        - saveFolder: string of the folder to store the image.

        - show: bool of whether to show the image.

        - perObj: bool, show grasps on each object

        **Output:**
        
        - No output but to save the rendered image and maybe show the result.
        '''
        from .utils.vis import vis6D
        sceneIds = sceneIds if _isArrayLike(sceneIds) else [sceneIds]
        if len(sceneIds) == 0:
            print('You need specify scene ids.')
            return 0
        if not os.path.exists(saveFolder):
            os.mkdir(saveFolder)
        for sceneId in sceneIds:
            scene_name = str(sceneId).zfill(6)
            vis6D(self.root, scene_name, 0, self.camera,
                   save_folder=saveFolder, show=show, per_obj=perObj)

    def removeInvisibleGrasp(self, scene_cloud, visible_mask, grasp_points, offsets, fric_coefs, collision, pose, th=0.05):                

        visible_mask = visible_mask[:, :, 0]
        scene_cloud = scene_cloud[visible_mask > 0]

        # transform grasp points to the object frame
        ones = np.ones(grasp_points.shape[0])[:, np.newaxis]
        cloud_ = np.concatenate([grasp_points, ones], axis=1)
        cloud_transformed = np.dot(pose, cloud_.T).T
        cloud_transformed = cloud_transformed[:, :3]

        if scene_cloud.shape[0] > 20000:
            scene_cloud = scene_cloud[np.random.choice(scene_cloud.shape[0], 20000, replace=False)]

        # find the closest point in the object cloud
        cloud_transformed = cloud_transformed[:, np.newaxis, :]
        scene_cloud = scene_cloud[np.newaxis, :, :]
        dists = np.linalg.norm(cloud_transformed-scene_cloud, axis=-1)
        min_dists = dists.min(axis=1)
        visible_point_mask = (min_dists < th)

        # remove invisible grasp points (distance to the object is larger than th)
        grasp_points = grasp_points[visible_point_mask]
        offsets = offsets[visible_point_mask]
        fric_coefs = fric_coefs[visible_point_mask]
        collision = collision[visible_point_mask]
        return grasp_points, offsets, fric_coefs, collision
