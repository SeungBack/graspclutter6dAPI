# Author: Seunghyeok Back (shback@kimm.re.kr)
# KIMM & GIST, Republic of Korea
# Base codes are from GraspNetAPI (hsfang, mhgou, cxwang), https://github.com/graspnet/graspnetAPI


import numpy as np
import os
import time
import pickle
import open3d as o3d
import json

from tqdm import tqdm

from .graspclutter6d import GraspClutter6D
from .grasp import GraspGroup
from .utils.config import get_config
from .utils.eval_utils import get_scene_name, create_table_points, parse_posevector, load_dexnet_model, transform_points, collision_detection_per_object_optimized, compute_point_distance, compute_closest_points, voxel_sample_points, topk_grasps, get_grasp_score, collision_detection, collision_detection_per_object, eval_grasp
from .utils.xmlhandler import xmlReader
from .utils.utils import generate_scene_model

class GraspClutter6DEval(GraspClutter6D):
    '''
    Class for evaluation on GraspClutter6D dataset.
    
    **Input:**

    - root: string of root path for the dataset.

    - camera: string of type of the camera.

    - split: string of the date split.
    '''
    def __init__(self, root, camera, split = 'test'):
        super(GraspClutter6DEval, self).__init__(root, camera, split)
        
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

    def get_scene_models(self, sceneId, annId):
        '''
            return models in model coordinate
        '''

        obj_list = []
        model_list = []
        dexmodel_list = []
        scene_gt_path = os.path.join(self.root, 'scenes', '%06d' % sceneId, 'scene_gt.json')
        with open(scene_gt_path) as f:
            scene_gt = json.load(f)
        obj_list = []
        for obj in scene_gt[str(self.annId2ImgId(annId))]:
            obj_list.append(obj['obj_id'])
        for obj_idx in obj_list:
            model = o3d.io.read_point_cloud(os.path.join(self.root, 'models_m', 'obj_%06d.ply'%obj_idx))
            dex_cache_path = os.path.join(self.root, 'dex_models', 'obj_%06d.pkl' % obj_idx)
            if os.path.exists(dex_cache_path):
                with open(dex_cache_path, 'rb') as f:
                    dexmodel = pickle.load(f)
            else:
                dexmodel = load_dexnet_model(os.path.join(self.root, 'models_obj_m', 'obj_%06d' % obj_idx))
                # save dexmodel
                print('Saving dexmodel to {}'.format(dex_cache_path) + ' It may take a while, but only need to run once.')
                with open(dex_cache_path, 'wb') as f:
                    pickle.dump(dexmodel, f)
            points = np.array(model.points)
            model_list.append(points)
            dexmodel_list.append(dexmodel)
        return model_list, dexmodel_list, obj_list


    def get_model_poses(self, sceneId, annId):
        '''
        **Input:**

        - sceneId: int of the scen index.

        - annId: int of the annotation index.

        **Output:**

        - obj_list: list of int of object index.

        - pose_list: list of 4x4 matrices of object poses.

        - camera_pose: 4x4 matrix of the camera pose relative to the first frame.

        - align mat: 4x4 matrix of camera relative to the table.
        '''
        imgId = self.annId2ImgId(annId, self.camera)
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
        return obj_list, pose_list, camera_pose
        
    def eval_scene(self, sceneId, dump_folder, TOP_K = 50, return_list = False,vis = False, max_width = 0.14):
        '''
        **Input:**

        - sceneId: int of the scene index.
        
        - dump_folder: string of the folder that saves the dumped npy files.

        - TOP_K: int of the top number of grasp to evaluate

        - return_list: bool of whether to return the result list.

        - vis: bool of whether to show the result

        - max_width: float of the maximum gripper width in evaluation

        **Output:**

        - scene_accuracy: np.array of shape (256, 50, 6) of the accuracy tensor.
        '''
        config = get_config()
        list_coe_of_friction = [0.2,0.4,0.6,0.8,1.0,1.2]

        model_list, dexmodel_list, _ = self.get_scene_models(sceneId, annId=0)

        model_sampled_list = list()
        for model in model_list:
            model_sampled = voxel_sample_points(model, 0.008)
            model_sampled_list.append(model_sampled)

        scene_accuracy = []
        grasp_list_list = []
        score_list_list = []
        collision_list_list = []

        for annId in range(13):
            img_num = self.annId2ImgId(annId, self.camera)
            if not os.path.exists(os.path.join(dump_folder, get_scene_name(sceneId), self.camera, '%06d.npy' % (img_num,))):
                grasp_accuracy = np.zeros((TOP_K,len(list_coe_of_friction)))
                scene_accuracy.append(grasp_accuracy)
                grasp_list_list.append([])
                score_list_list.append([])
                collision_list_list.append([])
                print('\rMean Accuracy for scene:{} im:{}='.format(sceneId, img_num),np.mean(grasp_accuracy[:,:]), end='')
                continue
            grasp_group = GraspGroup().from_npy(os.path.join(dump_folder, get_scene_name(sceneId), self.camera, '%06d.npy' % (img_num,)))
            obj_list, pose_list, camera_pose = self.get_model_poses(sceneId, annId)

            # clip width to [0,max_width]
            gg_array = grasp_group.grasp_group_array
            min_width_mask = (gg_array[:,1] < 0)
            max_width_mask = (gg_array[:,1] > max_width)
            gg_array[min_width_mask,1] = 0
            gg_array[max_width_mask,1] = max_width
            grasp_group.grasp_group_array = gg_array

            grasp_list, score_list, collision_mask_list = eval_grasp(grasp_group, model_sampled_list, dexmodel_list, pose_list, config, voxel_size=0.008, TOP_K = TOP_K)

            # remove empty
            grasp_list = [x for x in grasp_list if len(x) != 0]
            score_list = [x for x in score_list if len(x) != 0]
            collision_mask_list = [x for x in collision_mask_list if len(x)!=0]

            if len(grasp_list) == 0:
                grasp_accuracy = np.zeros((TOP_K,len(list_coe_of_friction)))
                scene_accuracy.append(grasp_accuracy)
                grasp_list_list.append([])
                score_list_list.append([])
                collision_list_list.append([])
                print('\rMean Accuracy for scene:{} im:{}='.format(sceneId, img_num),np.mean(grasp_accuracy[:,:]), end='')
                continue

            # concat into scene level
            grasp_list, score_list, collision_mask_list = np.concatenate(grasp_list), np.concatenate(score_list), np.concatenate(collision_mask_list)
            
            if vis:
                t = o3d.geometry.PointCloud()
                # t.points = o3d.utility.Vector3dVector(table_trans)
                model_list = generate_scene_model(self.root, 'scene_%04d' % sceneId , annId, return_poses=False, align=False, camera=self.camera)
                import copy
                gg = GraspGroup(copy.deepcopy(grasp_list))
                scores = np.array(score_list)
                scores = scores / 2 + 0.5 # -1 -> 0, 0 -> 0.5, 1 -> 1
                scores[collision_mask_list] = 0.3
                gg.scores = scores
                gg.widths = 0.1 * np.ones((len(gg)), dtype = np.float16)
                grasps_geometry = gg.to_open3d_geometry_list()
                pcd = self.loadScenePointCloud(sceneId, self.camera, annId)

                o3d.visualization.draw_geometries([pcd, *grasps_geometry])
                o3d.visualization.draw_geometries([pcd, *grasps_geometry, *model_list])
                o3d.visualization.draw_geometries([*grasps_geometry, *model_list, t])
            
            # sort in scene level
            grasp_confidence = grasp_list[:,0]
            indices = np.argsort(-grasp_confidence)
            grasp_list, score_list, collision_mask_list = grasp_list[indices], score_list[indices], collision_mask_list[indices]

            grasp_list_list.append(grasp_list)
            score_list_list.append(score_list)
            collision_list_list.append(collision_mask_list)

            #calculate AP
            grasp_accuracy = np.zeros((TOP_K,len(list_coe_of_friction)))
            for fric_idx, fric in enumerate(list_coe_of_friction):
                for k in range(0,TOP_K):
                    if k+1 > len(score_list):
                        grasp_accuracy[k,fric_idx] = np.sum(((score_list<=fric) & (score_list>0)).astype(int))/(k+1)
                    else:
                        grasp_accuracy[k,fric_idx] = np.sum(((score_list[0:k+1]<=fric) & (score_list[0:k+1]>0)).astype(int))/(k+1)

            print('\rMean Accuracy for scene:%04d im id:%04d = %.3f' % (sceneId, img_num, 100.0 * np.mean(grasp_accuracy[:,:])), end='', flush=True)
            scene_accuracy.append(grasp_accuracy)
        if not return_list:
            return scene_accuracy
        else:
            return scene_accuracy, grasp_list_list, score_list_list, collision_list_list

    def parallel_eval_scenes(self, sceneIds, dump_folder, proc = 2):
        '''
        **Input:**

        - sceneIds: list of int of scene index.

        - dump_folder: string of the folder that saves the npy files.

        - proc: int of the number of processes to use to evaluate.

        **Output:**

        - scene_acc_list: list of the scene accuracy.
        '''
        from multiprocessing import Pool
        p = Pool(processes = proc)
        res_list = []
        for sceneId in sceneIds:
            res_list.append(p.apply_async(self.eval_scene, (sceneId, dump_folder)))
        p.close()
        p.join()
        scene_acc_list = []
        for res in res_list:
            scene_acc_list.append(res.get())
        return scene_acc_list


    def eval_all(self, dump_folder, proc = 2):
        '''
        **Input:**

        - dump_folder: string of the folder that saves the npy files.

        - proc: int of the number of processes to use to evaluate.

        **Output:**

        - res: numpy array of the detailed accuracy.

        - ap: float of the AP for all split.
        '''
        with open(os.path.join(self.root, 'split_info', 'grasp_test_sceneIds.json')) as f:
            sceneIds = [int(x) for x in json.load(f)] # if you want small subset, use sceneIds[:10]
            # sceneIds = [int(x) for x in json.load(f)][:5] # shsh
        res = np.array(self.parallel_eval_scenes(sceneIds = sceneIds, dump_folder = dump_folder, proc = proc))
        _res = res.copy()
        _res = _res.transpose(3,0,1,2).reshape(6,-1)
        _res = np.mean(_res, axis=1)
        ap = [np.mean(res), _res[1], _res[3]]
        print('\nEvaluation Result:\n----------\n{}\nAP={:.2f}\nAP0.4={:.2f}\nAP0.8={:.2f}'.format(
                        self.camera, 100*ap[0], 100*ap[1], 100*ap[2]))
        return res, ap