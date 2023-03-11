import numpy
import numpy as np
import pickle
from ...utils import box_utils
from ...ops.iou3d_nms import iou3d_nms_utils
import copy
from pcdet.config import cfg
import glob
import os
import pickle as pkl
from random import sample
import scipy.stats
import matplotlib.pyplot as plt

class DataBaseSampler(object):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg
        self.logger = logger
        self.db_infos = {}

        self.class_names=class_names = []
        for x in sampler_cfg.SAMPLE_GROUPS:
            class_name, sample_num = x.split(':')
            class_names.append(class_name)

        for class_name in class_names:
            self.db_infos[class_name] = []

        if sampler_cfg.get('SELF_TRAIN_DB', None):
            if sampler_cfg.SELF_TRAIN_DB == 'kitti_to_waymo':
                class_names = copy.copy(class_names)
                class_names[0] = 'Vehicle'
                self.class_names = class_names

        for db_info_path in sampler_cfg.DB_INFO_PATH:
            db_info_path = self.root_path.resolve() / db_info_path
            with open(str(db_info_path), 'rb') as f:
                infos = pickle.load(f)
                [self.db_infos[cur_class].extend(infos[cur_class]) for cur_class in class_names]

        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)
        for x in sampler_cfg.SAMPLE_GROUPS:
            class_name, sample_num = x.split(':')
            # if class_name not in class_names:
            #     continue
            self.sample_class_num[class_name] = sample_num
            self.sample_groups[class_name] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name]))
            }

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def filter_by_difficulty(self, db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            pre_len = len(dinfos)
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
            if self.logger is not None:
                self.logger.info('Database filter by difficulty %s: %d => %d' % (key, pre_len, len(new_db_infos[key])))
        return new_db_infos

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)

                if self.logger is not None:
                    self.logger.info('Database filter by min points %s: %d => %d' %
                                     (name, len(db_infos[name]), len(filtered_infos)))
                db_infos[name] = filtered_infos

        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:
            sample_group:
        Returns:

        """
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict

    @staticmethod
    def put_boxes_on_road_planes(gt_boxes, road_planes, calib):
        """
        Only validate in KITTIDataset
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            road_planes: [a, b, c, d]
            calib:

        Returns:
        """
        a, b, c, d = road_planes
        center_cam = calib.lidar_to_rect(gt_boxes[:, 0:3])
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]
        mv_height = gt_boxes[:, 2] - gt_boxes[:, 5] / 2 - cur_lidar_height
        gt_boxes[:, 2] -= mv_height  # lidar view
        return gt_boxes, mv_height

    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict):
        gt_boxes_mask = data_dict['gt_boxes_mask']
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]
        points = data_dict['points']
        if self.sampler_cfg.get('USE_ROAD_PLANE', False):
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )
            data_dict.pop('calib')
            data_dict.pop('road_plane')

        obj_points_list = []
        for idx, info in enumerate(total_valid_sampled_dict):
            file_path = self.root_path / info['path']
            obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                [-1, self.sampler_cfg.NUM_POINT_FEATURES])

            obj_points[:, :3] += info['box3d_lidar'][:3]

            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                # mv height
                obj_points[:, 2] -= mv_height[idx]

            obj_points_list.append(obj_points)

        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])

        if cfg.DATA_CONFIG.get('SHIFT_COOR', None): # sample to target domain
            points[:, 0:3] += np.array(cfg.DATA_CONFIG.SHIFT_COOR, dtype=np.float32)
            sampled_gt_boxes[:, 0:3] += cfg.DATA_CONFIG.SHIFT_COOR

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )
        points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)
        points = np.concatenate([obj_points[:, :points.shape[-1]], points], axis=0)
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)

        # for ignored boxes
        if 'gt_classes' in data_dict.keys():
            sampled_gt_classes = np.array([
                self.class_names.index(n) + 1 for n in sampled_gt_names],
                dtype=np.int32)
            data_dict['gt_classes'] = np.concatenate([
                data_dict['gt_classes'], sampled_gt_classes], axis=0)

        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points
        return data_dict

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names'].astype(str)
        existed_boxes = gt_boxes
        total_valid_sampled_dict = []
        for class_name, sample_group in self.sample_groups.items():
            if self.limit_whole_scene:
                num_gt = np.sum(class_name == gt_names)
                sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt)
            if int(sample_group['sample_num']) > 0:
                sampled_dict = self.sample_with_fixed_number(class_name, sample_group)

                sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)

                if self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False):
                    sampled_boxes = box_utils.boxes3d_kitti_fakelidar_to_lidar(sampled_boxes)

                iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
                iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
                iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
                iou1 = iou1 if iou1.shape[1] > 0 else iou2
                valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0).nonzero()[0]
                valid_sampled_dict = [sampled_dict[x] for x in valid_mask]
                valid_sampled_boxes = sampled_boxes[valid_mask]

                existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes[:,:7]), axis=0)
                total_valid_sampled_dict.extend(valid_sampled_dict)

        sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]
        if total_valid_sampled_dict.__len__() > 0:
            data_dict = self.add_sampled_boxes_to_scene(data_dict, sampled_gt_boxes, total_valid_sampled_dict)

        data_dict['gt_boxes_mask'] = np.ones(data_dict['gt_boxes'].shape[0], dtype=np.bool_)
        return data_dict

def ps_sampling(data_dict, sampled_dict=None):
    """
    Args:
        data_dict:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

    Returns:
    :param sample_class_names:

    """

    # Load PS Objects Labels
    class_names = ['Vehicle', 'Pedestrian', 'Cyclist']

    ps_box_list = glob.glob(os.path.join(cfg.SELF_TRAIN.PS_SAMPLING.PS_OBJECT_PATH, 'ps_box_e*.pkl'))
    ps_pnt_list = glob.glob(os.path.join(cfg.SELF_TRAIN.PS_SAMPLING.PS_OBJECT_PATH,'ps_point_e*.pkl'))
    if cfg.SELF_TRAIN.get('DIVERSITY_SAMPLING', None):
        ps_diversity_list = glob.glob(os.path.join(cfg.SELF_TRAIN.PS_SAMPLING.PS_OBJECT_PATH, 'ps_diverse_e*.pkl'))

    try:

        if cfg.SELF_TRAIN.get('DIVERSITY_SAMPLING', None):
            ps_diversity_list.sort(key=os.path.getmtime, reverse=True)
            with open(ps_diversity_list[-1], 'rb') as f:
                ps_diversity = pkl.load(f)

        ps_box_list.sort(key=os.path.getmtime, reverse=True)
        with open(ps_box_list[-1], 'rb') as f:
            ps_boxes = pkl.load(f)
        ps_pnt_list.sort(key=os.path.getmtime, reverse=True)
        with open(ps_pnt_list[-1], 'rb') as f:
            ps_points = pkl.load(f)
    except IndexError: # No file found
        return data_dict

    sample_groups = {}
    sample_class_num = {}
    for class_name, sample_num in cfg.SELF_TRAIN.PS_SAMPLING.SAMPLE_GROUPS.items():
        if class_name not in cfg.CLASS_NAMES:
            continue
        sample_class_num[class_name] = sample_num
        sample_groups[class_name] = {
            'sample_num': sample_num
        }

    gt_boxes = data_dict['gt_boxes']
    gt_names = data_dict['gt_names'].astype(str)
    existed_boxes = gt_boxes
    sampled_class, total_sampled_pnts = [], []

    # sample_groups = cfg.SELF_TRAIN.PS_SAMPLING.SAMPLE_GROUPS
    for class_name, sample_group in sample_groups.items():

        if int(sample_group['sample_num']) > 0:
            sample_idx = np.random.randint(len(ps_boxes[class_name]), size=int(sample_group['sample_num']))
            sampled_boxes = np.stack([ps_boxes[class_name][i] for i in sample_idx]).astype(np.float32)[:,:7] # (ps box num, 7)
            sampled_pnts = [ps_points[class_name][i] for i in sample_idx]
            iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
            iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
            iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
            iou1 = iou1 if iou1.shape[1] > 0 else iou2
            valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0).nonzero()[0]
            valid_sampled_boxes = sampled_boxes[valid_mask]
            existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes), axis=0)

            # sampled classes
            sampled_class.extend([class_name for i in range(len(valid_mask))])

            # sampled points
            for mask_idx in range(valid_mask.shape[0]):
                total_sampled_pnts.append(sampled_pnts[mask_idx])

    sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]
    if sampled_gt_boxes.shape[0] > 0:
        data_dict = add_sampled_ps_boxes_to_scene(data_dict,sampled_gt_boxes, total_sampled_pnts, sampled_class)

    return data_dict


def add_sampled_ps_boxes_to_scene(data_dict, sampled_ps_boxes,
                                  sampled_ps_pnts, sampled_classes):

    gt_boxes = data_dict['gt_boxes']
    gt_names = data_dict['gt_names']
    points = data_dict['points']

    ps_pnts = np.concatenate(sampled_ps_pnts, axis=0)
    extra_dim = points.shape[-1] -3
    zero_dim = np.zeros((ps_pnts.shape[0], extra_dim)).reshape(
        ps_pnts.shape[0], extra_dim)
    ps_pnts = np.concatenate((ps_pnts, zero_dim), axis=1)

    points = box_utils.remove_points_in_boxes3d(points, sampled_ps_boxes)

    points = np.concatenate([ps_pnts[:, :points.shape[-1]], points], axis=0)
    gt_names = np.concatenate([gt_names, sampled_classes], axis=0)
    gt_boxes = np.concatenate([gt_boxes, sampled_ps_boxes], axis=0)

    data_dict['gt_boxes'] = gt_boxes
    data_dict['gt_names'] = gt_names
    data_dict['points'] = points
    if 'gt_classes' in data_dict.keys():
        sampled_gt_classes = np.array([
            cfg.CLASS_NAMES.index(n) + 1 for n in sampled_classes],
            dtype=np.int32)
        data_dict['gt_classes'] = np.concatenate([
            data_dict['gt_classes'], sampled_gt_classes], axis=0)

    return data_dict