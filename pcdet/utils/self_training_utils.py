

import torch
import os
import glob
import tqdm
import numpy as np
import torch.distributed as dist
from pcdet.config import cfg
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils, commu_utils, memory_ensemble_utils
import pickle as pkl
import re
from pcdet.models.model_utils.dsnorm import set_ds_target
from multiprocessing import Manager
from scipy.stats import hmean
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
import wandb
from pcdet.utils.box_utils import remove_points_in_boxes3d, enlarge_box3d, \
    boxes3d_kitti_lidar_to_fakelidar, boxes_to_corners_3d
import copy
import scipy.stats

PSEUDO_LABELS = {}
PSEUDO_LABELS = Manager().dict()
NEW_PSEUDO_LABELS = {}


def check_already_exsit_pseudo_label(ps_label_dir, start_epoch):
    """
    if we continue training, use this to directly
    load pseudo labels from exsiting result pkl

    if exsit, load latest result pkl to PSEUDO LABEL
    otherwise, return false and

    Args:
        ps_label_dir: dir to save pseudo label results pkls.
        start_epoch: start epoc
    Returns:

    """
    # support init ps_label given by cfg
    if start_epoch == 0 and cfg.SELF_TRAIN.get('INIT_PS', None):
        if os.path.exists(cfg.SELF_TRAIN.INIT_PS):
            init_ps_label = pkl.load(open(cfg.SELF_TRAIN.INIT_PS, 'rb'))
            PSEUDO_LABELS.update(init_ps_label)

            if cfg.LOCAL_RANK == 0:
                ps_path = os.path.join(ps_label_dir, "ps_label_e0.pkl")
                with open(ps_path, 'wb') as f:
                    pkl.dump(PSEUDO_LABELS, f)

            return cfg.SELF_TRAIN.INIT_PS

    ps_label_list = glob.glob(os.path.join(ps_label_dir, 'ps_label_e*.pkl'))
    if len(ps_label_list) == 0:
        return

    ps_label_list.sort(key=os.path.getmtime, reverse=True)
    for cur_pkl in ps_label_list:
        num_epoch = re.findall('ps_label_e(.*).pkl', cur_pkl)
        assert len(num_epoch) == 1

        # load pseudo label and return
        if int(num_epoch[0]) <= start_epoch:
            latest_ps_label = pkl.load(open(cur_pkl, 'rb'))
            PSEUDO_LABELS.update(latest_ps_label)
            return cur_pkl

    return None


def save_pseudo_label_epoch(model, val_loader, rank, leave_pbar, ps_label_dir,
                            cur_epoch, source_reader=None, source_model=None):
    """
    Generate pseudo label with given model.

    Args:
        model: model to predict result for pseudo label
        val_loader: data_loader to predict pseudo label
        rank: process rank
        leave_pbar: tqdm bar controller
        ps_label_dir: dir to save pseudo label
        cur_epoch
    """
    val_dataloader_iter = iter(val_loader)
    total_it_each_epoch = len(val_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                         desc='generate_ps_e%d' % cur_epoch, dynamic_ncols=True)

    pos_ps_nmeter = common_utils.NAverageMeter(len(cfg.CLASS_NAMES))
    ign_ps_nmeter = common_utils.NAverageMeter(len(cfg.CLASS_NAMES))

    if cfg.SELF_TRAIN.get('DSNORM', None):
        model.apply(set_ds_target)

    # related_boxes_count = Overlapped Boxes Counting (OBC) in paper
    related_boxes_count_list = [] if \
        cfg.SELF_TRAIN.get('DIVERSITY_SAMPLING', None) else None

    model.eval()

    total_quality_metric = None
    if cfg.SELF_TRAIN.get('REPORT_PS_LABEL_QUALITY', None) and \
            cfg.SELF_TRAIN.REPORT_PS_LABEL_QUALITY:
        total_quality_metric = {
            cls_id:{'gt': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'scale_err': 0}
            for cls_id in range(len(cfg.CLASS_NAMES))}

    for cur_it in range(total_it_each_epoch):
        try:
            target_batch = next(val_dataloader_iter)
        except StopIteration:
            target_dataloader_iter = iter(val_loader)
            target_batch = next(target_dataloader_iter)

        # generate gt_boxes for target_batch and update model weights
        with torch.no_grad():
            load_data_to_gpu(target_batch)
            pred_dicts, ret_dict = model(target_batch)

        pos_ps_batch_nmeters, ign_ps_batch_nmeters = save_pseudo_label_batch(
            target_batch, pred_dicts=pred_dicts,
            need_update=(cfg.SELF_TRAIN.get('MEMORY_ENSEMBLE', None) and
                         cfg.SELF_TRAIN.MEMORY_ENSEMBLE.ENABLED and
                         cur_epoch > 0),
            total_quality_metric=total_quality_metric,
            source_reader=source_reader,
            model=model,
            source_model=source_model,
            related_boxes_count_list=related_boxes_count_list
        )

        # log to console and tensorboard
        pos_ps_nmeter.update(pos_ps_batch_nmeters)
        ign_ps_nmeter.update(ign_ps_batch_nmeters)
        pos_ps_result = pos_ps_nmeter.aggregate_result()
        ign_ps_result = ign_ps_nmeter.aggregate_result()

        disp_dict = {'pos_ps_box': pos_ps_result, 'ign_ps_box': ign_ps_result}

        if rank == 0:
            pbar.update()
            pbar.set_postfix(disp_dict)
            pbar.refresh()

    if rank == 0:
        pbar.close()

        if cfg.SELF_TRAIN.get('PROGRESSIVE_SAMPLING', None) and cfg.SELF_TRAIN.PROGRESSIVE_SAMPLING.ENABLE and cur_epoch != cfg.OPTIMIZATION.NUM_EPOCHS:
            gt_reduce = cfg.SELF_TRAIN.PROGRESSIVE_SAMPLING.GT_REDUCE
            ps_grow = cfg.SELF_TRAIN.PROGRESSIVE_SAMPLING.PS_GROW
            if cfg.SELF_TRAIN.get('PS_SAMPLING', None):
                for k in cfg.SELF_TRAIN.PS_SAMPLING.SAMPLE_GROUPS:
                    cfg.SELF_TRAIN.PS_SAMPLING.SAMPLE_GROUPS[k] += ps_grow
            if cfg.DATA_CONFIG_TAR.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].NAME == 'gt_sampling' and \
                    'gt_sampling' not in cfg.DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST:
                new_sample_groups = []
                for i in cfg.DATA_CONFIG_TAR.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].SAMPLE_GROUPS:
                    new_sample_num = str(int(i.split(":")[-1])-gt_reduce)
                    new_i = i.split(":")[0] + ':' + new_sample_num
                    new_sample_groups.append(new_i)
                cfg.DATA_CONFIG_TAR.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].SAMPLE_GROUPS = new_sample_groups

        if cfg.SELF_TRAIN.get('DIVERSITY_SAMPLING', None):
            # remove outliers
            related_boxes_count_list = [ i if len(i.shape) == 1
                                         else np.array([])
                                         for i in related_boxes_count_list]
            related_boxes_count_all = np.concatenate(related_boxes_count_list)

        if total_quality_metric is not None:
            for cls_id in total_quality_metric.keys():
                for key, val in total_quality_metric[cls_id].items():
                    wandb.log(
                        {'PS-Label Quality Class-' +
                         str(cls_id) + '/' + key: val})
    model.train()
    gather_and_dump_pseudo_label_result(rank, ps_label_dir, cur_epoch)


def gather_and_dump_pseudo_label_result(rank, ps_label_dir, cur_epoch):
    commu_utils.synchronize()

    if dist.is_initialized():
        part_pseudo_labels_list = commu_utils.all_gather(NEW_PSEUDO_LABELS)

        new_pseudo_label_dict = {}
        for pseudo_labels in part_pseudo_labels_list:
            new_pseudo_label_dict.update(pseudo_labels)

        NEW_PSEUDO_LABELS.update(new_pseudo_label_dict)

    # dump new pseudo label to given dir

    if rank == 0:
        ps_path = os.path.join(ps_label_dir, "ps_label_e{}.pkl".format(cur_epoch))
        with open(ps_path, 'wb') as f:
            pkl.dump(NEW_PSEUDO_LABELS, f)


        if cfg.SELF_TRAIN.get('PS_SAMPLING',None) and \
                cfg.SELF_TRAIN.PS_SAMPLING.ENABLE:

            cfg.SELF_TRAIN.PS_SAMPLING.PS_OBJECT_PATH = ps_label_dir
            ps_point_path = os.path.join(ps_label_dir,
                                          "ps_point_e{}.pkl".format(cur_epoch))
            ps_box_path = os.path.join(ps_label_dir,
                                          "ps_box_e{}.pkl".format(cur_epoch))


            class_names = cfg.CLASS_NAMES
            ps_boxes_dict = {class_name: [] for class_name in class_names}
            ps_points_dict = {class_name: [] for class_name in class_names}

            if cfg.SELF_TRAIN.get('DIVERSITY_SAMPLING', None):
                ps_related_counts_dict = {class_name: [] for class_name in class_names}

            for frame_id, frame in NEW_PSEUDO_LABELS.items():
                ps_boxes = frame['gt_boxes']
                if 'quality_mask' in frame.keys():
                    quality_mask = frame['quality_mask'].astype(np.bool)
                else:
                    quality_mask = np.ones(ps_boxes.shape[0]).astype(np.bool)
                pos_ps_mask = ps_boxes[:, -2] > 0
                quality_mask = quality_mask[pos_ps_mask]
                ps_boxes = ps_boxes[pos_ps_mask]
                ps_labels = ps_boxes[:, -2]
                ps_scores = ps_boxes[:, -1]
                if cfg.SELF_TRAIN.get('DIVERSITY_SAMPLING', None):
                    related_boxes_count = frame['related_box_count'][pos_ps_mask]

                ps_points_list = []
                for mask_idx in range(pos_ps_mask.shape[0]):
                    if pos_ps_mask[mask_idx]:
                        ps_points_list.append(frame['gt_points'][mask_idx])

                remain_mask = quality_mask

                remain_ps_points_list = []
                for mask_idx in range(remain_mask.shape[0]):
                    if remain_mask[mask_idx]:
                        remain_ps_points_list.append(ps_points_list[mask_idx])

                ps_boxes = ps_boxes[remain_mask]
                ps_labels = ps_labels[remain_mask]
                if cfg.SELF_TRAIN.get('DIVERSITY_SAMPLING', None):
                    related_boxes_count = related_boxes_count[remain_mask]

                # ps_points = ps_points[remain_mask]
                ps_names = [class_names[int(class_id - 1)]
                            for class_id in ps_labels.tolist()]
                for inx in range(len(ps_names)):
                    ps_boxes_dict[ps_names[inx]].append(ps_boxes[inx])
                    ps_points_dict[ps_names[inx]].append(remain_ps_points_list[inx])
                    if cfg.SELF_TRAIN.get('DIVERSITY_SAMPLING', None):
                        ps_related_counts_dict[ps_names[inx]].append(related_boxes_count[inx])

                """ Print how many ps-labels for sampling """
            print('Distribution of PS-Labels for sampling at epoch {}:'.format(cur_epoch))
            for cls in ps_boxes_dict.keys():
                print('# of ps labels of class {} is {}'.format(cls, len(ps_boxes_dict[cls])))
                key = 'PS-Label before div-sampling/' + cls
                wandb.log({key: len(ps_boxes_dict[cls])})

            """------------------ OBC-based Diversity Down-sampling ------------------"""
            if cfg.SELF_TRAIN.get('DIVERSITY_SAMPLING', None):
                ps_diverse_path = os.path.join(ps_label_dir, "ps_diverse_e{}.pkl".format(cur_epoch))
                for class_name in cfg.SELF_TRAIN.DIVERSITY_SAMPLING.SAMPLE_CLASSES:
                    ps_boxes = np.stack(ps_boxes_dict[class_name])
                    related_box_counts = np.stack(ps_related_counts_dict[class_name])
                    kde = scipy.stats.gaussian_kde(related_box_counts.T)
                    p = 1 / kde.pdf(related_box_counts.T)
                    p /= np.sum(p)
                    sample_size = int(len(related_box_counts) / cfg.SELF_TRAIN.DIVERSITY_SAMPLING.DOWNSAMPLE_RATE)
                    sample_idx = np.random.choice(np.arange(len(related_box_counts)), size=sample_size, replace=False, p=p)
                    sampled_boxes = ps_boxes[sample_idx].astype(np.float32)[:, :7]  # (ps box num, 7)
                    sampled_pnts = [ps_points_dict[class_name][i] for i in sample_idx]

                    ps_boxes_dict[class_name] = sampled_boxes
                    ps_points_dict[class_name] = sampled_pnts
                    ps_related_counts_dict[class_name]= related_box_counts[sample_idx]

                with open(ps_diverse_path, 'wb') as f:
                    pkl.dump(ps_related_counts_dict, f)
                """ Print how many ps-labels for sampling """
                print('Distribution of PS-Labels for DIVERSE sampling at epoch {}:'.format(cur_epoch))
                for cls in ps_boxes_dict.keys():
                    print('# of ps labels of class {} is {}'.format(cls, len(ps_boxes_dict[cls])))
                    key = 'PS-Label after div-sampling/' + cls
                    wandb.log({key: len(ps_boxes_dict[cls])})


            with open(ps_box_path, 'wb') as f:
                pkl.dump(ps_boxes_dict, f)
            with open(ps_point_path, 'wb') as f:
                pkl.dump(ps_points_dict, f)

    commu_utils.synchronize()
    PSEUDO_LABELS.clear()
    PSEUDO_LABELS.update(NEW_PSEUDO_LABELS)
    NEW_PSEUDO_LABELS.clear()


def save_pseudo_label_batch(input_dict,
                            pred_dicts=None,
                            need_update=True,
                            total_quality_metric=None,
                            source_reader=None,
                            model = None,
                            source_model=None,
                            related_boxes_count_list=None):
    """
    Save pseudo label for give batch.
    If model is given, use model to inference pred_dicts,
    otherwise, directly use given pred_dicts.

    Args:
        input_dict: batch data read from dataloader
        pred_dicts: Dict if not given model.
            predict results to be generated pseudo label and saved
        need_update: Bool.
            If set to true, use consistency matching to update pseudo label
    """
    pos_ps_nmeter = common_utils.NAverageMeter(len(cfg.CLASS_NAMES))
    ign_ps_nmeter = common_utils.NAverageMeter(len(cfg.CLASS_NAMES))


    batch_size = len(pred_dicts)
    for b_idx in range(batch_size):
        pred_cls_scores = pred_iou_scores = None
        if 'pred_cls_scores' in pred_dicts[b_idx]:
            pred_cls_scores = pred_dicts[b_idx][
                'pred_cls_scores'].detach().cpu().numpy()
        if 'pred_iou_scores' in pred_dicts[b_idx]:
            pred_iou_scores = pred_dicts[b_idx][
                'pred_iou_scores'].detach().cpu().numpy()

        if 'pred_boxes' in pred_dicts[b_idx]:
            # Exist predicted boxes passing self-training score threshold
            pred_boxes = pred_dicts[b_idx]['pred_boxes'].detach().cpu().numpy()
            pred_labels = pred_dicts[b_idx]['pred_labels'].detach().cpu().numpy()
            pred_scores = pred_dicts[b_idx]['pred_scores'].detach().cpu().numpy()

            if cfg.SELF_TRAIN.get('DIVERSITY_SAMPLING', None):
                before_nms_boxes = \
                    pred_dicts[b_idx]['pred_boxes_pre_nms'].detach().cpu().numpy()
            else:
                before_nms_boxes = None

            '''------------------ Cross-domain Examination (CDE) ------------------'''
            if cfg.SELF_TRAIN.get('CROSS_DOMAIN_DETECTION', None):
                pred_boxes_pnts = []
                batch_points = \
                    input_dict['points'][
                        input_dict['points'][:, 0] == b_idx][:,1:].cpu().numpy()
                internal_pnts_mask = \
                    points_in_boxes_cpu(batch_points, enlarge_box3d(pred_boxes[:, :7], extra_width=[1, 0.5, 0.5]))
                for msk_idx in range(pred_boxes.shape[0]):
                    pred_boxes_pnts.append(
                        batch_points[internal_pnts_mask[msk_idx] == 1])
                quality, selected, related_boxes_count = cross_domain_detection(
                    source_reader, pred_boxes, pred_scores, pred_labels,
                    pred_boxes_pnts, source_model,
                    input_dict['gt_boxes'][b_idx], before_nms_boxes, batch_points)

                quality_mask = np.zeros(len(pred_labels))
                quality_mask[selected] = True

                if cfg.SELF_TRAIN.CROSS_DOMAIN_DETECTION.WITH_IOU_SCORE:
                    if cfg.SELF_TRAIN.get('NEG_THRESH', None):
                        labels_remove_scores = np.array(cfg.SELF_TRAIN.NEG_THRESH)[np.abs(pred_labels) - 1]
                        remain_mask = pred_scores >= labels_remove_scores
                        pred_labels = pred_labels[remain_mask]
                        quality_mask = quality_mask[remain_mask]
                        pred_scores = pred_scores[remain_mask]
                        pred_boxes = pred_boxes[remain_mask]
                        if cfg.SELF_TRAIN.get('DIVERSITY_SAMPLING', None):
                            related_boxes_count = related_boxes_count[remain_mask]
                        if 'pred_cls_scores' in pred_dicts[b_idx]:
                            pred_cls_scores = pred_cls_scores[remain_mask]
                        if 'pred_iou_scores' in pred_dicts[b_idx]:
                            pred_iou_scores = pred_iou_scores[remain_mask]

                    labels_ignore_scores = np.array(cfg.SELF_TRAIN.SCORE_THRESH)[np.abs(pred_labels) - 1]

                    ignore_mask = np.logical_and(pred_scores < labels_ignore_scores, pred_labels > 0)
                    pred_labels[ignore_mask] = -pred_labels[ignore_mask]
                if cfg.SELF_TRAIN.get('DIVERSITY_SAMPLING', None):
                    """ Count ralated boxes """
                    related_boxes_count_list.append(related_boxes_count[pred_labels==1])

                gt_box = np.concatenate((pred_boxes,
                                         pred_labels.reshape(-1, 1),
                                         pred_scores.reshape(-1, 1)),
                                        axis=1)

            else: # Not using CDE
                # remove boxes under negative threshold
                if cfg.SELF_TRAIN.get('NEG_THRESH', None):
                    labels_remove_scores = np.array(cfg.SELF_TRAIN.NEG_THRESH)[pred_labels - 1]
                    remain_mask = pred_scores >= labels_remove_scores
                    pred_labels = pred_labels[remain_mask]
                    pred_scores = pred_scores[remain_mask]
                    pred_boxes = pred_boxes[remain_mask]
                    if 'pred_cls_scores' in pred_dicts[b_idx]:
                        pred_cls_scores = pred_cls_scores[remain_mask]
                    if 'pred_iou_scores' in pred_dicts[b_idx]:
                        pred_iou_scores = pred_iou_scores[remain_mask]

                labels_ignore_scores = np.array(cfg.SELF_TRAIN.SCORE_THRESH)[pred_labels - 1]
                ignore_mask = pred_scores < labels_ignore_scores
                pred_labels[ignore_mask] = -pred_labels[ignore_mask]

                gt_box = np.concatenate((pred_boxes,
                                         pred_labels.reshape(-1, 1),
                                         pred_scores.reshape(-1, 1)), axis=1)

        else:
            # no predicted boxes passes self-training score threshold
            gt_box = np.zeros((0, 9), dtype=np.float32)


        '''--------- Target ReD Sampling ---------'''
        gt_points = None
        if cfg.SELF_TRAIN.get('PS_SAMPLING',None) and \
                cfg.SELF_TRAIN.PS_SAMPLING.ENABLE:
            gt_points = []
            batch_points = \
                input_dict['points'][
                    input_dict['points'][:,0]==b_idx][:,1:].cpu().numpy()
            internal_pnts_mask = \
                points_in_boxes_cpu(batch_points, gt_box[:, :7])
            for msk_idx in range(gt_box.shape[0]):
                gt_points.append(batch_points[internal_pnts_mask[msk_idx]==1])


        '''Ground Truth Infos for Saving & Next Round Training'''
        gt_infos = {
            'gt_boxes': gt_box,
            'cls_scores': pred_cls_scores,
            'iou_scores': pred_iou_scores,
            'final_scores': pred_scores,
            'memory_counter': np.zeros(gt_box.shape[0])
        }
        if cfg.SELF_TRAIN.get('PS_SAMPLING',  None) and cfg.SELF_TRAIN.PS_SAMPLING.ENABLE:
            gt_infos.update({'gt_points': gt_points})
        if cfg.SELF_TRAIN.get('CROSS_DOMAIN_DETECTION', None) and cfg.SELF_TRAIN.CROSS_DOMAIN_DETECTION.ENABLE:
            gt_infos.update({'quality_mask': quality_mask})
        if related_boxes_count_list is not None:
            gt_infos.update({'related_box_count': related_boxes_count if related_boxes_count_list is not None else None})

        # record pseudo label to pseudo label dict
        if need_update:
            ensemble_func = getattr(memory_ensemble_utils, cfg.SELF_TRAIN.MEMORY_ENSEMBLE.NAME)
            gt_infos = memory_ensemble_utils.memory_ensemble(
                PSEUDO_LABELS[input_dict['frame_id'][b_idx]], gt_infos,
                cfg.SELF_TRAIN.MEMORY_ENSEMBLE, ensemble_func
            )
        # counter the number of ignore boxes for each class
        for i in range(ign_ps_nmeter.n):
            num_total_boxes = (np.abs(gt_infos['gt_boxes'][:, 7]) == (i+1)).sum()
            ign_ps_nmeter.update((gt_infos['gt_boxes'][:, 7] == -(i+1)).sum(), index=i)
            pos_ps_nmeter.update(num_total_boxes - ign_ps_nmeter.meters[i].val, index=i)
        NEW_PSEUDO_LABELS[input_dict['frame_id'][b_idx]] = gt_infos

    return pos_ps_nmeter, ign_ps_nmeter


def load_ps_label(frame_id):
    """
    :param frame_id: file name of pseudo label
    :return gt_box: loaded gt boxes (N, 9) [x, y, z, w, l, h, ry, label, scores]
    """
    if frame_id in PSEUDO_LABELS:
        gt_box = PSEUDO_LABELS[frame_id]['gt_boxes']
    else:
        raise ValueError('Cannot find pseudo label for frame: %s' % frame_id)

    return gt_box

def count_tp_fp_fn_gt(pred_boxes, gt_boxes, iou_thresh=0.7, points=None):
    """ Count the number of tp, fp, fn and gt. Return tp boxes and their corresponding gt boxes
    """
    quality_metric = {'gt': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'scale_err': 0}
    assert gt_boxes.shape[1] == 7 and pred_boxes.shape[1] == 7
    quality_metric['gt'] += gt_boxes.shape[0]

    if gt_boxes.shape[0] == 0:
        quality_metric['fp'] += pred_boxes.shape[0]
        return None, None
    elif pred_boxes.shape[0] == 0:
        quality_metric['fn'] += gt_boxes.shape[0]
        return None, None

    pred_boxes, _ = common_utils.check_numpy_to_torch(pred_boxes)
    gt_boxes, _ = common_utils.check_numpy_to_torch(gt_boxes)

    if not (pred_boxes.is_cuda and gt_boxes.is_cuda):
        pred_boxes, gt_boxes = pred_boxes.cuda(), gt_boxes.cuda()

    iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes[:, :7], gt_boxes[:, :7])
    max_ious, match_idx = torch.max(iou_matrix, dim=1)
    assert max_ious.shape[0] == pred_boxes.shape[0]

    # max iou > iou_thresh is tp
    tp_mask = max_ious >= iou_thresh
    ntps = tp_mask.sum().item()
    quality_metric['tp'] += ntps
    quality_metric['fp'] += max_ious.shape[0] - ntps

    # gt boxes that missed by tp boxes are fn boxes
    quality_metric['fn'] += gt_boxes.shape[0] - ntps

    # get tp boxes and their corresponding gt boxes
    tp_boxes = pred_boxes[tp_mask]
    tp_gt_boxes = gt_boxes[match_idx[tp_mask]]

    if ntps > 0:
        scale_diff, debug_boxes = cal_scale_diff(tp_boxes, tp_gt_boxes)
        quality_metric['scale_err'] += scale_diff

    return quality_metric, match_idx[tp_mask].cpu().numpy()

def cal_scale_diff(tp_boxes, gt_boxes):
    assert tp_boxes.shape[0] == gt_boxes.shape[0]

    aligned_tp_boxes = tp_boxes.detach().clone()

    # shift their center together
    aligned_tp_boxes[:, 0:3] = gt_boxes[:, 0:3]

    # align their angle
    aligned_tp_boxes[:, 6] = gt_boxes[:, 6]

    iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(aligned_tp_boxes[:, 0:7], gt_boxes[:, 0:7])

    max_ious, _ = torch.max(iou_matrix, dim=1)

    scale_diff = (1 - max_ious).sum().item()

    return scale_diff, aligned_tp_boxes.cpu().numpy()

def cross_domain_detection(source_reader, pred_boxes, pred_scores, pred_labels,
                           pred_boxes_pnts, model, target_gt_box, before_nms_boxes=None,
                           target_points=None):

    if cfg.SELF_TRAIN.CROSS_DOMAIN_DETECTION.ENABLE:
        b_idx = 0
        source_batch = source_reader.read_data()
        # get all points of a single PC from batch [# of point, 3]
        single_pc_pnts = \
            source_batch['points'][source_batch['points'][:, 0] == b_idx][:, 1:]
        s_gt_box = source_batch['gt_boxes'][b_idx] # source gt box

        s_gt_box = enlarge_box3d(s_gt_box, extra_width=[1, 0.5, 0.5])
        # Remove GT boxes and points in source PC [# of point, 3]
        single_pc_pnts = \
            remove_points_in_boxes3d(single_pc_pnts, s_gt_box[:,:7])

        # Remove points at PS boxes in source PC [# of point, 3]
        single_pc_pnts = \
            remove_points_in_boxes3d(single_pc_pnts, enlarge_box3d(pred_boxes[:, :7], extra_width=[1, 0.5, 0.5]))


        # remove all points of this single PC from batch [# of point, 4]
        source_batch['points'] = \
            source_batch['points'][source_batch['points'][:, 0] != b_idx]

        # Add PS objects points into source PC
        ps_pnts_to_sample = None
        for obj_pnts in pred_boxes_pnts:
            ps_pnts_to_sample = obj_pnts if ps_pnts_to_sample is None else np.concatenate([ps_pnts_to_sample, obj_pnts])
        try:
            single_pc_pnts = np.concatenate([ps_pnts_to_sample, single_pc_pnts])
        except ValueError:
            pass # ps_pnts_to_sample is None

        """  Rebuild voxels to batch"""
        config = cfg.DATA_CONFIG.DATA_PROCESSOR[-1]
        if config.get('VOXEL_SIZE', None):
            voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
                num_point_features=source_reader.dataloader.dataset.point_feature_encoder.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[source_reader.dataloader.dataset.mode]
            )

            voxel_output = voxel_generator.generate(single_pc_pnts)
            voxels, coordinates, num_points = voxel_output

            voxel_coords_single_pc = source_batch['voxel_coords'][
                source_batch['voxel_coords'][:, 0] == b_idx]
            source_batch['voxel_coords'] = source_batch['voxel_coords'][
                source_batch['voxel_coords'][:, 0] != b_idx]
            voxel_num = len(voxel_coords_single_pc)

            # Combine processed voxels to existing voxels,
            # remove existed voxels in this batch as well
            source_batch['voxels'] = \
                np.concatenate((voxels, source_batch['voxels'][voxel_num:]))
            source_batch['voxel_num_points'] = \
                np.concatenate((num_points,
                                source_batch['voxel_num_points'][voxel_num:]))
            batch_dim_to_cat = np.zeros(voxels.shape[0])
            batch_dim_to_cat[batch_dim_to_cat==0] = b_idx
            coordinates = np.concatenate(
                [batch_dim_to_cat.reshape(coordinates.shape[0], 1), coordinates],
                axis=1)
            source_batch['voxel_coords'] = \
                np.concatenate((coordinates, source_batch['voxel_coords']))

        """  Rebuild points to batch  """
        batch_dim_to_cat = np.zeros(single_pc_pnts.shape[0])
        batch_dim_to_cat[batch_dim_to_cat==0] = b_idx
        single_pc_pnts = np.concatenate(
            [batch_dim_to_cat.reshape(single_pc_pnts.shape[0], 1), single_pc_pnts],
            axis=1)
        source_batch['points'] = \
            np.concatenate([single_pc_pnts, source_batch['points']])

        load_data_to_gpu(source_batch)
        batch_pred_dict = model(source_batch)[0][b_idx]
        pred_boxes_from_source = batch_pred_dict['pred_boxes']
        pred_labels_from_source = batch_pred_dict['pred_labels']

        quality_metric, selected_ids = count_tp_fp_fn_gt(
            pred_boxes_from_source[:, 0:7],
            pred_boxes[:, 0:7],
            iou_thresh=cfg.SELF_TRAIN.CROSS_DOMAIN_DETECTION.CDE_IOU_TH
        )

    else:
        quality_metric=None
        selected_ids=np.array(range(0, pred_boxes.shape[0]), dtype='i')

    """ Counting OBC before NMS to each predicted boxes """
    pred_boxes, _ = common_utils.check_numpy_to_torch(pred_boxes)

    if cfg.SELF_TRAIN.get('DIVERSITY_SAMPLING', None):
        before_nms_boxes, _ = common_utils.check_numpy_to_torch(before_nms_boxes)
        if not (pred_boxes.is_cuda and before_nms_boxes.is_cuda):
            pred_boxes, before_nms_boxes = pred_boxes.cuda(), \
                                           before_nms_boxes.cuda()

        iou_matrix = iou3d_nms_utils.boxes_bev_iou_cpu(pred_boxes[:, :7].cpu().numpy(),
                                          before_nms_boxes.cpu().numpy())

        related_boxes = iou_matrix > 0.3
        related_boxes_count = (related_boxes != 0).sum(axis=1) # OBC

    return quality_metric, selected_ids, related_boxes_count



def vis(bg_points, gt_boxes = None, gt_labels = None, gt_scores = None,
        obj_points=None, ref_boxes=None, ref_labels=None, ref_scores=None,
        use_fakelidar=False, target_gt_box=None,
        selected_related_boxes_count=None, color='green'):
    label_names = {1: 'Vehicle', 2: 'Pedestrian', 3: 'Cyclist', 0: ' '}

    if isinstance(bg_points, torch.Tensor):
        bg_points = bg_points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().detach().numpy()
    if isinstance(ref_scores, torch.Tensor):
        scores = ref_scores.cpu().numpy()
    if isinstance(obj_points, torch.Tensor):
        obj_points = obj_points.cpu().numpy()
    if isinstance(gt_labels, torch.Tensor):
        gt_labels = gt_labels.cpu().numpy()
    if isinstance(gt_scores, torch.Tensor):
        gt_scores = gt_scores.cpu().numpy()
    if isinstance(gt_scores, torch.Tensor):
        gt_scores = gt_scores.cpu().numpy()
    if isinstance(target_gt_box, torch.Tensor):
        target_gt_box = target_gt_box.cpu().numpy()

    """------- Draw background points -------"""
    rgb = np.zeros((bg_points.shape[0],3))
    rgb[:, ] = [220,220,220] # bg points color is white
    bg_points_rgb = np.array([[p[0], p[1], p[2], c[0], c[1], c[2]] for p, c in zip(bg_points, rgb)])

    """------- Draw objects points -------"""
    if obj_points is not None:
        rgb = np.zeros((bg_points.shape[0], 3))
        rgb[:, ] = [0, 0, 255]  # bg points color is white
        obj_points_rgb = np.array(
            [[p[0], p[1], p[2], c[0], c[1], c[2]] for p, c in zip(obj_points, rgb)])

        points_rgb = np.concatenate((bg_points_rgb, obj_points_rgb))
    else:
        points_rgb = bg_points_rgb

    boxes = []
    """------- Draw Reference boxes -------"""
    if ref_boxes is not None:
        for i in range(len(boxes_to_corners_3d(ref_boxes).tolist())):
            box = boxes_to_corners_3d(ref_boxes).tolist()[i]
            label = " "
            if ref_labels is not None:
                label = "PS {}".format(label_names[int(ref_labels[i])])
            if ref_scores is not None:
                label = "{:.2f}".format(round(ref_scores[i], 2))
            if ref_labels is not None and ref_scores is not None:
                label = "PS {}: {:.2f}".format(label_names[int(ref_labels[i])],
                                        round(ref_scores[i],2))

            boxes_ref_label = {
                "corners": list(box),
                # optionally customize each label
                "label": label,
                "color": [255, 255, 0],
            }
            boxes.append(boxes_ref_label)


    """------- Draw GT boxes -------"""
    for i in range(len(boxes_to_corners_3d(gt_boxes).tolist())):
        box = boxes_to_corners_3d(gt_boxes).tolist()[i]

        label = " "
        if gt_labels is not None:
            label = "{}".format(label_names[int(gt_labels[i])])
        if gt_scores is not None:
            label = "{:.2f}".format(round(gt_scores[i],2))
        if gt_labels is not None and gt_scores is not None:
            label = "{}: {:.2f}".format(label_names[int(gt_labels[i])],
                                    round(gt_scores[i], 2))
        if gt_labels is not None and gt_scores is not None and \
            selected_related_boxes_count is not None:
            label = "{}: {:.2f} {}".format(label_names[int(gt_labels[i])],
                                           round(gt_scores[i], 2),
                                           selected_related_boxes_count[i])
        elif selected_related_boxes_count is not None:
            label = "Count: {}".format(selected_related_boxes_count[i])

        box_color = [0, 255, 0] if color == 'green' else [255, 0, 0]
        boxes_true_label = {
            "corners": list(box),
            # optionally customize each label
            "label": label,
            "color": box_color, # green 0, 255, 0 red 255,0,0
        }
        boxes.append(boxes_true_label)



    if target_gt_box is not None:
        for i in range(len(boxes_to_corners_3d(target_gt_box[:,:7]).tolist())):
            box = boxes_to_corners_3d(target_gt_box[:,:7]).tolist()[i]

            label = "GT {}".format(label_names[int(target_gt_box[:,-1][i])])

            boxes_tgt_label = {
                "corners": list(box),
                # optionally customize each label
                # "label": label,
                "color": [0, 255, 0],
            }
            boxes.append(boxes_tgt_label)

    boxes = np.array(boxes)

    wandb.log(
    {"3d point cloud":
        wandb.Object3D({
            "type": "lidar/beta",
            "points": points_rgb,
            "boxes": boxes
        })
    })



class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points

