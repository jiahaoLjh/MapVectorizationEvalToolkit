import json
import argparse
import numpy as np
import cv2
from mmcv import Config

from map_evaluator import MapEvaluator


def instance_to_segm_polygon(params, cfg):
    normalized_params = np.copy(params)
    normalized_params[:, 0] = (normalized_params[:, 0] - cfg.PC_RANGE[0]) / (cfg.PC_RANGE[3] - cfg.PC_RANGE[0]) * cfg.WIDTH
    normalized_params[:, 1] = (normalized_params[:, 1] - cfg.PC_RANGE[1]) / (cfg.PC_RANGE[4] - cfg.PC_RANGE[1]) * cfg.HEIGHT
    points = np.around(normalized_params).astype(int)
    mask = np.zeros([cfg.HEIGHT, cfg.WIDTH])
    cv2.fillPoly(mask, [points], 1)
    return (mask > 0).astype(float)


def instance_to_segm_line(params, cfg):
    normalized_params = np.copy(params)
    normalized_params[:, 0] = (normalized_params[:, 0] - cfg.PC_RANGE[0]) / (cfg.PC_RANGE[3] - cfg.PC_RANGE[0]) * cfg.WIDTH
    normalized_params[:, 1] = (normalized_params[:, 1] - cfg.PC_RANGE[1]) / (cfg.PC_RANGE[4] - cfg.PC_RANGE[1]) * cfg.HEIGHT
    points = np.around(normalized_params).astype(int)
    mask = np.zeros([cfg.HEIGHT, cfg.WIDTH])
    cv2.polylines(mask, [points], False, 1, 1)
    return (mask > 0).astype(float)


def postprocess_instance_polygon(gt_instances, dt_instances, cfg):
    gts = []
    dts = []

    for instance in gt_instances:
        if instance['class'] in cfg.POLYGON_CLASSES.values():
            gts.append({
                'class': instance['class'],
                'segmentation': instance_to_segm_polygon(instance['pts'], cfg),
            })

    for instance in dt_instances:
        if instance['score'] < cfg.INST_THRESHOLD_POLYGON or len(instance['pts']) < 2:
            continue
        if instance['class'] in cfg.POLYGON_CLASSES.values():
            dts.append({
                "class": instance['class'],
                "segmentation": instance_to_segm_polygon(instance['pts'], cfg),
                "score": instance['score'],
            })

    return gts, dts


def postprocess_instance_line(gt_instances, dt_instances, cfg):
    gts = []
    dts = []

    for instance in gt_instances:
        if instance['class'] in cfg.LINE_CLASSES.values():
            gts.append({
                'class': instance['class'],
                'segmentation': instance_to_segm_line(instance['pts'], cfg),
            })

    for instance in dt_instances:
        if instance['score'] < cfg.INST_THRESHOLD_LINE or len(instance['pts']) < 2:
            continue
        if instance['class'] in cfg.LINE_CLASSES.values():
            dts.append({
                "class": instance['class'],
                "segmentation": instance_to_segm_line(instance['pts'], cfg),
                "score": instance['score'],
            })

    return gts, dts


def parse_args():
    parser = argparse.ArgumentParser(description='Map Evaluation Toolkit')
    parser.add_argument('--cfg', help='config file path')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.cfg)

    with open('./sample_data.json', 'r') as f:
        sample_data = json.load(f)

    iouThrs_line = np.linspace(cfg.IOUTHRS_LINE.start, cfg.IOUTHRS_LINE.end, int(np.round((cfg.IOUTHRS_LINE.end - cfg.IOUTHRS_LINE.start) / cfg.IOUTHRS_LINE.step)) + 1)
    iouThrs_polygon = np.linspace(cfg.IOUTHRS_POLYGON.start, cfg.IOUTHRS_POLYGON.end, int(np.round((cfg.IOUTHRS_POLYGON.end - cfg.IOUTHRS_POLYGON.start) / cfg.IOUTHRS_POLYGON.step)) + 1)
    evaluators = [dict(type='line',
                       evaluator=MapEvaluator(classes=cfg.LINE_CLASSES,
                                              parameterization='instanceseg',
                                              iouThrs=iouThrs_line,
                                              maxDets=cfg.MAXDETS_LINE,
                                              query=cfg.QUERY_LINE,
                                              dilation=cfg.DILATION_LINE,
                                              to_thin=False,
                                              width=cfg.WIDTH,
                                              height=cfg.HEIGHT),
                       postprocess_func=postprocess_instance_line,
                       per_frame_results=list()),
                  dict(type='polygon',
                       evaluator=MapEvaluator(classes=cfg.POLYGON_CLASSES,
                                              parameterization='instanceseg',
                                              iouThrs=iouThrs_polygon,
                                              maxDets=cfg.MAXDETS_POLYGON,
                                              query=cfg.QUERY_POLYGON,
                                              dilation=cfg.DILATION_POLYGON,
                                              to_thin=False,
                                              width=cfg.WIDTH,
                                              height=cfg.HEIGHT),
                       postprocess_func=postprocess_instance_polygon,
                       per_frame_results=list())]

    for sample in sample_data:
        gt = sample['gt']  # list of dict(class=..., pts=...)
        dt = sample['dt']  # list of dict(class=..., pts=..., score=...)
        for evaluator in evaluators:
            evaluator['per_frame_results'].append(evaluator['evaluator'].evaluate(*evaluator['postprocess_func'](gt, dt, cfg)))

    eval_results = dict()
    for evaluator in evaluators:
        eval_results[evaluator['type']] = evaluator['evaluator'].summarize(evaluator['per_frame_results'])

    with open('./eval_results.json', 'w') as f:
        json.dump(eval_results, f, indent=4)
    print('Results written to eval_results.json')


if __name__ == '__main__':
    main()
