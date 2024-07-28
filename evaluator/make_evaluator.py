from importlib import import_module
from dataset.info import DatasetInfo


def _evaluator_factory(name, result_dir, anno_file, eval_format, comb_th, is_rcnn=False):
    file = import_module('evaluator.{}.snake'.format(name)) if not is_rcnn else import_module('evaluator.{}.rcnn_snake'.format(name))
    if eval_format == 'segm':
        if is_rcnn:
            evaluator = file.Evaluator(result_dir, anno_file)
        else:
            evaluator = file.Evaluator(result_dir, anno_file, comb_th)
    else:
        evaluator = file.DetectionEvaluator(result_dir, anno_file)
    return evaluator


def make_evaluator(cfg):
    name = cfg.test.dataset.split('_')[0]
    anno_file = DatasetInfo.dataset_info[cfg.test.dataset]['anno_dir']
    eval_format = cfg.test.segm_or_bbox
    comb_th =  cfg.test.combine_threshold if hasattr(cfg.test, 'combine_threshold') else 1
    is_rcnn = cfg.model.rcnn
    return _evaluator_factory(name, cfg.commen.result_dir, anno_file, eval_format, comb_th, is_rcnn)
