import time
from datetime import datetime
from pathlib import Path
import xlwt
import os
import sys
#sys.path.insert(0,"/home/lixinyang/.mscache/mindspore/ascend/1.2/vgg16")

from pygmtools.benchmark import Benchmark

from api.dataset.ms_dataloader import GMDataset, get_dataloader
from api.evaluation_metric import *
from api.utils.model_sl import load_model
from api.utils.timer import Timer

from api.utils.config import cfg

import hashlib
import numpy
import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore.ops as ops
import mindspore.numpy as np
from mindspore.common.tensor import Tensor
from mindspore import Model
import mindspore_hub as mshub
import mindspore.context as context

context.set_context(device_target="GPU")
mindspore.context.reset_auto_parallel_context()


def eval_model(model, classes, bm, verbose=False, xls_sheet=None):
    print('Start evaluation...')
    since = time.time()

    columns_names = ['P1', 'P2', 'n1', 'n2', 'e1', 'e2',
                     'G1', 'G2', 'H1', 'H2', 'A1', 'A2', 'univ_size0', 'univ_size1', 'img0', 'img1']
    dataloaders = []
    datasets = []

    for cls in classes:
        image_dataset = GMDataset(
            bm=bm,
            length=None,
            cls=cls,
            problem=cfg.PROBLEM.TYPE)
        mindspore.set_seed(cfg.RANDOM_SEED)
        dataloader = get_dataloader(image_dataset, columns_names)
        dataloaders.append(dataloader)
        datasets.append(image_dataset)

    recalls = []
    precisions = []
    f1s = []
    coverages = []
    pred_time = []
    objs = np.zeros(len(classes))
    cluster_acc = []
    cluster_purity = []
    cluster_ri = []

    timer = Timer()

    prediction = []

    for i, cls in enumerate(classes):
        if verbose:
            print('Evaluating class {}: {}/{}'.format(cls, i, len(classes)))

        running_since = time.time()
        iter_num = 0

        pred_time_list = []
        obj_total_num = np.zeros(1)
        cluster_acc_list = []
        cluster_purity_list = []
        cluster_ri_list = []
        prediction_cls = []
        for inputs_count, inputs in enumerate(dataloaders[i].create_dict_iterator()):

            batch_num = inputs['P1'].shape[0]
            if inputs_count >= cfg.EVAL.SAMPLES / batch_num:
                break

            inputs['img0'] = Tensor(inputs['img0'], mindspore.float32)
            inputs['img1'] = Tensor(inputs['img1'], mindspore.float32)
            inputs['P1'] = Tensor(inputs['P1'], mindspore.float32)
            inputs['P2'] = Tensor(inputs['P2'], mindspore.float32)
            inputs['n1'] = Tensor(inputs['n1'], mindspore.int32)
            inputs['n2'] = Tensor(inputs['n2'], mindspore.int32)
            inputs['A1'] = Tensor(inputs['A1'], mindspore.float32)
            inputs['A2'] = Tensor(inputs['A2'], mindspore.float32)

            iter_num = iter_num + 1

            timer.tick()
            outputs = Model(model).predict(inputs)

            pred_time_list.append(np.full((batch_num,), timer.toc() / batch_num))

            # Evaluate matching accuracy
            if cfg.PROBLEM.TYPE == '2GM':
                assert 'perm_mat' in outputs

                for p in range(outputs['perm_mat'].shape[0]):
                    perm_mat = outputs['perm_mat'].asnumpy()
                    perm_mat = perm_mat[p, :outputs['n1'][p].asnumpy().item(), :outputs['n2'][p].asnumpy().item()]
                    eval_dict = dict()
                    id_pair = datasets[i].hash2id[hashlib.md5(inputs['img0'][p].asnumpy()[0, 0]).hexdigest()], \
                              datasets[i].hash2id[hashlib.md5(inputs['img1'][p].asnumpy()[0, 0]).hexdigest()]
                    eval_dict['ids'] = id_pair
                    eval_dict['cls'] = cls
                    eval_dict['perm_mat'] = perm_mat
                    prediction.append(eval_dict)
                    prediction_cls.append(eval_dict)

                if 'aff_mat' in outputs:
                    pred_obj_score = objective_score(outputs['perm_mat'], outputs['aff_mat'], outputs['n1'])
                    gt_obj_score = objective_score(outputs['gt_perm_mat'], outputs['aff_mat'], outputs['n1'])
                    objs[i] += P.ReduceSum()(pred_obj_score / gt_obj_score)
                    obj_total_num += batch_num
            elif cfg.PROBLEM.TYPE in ['MGM', 'MGMC']:
                assert 'graph_indices' in outputs
                assert 'perm_mat_list' in outputs
                assert 'gt_perm_mat_list' in outputs

                ns = (outputs['n1'], outputs['n2'])
                for x_pred, x_gt, (idx_src, idx_tgt) in \
                        zip(outputs['perm_mat_list'], outputs['gt_perm_mat_list'], outputs['graph_indices']):
                    recall, _, __ = matching_accuracy(x_pred, x_gt, ns[idx_src])
                    recall_list.append(recall)
                    precision, _, __ = matching_precision(x_pred, x_gt, ns[idx_src])
                    precision_list.append(precision)
                    f1 = 2 * (precision * recall) / (precision + recall)
                    f1[np.isnan(f1)] = 0
                    f1_list.append(f1)
            else:
                raise ValueError('Unknown problem type {}'.format(cfg.PROBLEM.TYPE))

            # Evaluate clustering accuracy
            if cfg.PROBLEM.TYPE == 'MGMC':
                assert 'pred_cluster' in outputs
                assert 'cls' in outputs

                pred_cluster = outputs['pred_cluster']
                cls_gt_transpose = [[] for _ in range(batch_num)]
                for batched_cls in outputs['cls']:
                    for b, _cls in enumerate(batched_cls):
                        cls_gt_transpose[b].append(_cls)
                cluster_acc_list.append(clustering_accuracy(pred_cluster, cls_gt_transpose))
                cluster_purity_list.append(clustering_purity(pred_cluster, cls_gt_transpose))
                cluster_ri_list.append(rand_index(pred_cluster, cls_gt_transpose))

            if iter_num % cfg.STATISTIC_STEP == 0 and verbose:
                running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
                print('Class {} Iteration {:<4} {:>4.2f}sample/s'.format(cls, iter_num, running_speed))
                running_since = time.time()

        objs[i] = objs[i] / obj_total_num
        pred_time.append(P.Concat()(pred_time_list))
        if cfg.PROBLEM.TYPE == 'MGMC':
            cluster_acc.append(P.Concat()(cluster_acc_list))
            cluster_purity.append(P.Concat()(cluster_purity_list))
            cluster_ri.append(P.Concat()(cluster_ri_list))

        if verbose:
            if cfg.PROBLEM.TYPE != 'MGM3':
                bm.eval_cls(prediction_cls, cls, verbose=verbose)
            print('Class {} norm obj score = {:.4f}'.format(cls, objs[i].asnumpy().item()))
            print('Class {} pred time = {}s'.format(cls, format_metric(pred_time[i])))
            if cfg.PROBLEM.TYPE == 'MGMC':
                print('Class {} cluster acc={}'.format(cls, format_metric(cluster_acc[i])))
                print('Class {} cluster purity={}'.format(cls, format_metric(cluster_purity[i])))
                print('Class {} cluster rand index={}'.format(cls, format_metric(cluster_ri[i])))

    if cfg.PROBLEM.TYPE == 'MGM3':
        result = bm.eval(prediction, classes[0], verbose=True)
        for cls in classes[0]:
            precision = result[cls]['precision']
            recall = result[cls]['recall']
            f1 = result[cls]['f1']
            coverage = result[cls]['coverage']

            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
            coverages.append(coverage)
    else:
        result = bm.eval(prediction, classes, verbose=True)
        for cls in classes:
            precision = result[cls]['precision']
            recall = result[cls]['recall']
            f1 = result[cls]['f1']
            coverage = result[cls]['coverage']

            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
            coverages.append(coverage)

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if xls_sheet:
        for idx, cls in enumerate(classes):
            xls_sheet.write(0, idx+1, cls)
        xls_sheet.write(0, idx+2, 'mean')

    xls_row = 1

    # show result
    if xls_sheet:
        xls_sheet.write(xls_row, 0, 'precision')
        xls_sheet.write(xls_row+1, 0, 'recall')
        xls_sheet.write(xls_row+2, 0, 'f1')
        xls_sheet.write(xls_row + 3, 0, 'coverage')
    for idx, (cls, cls_p, cls_r, cls_f1, cls_cvg) in enumerate(zip(classes, precisions, recalls, f1s, coverages)):
        if xls_sheet:
            xls_sheet.write(xls_row, idx+1, cls_p.item())
            xls_sheet.write(xls_row+1, idx+1, cls_r.item())
            xls_sheet.write(xls_row+2, idx+1, cls_f1.item())
            xls_sheet.write(xls_row + 3, idx + 1, '{:.4f}'.format(cls_cvg))

    if xls_sheet:
        xls_sheet.write(xls_row, idx + 2, numpy.mean(numpy.array(precisions)).item())
        xls_sheet.write(xls_row + 1, idx + 2, numpy.mean(numpy.array(recalls)).item())
        xls_sheet.write(xls_row + 2, idx + 2, numpy.mean(numpy.array(f1s)).item())
        xls_row += 4

    if not np.isnan(objs).any:
        print('Normalized objective score')
        if xls_sheet: xls_sheet.write(xls_row, 0, 'norm objscore')
        for idx, (cls, cls_obj) in enumerate(zip(classes, objs)):
            print('{} = {:.4f}'.format(cls, cls_obj))
            if xls_sheet: xls_sheet.write(xls_row, idx+1, cls_obj.asnumpy().item())
        print('average objscore = {:.4f}'.format(np.mean(objs).asnumpy().item()))
        if xls_sheet:
            xls_sheet.write(xls_row, idx+2, np.mean(objs).asnumpy().item())
            xls_row += 1

    if cfg.PROBLEM.TYPE == 'MGMC':
        print('Clustering accuracy')
        if xls_sheet: xls_sheet.write(xls_row, 0, 'cluster acc')
        for idx, (cls, cls_acc) in enumerate(zip(classes, cluster_acc)):
            print('{} = {}'.format(cls, format_metric(cls_acc)))
            if xls_sheet: xls_sheet.write(xls_row, idx+1, np.mean(cls_acc).asnumpy().item())
        print('average clustering accuracy = {}'.format(format_metric(P.Concat()(cluster_acc))))
        if xls_sheet:
            xls_sheet.write(xls_row, idx+2, np.mean(P.Concat()(cluster_acc)).asnumpy().item())
            xls_row += 1

        print('Clustering purity')
        if xls_sheet: xls_sheet.write(xls_row, 0, 'cluster purity')
        for idx, (cls, cls_acc) in enumerate(zip(classes, cluster_purity)):
            print('{} = {}'.format(cls, format_metric(cls_acc)))
            if xls_sheet: xls_sheet.write(xls_row, idx+1, np.mean(cls_acc).asnumpy().item())
        print('average clustering purity = {}'.format(format_metric(P.Concat()(cluster_purity))))
        if xls_sheet:
            xls_sheet.write(xls_row, idx+2, np.mean(P.Concat()(cluster_purity)).asnumpy().item())
            xls_row += 1

        print('Clustering rand index')
        if xls_sheet: xls_sheet.write(xls_row, 0, 'rand index')
        for idx, (cls, cls_acc) in enumerate(zip(classes, cluster_ri)):
            print('{} = {}'.format(cls, format_metric(cls_acc)))
            if xls_sheet: xls_sheet.write(xls_row, idx+1, np.mean(cls_acc).asnumpy().item())
        print('average rand index = {}'.format(format_metric(P.Concat()(cluster_ri))))
        if xls_sheet:
            xls_sheet.write(xls_row, idx+2, np.mean(P.Concat()(cluster_ri)).asnumpy().item())
            xls_row += 1

    print('Predict time')
    if xls_sheet: xls_sheet.write(xls_row, 0, 'time')
    for idx, (cls, cls_time) in enumerate(zip(classes, pred_time)):
        print('{} = {}'.format(cls, format_metric(cls_time)))
        if xls_sheet: xls_sheet.write(xls_row, idx + 1, np.mean(cls_time).asnumpy().item())

    print('average time = {}'.format(format_metric(P.Concat()(pred_time))))
    if xls_sheet:
        xls_sheet.write(xls_row, idx+2, np.mean(P.Concat()(pred_time)).asnumpy().item())
        xls_row += 1

    return Tensor(recalls)


if __name__ == '__main__':
    from api.utils.dup_stdout_manager import DupStdoutFileManager
    from api.utils.parse_args import parse_args
    from api.utils.print_easydict import print_easydict
    from api.utils.count_model_params import count_parameters

    args = parse_args('Deep learning of graph matching evaluation code.')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    mindspore.set_seed(cfg.RANDOM_SEED)

    benchmark = Benchmark(name=cfg.DATASET_FULL_NAME,
                          sets='test',
                          problem=cfg.PROBLEM.TYPE,
                          obj_resize=cfg.PROBLEM.RESCALE,
                          filter=cfg.PROBLEM.FILTER)
    cls = None if cfg.EVAL.CLASS in ['none', 'all'] else cfg.EVAL.CLASS
    if cls is None:
        clss = benchmark.classes
    else:
        clss = [cls]

    model = Net()

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    wb = xlwt.Workbook()
    ws = wb.add_sheet('epoch{}'.format(cfg.EVAL.EPOCH))
    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('eval_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        print('Number of parameters: {:.2f}M'.format(count_parameters(model) / 1e6))

        model_path = ''
        if cfg.EVAL.EPOCH is not None and cfg.EVAL.EPOCH > 0:
            model_path = str(Path(cfg.OUTPUT_PATH) / 'params' / 'params_{:04}.pt'.format(cfg.EVAL.EPOCH))
        if len(cfg.PRETRAINED_PATH) > 0:
            model_path = cfg.PRETRAINED_PATH
        if len(model_path) > 0:
            print('Loading model parameters from {}'.format(model_path))
            load_model(model, model_path, strict=False)

        eval_model(
            model, clss,
            benchmark,
            verbose=True,
            xls_sheet=ws
        )
    wb.save(str(Path(cfg.OUTPUT_PATH) / ('eval_result_' + now_time + '.xls')))
