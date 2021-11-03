import torch
import time
from datetime import datetime
from pathlib import Path
import xlwt

from src.dataset.data_loader import QAPDataset, get_dataloader
from src.evaluation_metric import objective_score
from src.parallel import DataParallel
from src.utils.model_sl import load_model
from src.utils.data_to_cuda import data_to_cuda

from src.utils.config import cfg


def eval_model(model, dataloader, eval_epoch=None, verbose=False):
    print('Start evaluation...')
    since = time.time()

    device = next(model.parameters()).device

    if eval_epoch is not None:
        model_path = str(Path(cfg.OUTPUT_PATH) / 'params' / 'params_{:04}.pt'.format(eval_epoch))
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)

    was_training = model.training
    model.eval()

    ds = dataloader.dataset
    classes = ds.classes
    cls_cache = ds.cls

    accs = torch.zeros(len(classes), device=device)

    wb = xlwt.Workbook()
    sheet = wb.add_sheet('QAPLIB')
    name_idx = 0
    score_idx = 1
    time_idx = 2
    sheet.write(0, name_idx, 'instance')
    sheet.write(0, score_idx, 'score')
    sheet.write(0, time_idx, 'time')
    wb_idx = 1

    for i, cls in enumerate(classes):
        if verbose:
            print('Evaluating class {}: {}/{}'.format(cls, i, len(classes)))

        running_since = time.time()
        iter_num = 0

        ds.cls = cls
        acc_match_num = torch.zeros(1, device=device)
        acc_total_num = torch.zeros(1, device=device)
        rel_sum = torch.zeros(1, device=device)
        rel_num = torch.zeros(1, device=device)
        for inputs in dataloader:
            if model.module.device != torch.device('cpu'):
                inputs = data_to_cuda(inputs)

            ori_affmtx = inputs['aff_mat']
            solution = inputs['solution']
            name = inputs['name']
            n1_gt, n2_gt = inputs['ns']
            perm_mat = inputs['gt_perm_mat']

            batch_num = perm_mat.size(0)

            iter_num = iter_num + 1

            fwd_since = time.time()

            if 'esc16f' in name:
                print('esc16f - 0')
                continue

            with torch.set_grad_enabled(False):
                _ = None
                pred = model(inputs)
                x_pred, affmtx = pred['perm_mat'], pred['aff_mat']

            fwd_time = time.time() - fwd_since

            obj_score = objective_score(x_pred, ori_affmtx)
            opt_obj_score = objective_score(perm_mat, ori_affmtx)
            ori_obj_score = solution

            for n, x, y, z in zip(name, obj_score, opt_obj_score, ori_obj_score):
                rel = (x - z) / x
                print('{} - Solved: {:.0f}, Feas: {:.0f}, Opt/Bnd: {:.0f}, Gap: {:.0f}, Rel: {:.4f}, time: {:.3f}'.
                      format(n, x, y, z, x - z, rel, fwd_time))
                if not torch.isnan(rel):
                    rel_sum += rel
                sheet.write(wb_idx, name_idx, n)
                sheet.write(wb_idx, score_idx, x.item())
                sheet.write(wb_idx, time_idx, fwd_time)
                wb_idx += 1
                #rel_num += 1

            if iter_num % cfg.STATISTIC_STEP == 0 and verbose:
                running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
                print('Class {:<8} Iteration {:<4} {:>4.2f}sample/s'.format(cls, iter_num, running_speed))
                running_since = time.time()

        accs[i] = acc_match_num / acc_total_num
        if verbose:
            print('Class {} acc = {:.4f}'.format(cls, accs[i]))

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    model.train(mode=was_training)
    ds.cls = cls_cache

    # print result
    print('mean relative: {:.4f}'.format(float(rel_sum / rel_num)))

    print('Matching accuracy')
    for cls, single_acc in zip(classes, accs):
        print('{} = {:.4f}'.format(cls, single_acc))
    print('average = {:.4f}'.format(torch.mean(accs)))

    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    wb.save( str(Path(cfg.OUTPUT_PATH) / ('eval_' + now_time + '.xls')))

    return accs


if __name__ == '__main__':
    from src.utils.dup_stdout_manager import DupStdoutFileManager
    from src.utils.parse_args import parse_args
    from src.utils.print_easydict import print_easydict

    args = parse_args('Deep learning of graph matching evaluation code.')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    torch.manual_seed(cfg.RANDOM_SEED)

    qap_dataset = QAPDataset(cfg.DATASET_FULL_NAME,
                             sets='test',
                             length=cfg.EVAL.SAMPLES,
                             pad=cfg.PAIR.PADDING,
                             obj_resize=cfg.PAIR.RESCALE)
    dataloader = get_dataloader(qap_dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    model = model.to(device)
    model = DataParallel(model, device_ids=cfg.GPUS)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('eval_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        classes = dataloader.dataset.classes
        accs = eval_model(model, dataloader,
                          eval_epoch=cfg.EVAL.EPOCH if cfg.EVAL.EPOCH != 0 else None,
                          verbose=True)
