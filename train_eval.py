import torch.cuda
import torch.optim as optim
import time
import xlwt
from datetime import datetime
from pathlib import Path
from tensorboardX import SummaryWriter

from src.dataset.data_loader import GMDataset, get_dataloader
from src.displacement_layer import Displacement
from src.loss_func import *
from src.evaluation_metric import matching_accuracy
from src.parallel import DataParallel
from src.utils.model_sl import load_model, save_model
from eval import eval_model
from src.lap_solvers.hungarian import hungarian
from src.utils.data_to_cuda import data_to_cuda

from src.utils.config import cfg
from pygmtools.benchmark import Benchmark


def train_eval_model(model,
                     criterion,
                     optimizer,
                     optimizer_k,
                     image_dataset,
                     dataloader,
                     tfboard_writer,
                     benchmark,
                     num_epochs=25,
                     start_epoch=0,
                     xls_wb=None):
    print('Start training...')

    since = time.time()
    dataset_size = len(dataloader['train'].dataset)
    displacement = Displacement()

    device = next(model.parameters()).device
    print('model on device: {}'.format(device))

    checkpoint_path = Path(cfg.OUTPUT_PATH) / 'params'
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    model_path, optim_path, optim_k_path = '', '', ''
    if start_epoch != 0:
        model_path = str(checkpoint_path / 'params_{:04}.pt'.format(start_epoch))
        optim_path = str(checkpoint_path / 'optim_{:04}.pt'.format(start_epoch))
        if optimizer_k is not None:
            optim_k_path = str(checkpoint_path / 'optim_k_{:04}.pt'.format(start_epoch))
    if len(cfg.PRETRAINED_PATH) > 0:
        model_path = cfg.PRETRAINED_PATH
    if len(model_path) > 0:
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path, strict=False)
    if len(optim_path) > 0:
        print('Loading optimizer state from {}'.format(optim_path))
        optimizer.load_state_dict(torch.load(optim_path))
    if len(optim_k_path) > 0:
        try:
            print('Loading optimizer_k state from {}'.format(optim_k_path))
            optimizer_k.load_state_dict(torch.load(optim_k_path))
        except FileNotFoundError:
            print('Creating new optimizer for AFA modules')

    if optimizer_k is not None:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=cfg.TRAIN.LR_STEP,
                                                   gamma=cfg.TRAIN.LR_DECAY,
                                                   last_epoch=-1)  # cfg.TRAIN.START_EPOCH - 1
        scheduler_k = optim.lr_scheduler.MultiStepLR(optimizer_k,
                                                   milestones=cfg.TRAIN.LR_STEP,
                                                   gamma=cfg.TRAIN.LR_DECAY,
                                                   last_epoch=-1)  # cfg.TRAIN.START_EPOCH - 1
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=cfg.TRAIN.LR_STEP,
                                                   gamma=cfg.TRAIN.LR_DECAY,
                                                   last_epoch=cfg.TRAIN.START_EPOCH - 1)

    for epoch in range(start_epoch, num_epochs):
        # Reset seed after evaluation per epoch
        torch.manual_seed(cfg.RANDOM_SEED + epoch + 1)
        dataloader['train'] = get_dataloader(image_dataset['train'], shuffle=True, fix_seed=False)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()  # Set model to training mode
        model.module.trainings = True

        print('lr = ' + ', '.join(['{:.2e}'.format(x['lr']) for x in optimizer.param_groups]))
        if optimizer_k is not None:
            print('K_regression_lr = ' + ', '.join(['{:.2e}'.format(x['lr']) for x in optimizer_k.param_groups]))

        epoch_loss = 0.0
        running_loss = 0.0
        running_ks_loss = 0.0
        running_ks_error = 0
        running_since = time.time()
        iter_num = 0

        # Iterate over data.
        for inputs in dataloader['train']:
            if iter_num >= cfg.TRAIN.EPOCH_ITERS:
                break
            if model.module.device != torch.device('cpu'):
                inputs = data_to_cuda(inputs)

            iter_num = iter_num + 1

            # zero the parameter gradients
            optimizer.zero_grad()
            if optimizer_k is not None:
                optimizer_k.zero_grad()

            with torch.set_grad_enabled(True):
                # torch.autograd.set_detect_anomaly(True)
                # forward
                if 'common' in cfg.MODEL_NAME: # COMMON use the iter number to control the warmup temperature
                    outputs = model(inputs, training=True, iter_num=iter_num, epoch=epoch)
                else:
                    outputs = model(inputs)
                if cfg.PROBLEM.TYPE == '2GM':
                    assert 'ds_mat' in outputs
                    assert 'perm_mat' in outputs
                    assert 'gt_perm_mat' in outputs

                    # compute loss
                    if cfg.TRAIN.LOSS_FUNC == 'offset':
                        d_gt, grad_mask = displacement(outputs['gt_perm_mat'], *outputs['Ps'], outputs['ns'][0])
                        d_pred, _ = displacement(outputs['ds_mat'], *outputs['Ps'], outputs['ns'][0])
                        loss = criterion(d_pred, d_gt, grad_mask)
                    elif cfg.TRAIN.LOSS_FUNC in ['perm', 'ce', 'hung', 'ilp']:
                        loss = criterion(outputs['ds_mat'], outputs['gt_perm_mat'], *outputs['ns'])
                    elif cfg.TRAIN.LOSS_FUNC == 'hamming':
                        loss = criterion(outputs['perm_mat'], outputs['gt_perm_mat'])
                    elif cfg.TRAIN.LOSS_FUNC == 'custom':
                        loss = torch.sum(outputs['loss'])
                    else:
                        raise ValueError(
                            'Unsupported loss function {} for problem type {}'.format(cfg.TRAIN.LOSS_FUNC,
                                                                                      cfg.PROBLEM.TYPE))
                    if 'ks_loss' in outputs:
                        ks_loss = outputs['ks_loss']
                        ks_error = outputs['ks_error']

                    # compute accuracy
                    acc = matching_accuracy(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'], idx=0)

                elif cfg.PROBLEM.TYPE in ['MGM', 'MGM3']:
                    assert 'ds_mat_list' in outputs
                    assert 'graph_indices' in outputs
                    assert 'perm_mat_list' in outputs
                    if not 'gt_perm_mat_list' in outputs:
                        assert 'gt_perm_mat' in outputs
                        gt_perm_mat_list = [outputs['gt_perm_mat'][idx] for idx in outputs['graph_indices']]
                    else:
                        gt_perm_mat_list = outputs['gt_perm_mat_list']

                    # compute loss & accuracy
                    if cfg.TRAIN.LOSS_FUNC in ['perm', 'ce' 'hung']:
                        loss = torch.zeros(1, device=model.module.device)
                        ns = outputs['ns']
                        for s_pred, x_gt, (idx_src, idx_tgt) in \
                                zip(outputs['ds_mat_list'], gt_perm_mat_list, outputs['graph_indices']):
                            l = criterion(s_pred, x_gt, ns[idx_src], ns[idx_tgt])
                            loss += l
                        loss /= len(outputs['ds_mat_list'])
                    elif cfg.TRAIN.LOSS_FUNC == 'plain':
                        loss = torch.sum(outputs['loss'])
                    else:
                        raise ValueError(
                            'Unsupported loss function {} for problem type {}'.format(cfg.TRAIN.LOSS_FUNC,
                                                                                      cfg.PROBLEM.TYPE))

                    # compute accuracy
                    acc = torch.zeros(1, device=model.module.device)
                    for x_pred, x_gt, (idx_src, idx_tgt) in \
                            zip(outputs['perm_mat_list'], gt_perm_mat_list, outputs['graph_indices']):
                        a = matching_accuracy(x_pred, x_gt, ns, idx=idx_src)
                        acc += torch.sum(a)
                    acc /= len(outputs['perm_mat_list'])
                else:
                    raise ValueError('Unknown problem type {}'.format(cfg.PROBLEM.TYPE))

                # backward + optimize
                if cfg.FP16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    # with torch.autograd.detect_anomaly():
                    loss.backward()
                    if optimizer_k is not None:
                        ks_loss.backward()
                # for n, p in model.named_parameters():
                #     if p.grad is not None and torch.any(torch.isnan(p.grad)):
                #         print('NaN!!!')
                #         print('name:', n, '-->require_grad:', p.requires_grad)
                optimizer.step()
                if optimizer_k is not None:
                    optimizer_k.step()

                batch_num = inputs['batch_size']

                # tfboard writer
                loss_dict = dict()
                loss_dict['loss'] = loss.item()
                tfboard_writer.add_scalars('loss', loss_dict, epoch * cfg.TRAIN.EPOCH_ITERS + iter_num)

                accdict = dict()
                accdict['matching accuracy'] = torch.mean(acc)
                tfboard_writer.add_scalars(
                    'training accuracy',
                    accdict,
                    epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                )

                # statistics
                running_loss += loss.item() * batch_num
                epoch_loss += loss.item() * batch_num
                if 'ks_loss' in outputs:
                    running_ks_loss += ks_loss * batch_num
                    running_ks_error += ks_error * batch_num

                if iter_num % cfg.STATISTIC_STEP == 0:
                    running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
                    print('Epoch {:<4} Iteration {:<4} {:>4.2f}sample/s Loss={:<8.4f} Ks_Loss={:<8.4f} Ks_Error={:<8.4f}'
                          .format(epoch, iter_num, running_speed, running_loss / cfg.STATISTIC_STEP / batch_num, running_ks_loss / cfg.STATISTIC_STEP / batch_num, running_ks_error / cfg.STATISTIC_STEP / batch_num))
                    tfboard_writer.add_scalars(
                        'speed',
                        {'speed': running_speed},
                        epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                    )

                    tfboard_writer.add_scalars(
                        'learning rate',
                        {'lr_{}'.format(i): x['lr'] for i, x in enumerate(optimizer.param_groups)},
                        epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                    )

                    running_loss = 0.0
                    running_ks_loss = 0.0
                    running_ks_error = 0.0
                    running_since = time.time()

        epoch_loss = epoch_loss / cfg.TRAIN.EPOCH_ITERS / batch_num

        save_model(model, str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
        torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))
        if optimizer_k is not None:
            torch.save(optimizer_k.state_dict(), str(checkpoint_path / 'optim_k_{:04}.pt'.format(epoch + 1)))

        print('Epoch {:<4} Loss: {:.4f}'.format(epoch, epoch_loss))
        print()

        # Eval in each epoch
        if dataloader['test'].dataset.cls not in ['none', 'all', None]:
            clss = [dataloader['test'].dataset.cls]
        else:
            clss = dataloader['test'].dataset.bm.classes
        l_e = (epoch == (num_epochs - 1))
        accs = eval_model(model, clss, benchmark['test'], l_e,
                          xls_sheet=xls_wb.add_sheet('epoch{}'.format(epoch + 1)))
        acc_dict = {"{}".format(cls): single_acc for cls, single_acc in zip(dataloader['test'].dataset.classes, accs)}
        acc_dict['average'] = torch.mean(accs)
        tfboard_writer.add_scalars(
            'Eval acc',
            acc_dict,
            (epoch + 1) * cfg.TRAIN.EPOCH_ITERS
        )
        wb.save(wb.__save_path)

        scheduler.step()
        if optimizer_k is not None:
            scheduler_k.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'
          .format(time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))

    return model


if __name__ == '__main__':
    from src.utils.dup_stdout_manager import DupStdoutFileManager
    from src.utils.parse_args import parse_args
    from src.utils.print_easydict import print_easydict

    args = parse_args('Deep learning of graph matching training & evaluation code.')

    import importlib

    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    torch.manual_seed(cfg.RANDOM_SEED)

    dataset_len = {'train': cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, 'test': cfg.EVAL.SAMPLES}
    ds_dict = cfg[cfg.DATASET_FULL_NAME] if ('DATASET_FULL_NAME' in cfg) and (cfg.DATASET_FULL_NAME in cfg) else {}
    benchmark = {
        x: Benchmark(name=cfg.DATASET_FULL_NAME,
                     sets=x,
                     problem=cfg.PROBLEM.TYPE,
                     obj_resize=cfg.PROBLEM.RESCALE,
                     filter=cfg.PROBLEM.FILTER,
                     **ds_dict)
        for x in ('train', 'test')}

    image_dataset = {
        x: GMDataset(cfg.DATASET_FULL_NAME,
                     benchmark[x],
                     dataset_len[x],
                     cfg.PROBLEM.TRAIN_ALL_GRAPHS if x == 'train' else cfg.PROBLEM.TEST_ALL_GRAPHS,
                     cfg.TRAIN.CLASS if x == 'train' else cfg.EVAL.CLASS,
                     cfg.PROBLEM.TYPE)
        for x in ('train', 'test')}
    dataloader = {x: get_dataloader(image_dataset[x], shuffle=True, fix_seed=(x == 'test'))
                  for x in ('train', 'test')}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    model = model.to(device)

    if cfg.TRAIN.LOSS_FUNC.lower() == 'offset':
        criterion = OffsetLoss(norm=cfg.TRAIN.RLOSS_NORM)
    elif cfg.TRAIN.LOSS_FUNC.lower() == 'perm':
        criterion = PermutationLoss()
    elif cfg.TRAIN.LOSS_FUNC.lower() == 'ce':
        criterion = CrossEntropyLoss()
    elif cfg.TRAIN.LOSS_FUNC.lower() == 'focal':
        criterion = FocalLoss(alpha=.5, gamma=0.)
    elif cfg.TRAIN.LOSS_FUNC.lower() == 'hung':
        criterion = PermutationLossHung()
    elif cfg.TRAIN.LOSS_FUNC.lower() == 'hamming':
        criterion = HammingLoss()
    elif cfg.TRAIN.LOSS_FUNC.lower() == 'ilp':
        criterion = ILP_attention_loss()
    elif cfg.TRAIN.LOSS_FUNC.lower() == 'custom':
        criterion = None
        print('NOTE: You are setting the loss function as \'custom\', please ensure that there is a tensor with key '
              '\'loss\' in your model\'s returned dictionary.')
    else:
        raise ValueError('Unknown loss function {}'.format(cfg.TRAIN.LOSS_FUNC))

    optimizer_k = None

    if cfg.TRAIN.SEPARATE_BACKBONE_LR:
        if not cfg.TRAIN.SEPARATE_K_LR:
            backbone_ids = [id(item) for item in model.backbone_params]
            other_params = [param for param in model.parameters() if id(param) not in backbone_ids]

            model_params = [
                {'params': other_params},
                {'params': model.backbone_params, 'lr': cfg.TRAIN.BACKBONE_LR}
            ]
        else:
            backbone_ids = [id(item) for item in model.backbone_params]
            k_params = model.k_params_id
            other_params = [param for param in model.parameters() if id(param) not in k_params and id(param) not in backbone_ids]

            model_params = [
                {'params': other_params},
                {'params': model.backbone_params, 'lr': cfg.TRAIN.BACKBONE_LR}
            ]
            k_reg_params = model.k_params
            optimizer_k = optim.Adam(k_reg_params, lr=cfg.TRAIN.K_LR)

    else:
        model_params = model.parameters()

    if cfg.TRAIN.OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(model_params, lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, nesterov=True)
    elif cfg.TRAIN.OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(model_params, lr=cfg.TRAIN.LR)
    else:
        raise ValueError('Unknown optimizer {}'.format(cfg.TRAIN.OPTIMIZER))

    if cfg.FP16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to enable FP16.")
        model, optimizer = amp.initialize(model, optimizer)

    model = DataParallel(model, device_ids=cfg.GPUS)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)

    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    tfboardwriter = SummaryWriter(logdir=str(Path(cfg.OUTPUT_PATH) / 'tensorboard' / 'training_{}'.format(now_time)))
    wb = xlwt.Workbook()
    wb.__save_path = str(Path(cfg.OUTPUT_PATH) / ('train_eval_result_' + now_time + '.xls'))

    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('train_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        model = train_eval_model(model, criterion, optimizer, optimizer_k, image_dataset, dataloader, tfboardwriter, benchmark,
                                 num_epochs=cfg.TRAIN.NUM_EPOCHS,
                                 start_epoch=cfg.TRAIN.START_EPOCH,
                                 xls_wb=wb)

    wb.save(wb.__save_path)
