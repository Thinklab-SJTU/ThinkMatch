import torch.optim as optim
import time
from datetime import datetime
from pathlib import Path
from tensorboardX import SummaryWriter

from src.dataset.data_loader import QAPDataset, get_dataloader
from src.loss_func import *
from src.evaluation_metric import objective_score
from src.parallel import DataParallel
from src.utils.model_sl import load_model, save_model
from eval_qap import eval_model
from src.utils.data_to_cuda import data_to_cuda

from src.utils.config import cfg


def train_eval_model(model,
                     criterion,
                     optimizer,
                     dataloader,
                     tfboard_writer,
                     num_epochs=25,
                     start_epoch=0):
    print('Start training...')

    since = time.time()
    dataset_size = len(dataloader['train'].dataset)

    device = next(model.parameters()).device
    print('model on device: {}'.format(device))

    alphas = torch.tensor(cfg.EVAL.PCK_ALPHAS, dtype=torch.float32, device=device)  # for evaluation

    checkpoint_path = Path(cfg.OUTPUT_PATH) / 'params'
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    model_path, optim_path = '',''
    if start_epoch > 0:
        model_path = str(checkpoint_path / 'params_{:04}.pt'.format(start_epoch))
        optim_path = str(checkpoint_path / 'optim_{:04}.pt'.format(start_epoch))
    if len(cfg.PRETRAINED_PATH) > 0:
        model_path = cfg.PRETRAINED_PATH
    if len(model_path) > 0:
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path, strict=False)
    if len(optim_path) > 0:
        print('Loading optimizer state from {}'.format(optim_path))
        optimizer.load_state_dict(torch.load(optim_path))

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg.TRAIN.LR_STEP,
                                               gamma=cfg.TRAIN.LR_DECAY,
                                               last_epoch=cfg.TRAIN.START_EPOCH - 1)

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()  # Set model to training mode

        print('lr = ' + ', '.join(['{:.2e}'.format(x['lr']) for x in optimizer.param_groups]))

        epoch_loss = 0.0
        running_loss = 0.0
        running_since = time.time()
        iter_num = 0

        det_anomaly = False

        # Iterate over data.
        for inputs in dataloader['train']:
            if model.module.device != torch.device('cpu'):
                inputs = data_to_cuda(inputs)

            n1_gt, n2_gt = inputs['ns']
            perm_mat = inputs['gt_perm_mat'].cuda()

            iter_num = iter_num + 1

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                with torch.autograd.set_detect_anomaly(det_anomaly):
                    # forward
                    pred = model(inputs)
                    s_pred, affmtx = pred['ds_mat'], pred['aff_mat']

                    if type(s_pred) is list:
                        s_pred = s_pred[-1]

                    multi_loss = []
                    if cfg.TRAIN.LOSS_FUNC == 'perm' or cfg.TRAIN.LOSS_FUNC == 'hung':
                        loss = criterion(s_pred, perm_mat, n1_gt, n2_gt)
                    elif cfg.TRAIN.LOSS_FUNC == 'obj':
                        loss = criterion(s_pred, affmtx)
                    elif cfg.TRAIN.LOSS_FUNC == 'plain':
                        loss = torch.sum(pred['loss'])
                    else:
                        raise ValueError('Unknown loss function {}'.format(cfg.TRAIN.LOSS_FUNC))

                    if cfg.FP16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    det_anomaly = False

                    for param in model.parameters():
                        if param.grad is not None and torch.any(torch.isnan(param.grad)):
                            det_anomaly = True
                            break
                    if not det_anomaly:
                        optimizer.step()

                    # training accuracy statistic
                    #acc, _, __ = matching_accuracy(lap_solver(s_pred, n1_gt, n2_gt), perm_mat, n1_gt)
                    acc = 0

                    # tfboard writer
                    loss_dict = {'loss_{}'.format(i): l.item() for i, l in enumerate(multi_loss)}
                    loss_dict['loss'] = loss.item()
                    tfboard_writer.add_scalars('loss', loss_dict, epoch * cfg.TRAIN.EPOCH_ITERS + iter_num)
                    #accdict = {'PCK@{:.2f}'.format(a): p for a, p in zip(alphas, pck)}
                    accdict = dict()
                    accdict['matching accuracy'] = acc
                    tfboard_writer.add_scalars(
                        'training accuracy',
                        accdict,
                        epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                    )

                    # statistics
                    running_loss += loss.item() * perm_mat.size(0)
                    epoch_loss += loss.item() * perm_mat.size(0)

                    if iter_num % cfg.STATISTIC_STEP == 0:
                        running_speed = cfg.STATISTIC_STEP * perm_mat.size(0) / (time.time() - running_since)
                        print('Epoch {:<4} Iteration {:<4} {:>4.2f}sample/s Loss={:<8.4f}'
                              .format(epoch, iter_num, running_speed, running_loss / cfg.STATISTIC_STEP / perm_mat.size(0)))
                        tfboard_writer.add_scalars(
                            'speed',
                            {'speed': running_speed},
                            epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                        )
                        running_loss = 0.0
                        running_since = time.time()

        epoch_loss = epoch_loss / dataset_size

        #loss_dict = dict()
        #loss_dict['loss'] = loss.item()
        #tfboard_writer.add_scalars('loss', loss_dict, epoch * dataset_size + iter_num)

        save_model(model, str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
        torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))

        print('Epoch {:<4} Loss: {:.4f}'.format(epoch, epoch_loss))
        print()

        # Eval in each epoch
        #print('training set')
        #accs = eval_model(model, alphas, dataloader['train'])
        #print()
        #print('testing set')
        accs = eval_model(model, alphas, dataloader['test'])
        acc_dict = {"{}".format(cls): single_acc for cls, single_acc in zip(dataloader['train'].dataset.classes, accs)}
        acc_dict['average'] = torch.mean(accs)
        tfboard_writer.add_scalars(
            'Eval acc',
            acc_dict,
            (epoch + 1) * cfg.TRAIN.EPOCH_ITERS
        )

        scheduler.step()

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
    qap_dataset = {
        x: QAPDataset(cfg.DATASET_FULL_NAME,
                      sets=x,
                      length=dataset_len[x],
                      cls=cfg.TRAIN.CLASS if x == 'train' else None,
                      fetch_online=False)
        for x in ('train', 'test')}
    dataloader = {x: get_dataloader(qap_dataset[x], fix_seed=(x == 'test'), shuffle=(x == 'train'))
        for x in ('train', 'test')}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    model = model.to(device)

    if cfg.TRAIN.LOSS_FUNC.lower() == 'perm':
        criterion = CrossEntropyLoss()
    elif cfg.TRAIN.LOSS_FUNC.lower() == 'obj':
        criterion = lambda *x: torch.mean(objective_score(*x))
    elif cfg.TRAIN.LOSS_FUNC.lower() == 'hung':
        criterion = PermutationLossHung()
    else:
        raise ValueError('Unknown loss function {}'.format(cfg.TRAIN.LOSS_FUNC))

    if cfg.TRAIN.OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, nesterov=True)
    elif cfg.TRAIN.OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
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

    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('train_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        model = train_eval_model(model, criterion, optimizer, dataloader, tfboardwriter,
                                 num_epochs=cfg.TRAIN.NUM_EPOCHS,
                                 start_epoch=cfg.TRAIN.START_EPOCH)
