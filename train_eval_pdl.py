import paddle
import paddle.optimizer as optim
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from visualdl import LogWriter

from data.data_loader_pdl import GMDataset, get_dataloader
from models.GMN.displacement_layer_pdl import Displacement
from src.utils_pdl.offset_loss import RobustLoss
from src.utils_pdl.permutation_loss import CrossEntropyLoss
from src.utils_pdl.evaluation_metric import matching_accuracy
#from parallel import DataParallel
from src.utils_pdl.model_sl import load_model, save_model
from eval_pdl import eval_model
from src.lap_solvers_pdl.hungarian import hungarian

from src.utils.config import cfg


def train_eval_model(model,
                     criterion,
                     optimizer,
                     scheduler,
                     dataloader,
                     writer,
                     num_epochs=25,
                     resume=False,
                     start_epoch=0):
    print('Start training...')

    since = time.time()
    dataset_size = len(dataloader['train'].dataset)
    displacement = Displacement()
    lap_solver = hungarian
    """
    device = next(model.parameters()).device
    print('model on device: {}'.format(device))
    """
    paddle.set_device("gpu:0")

    checkpoint_path = Path(cfg.OUTPUT_PATH) / 'params'
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    if resume:
        assert start_epoch != 0
        model_path = str(checkpoint_path / 'params_{:04}.pdparams'.format(start_epoch))
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)

        optim_path = str(checkpoint_path / 'optim_{:04}.pdopt'.format(start_epoch))
        print('Loading optimizer state from {}'.format(optim_path))
        optimizer.set_state_dict(paddle.load(optim_path))


    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()  # Set model to training mode

        print('lr = ' + ', '.join(['{:.2e}'.format(scheduler.get_lr()) ]))

        epoch_loss = 0.0
        running_loss = 0.0
        running_since = time.time()
        iter_num = 0

        # Iterate over data.
        for inputs in dataloader['train']:
            if 'images' in inputs:
                data1, data2 = [paddle.to_tensor(data=_.numpy()) for _ in inputs['images']]
                inp_type = 'img'
            elif 'features' in inputs:
                data1, data2 = [paddle.to_tensor(data=_.numpy()) for _ in inputs['features']]
                inp_type = 'feat'
            else:
                raise ValueError('no valid data key (\'images\' or \'features\') found from dataloader!')
            P1_gt, P2_gt = [paddle.to_tensor(data=_, dtype='float32') for _ in inputs['Ps']]
            n1_gt, n2_gt = [paddle.to_tensor(data=_, dtype='int32') for _ in inputs['ns']]
            e1_gt, e2_gt = [paddle.to_tensor(data=_, dtype='float32') for _ in inputs['es']]
            G1_gt, G2_gt = [paddle.to_tensor(data=_, dtype='float32') for _ in inputs['Gs']]
            H1_gt, H2_gt = [paddle.to_tensor(data=_, dtype='float32') for _ in inputs['Hs']]
            KG, KH = [paddle.to_tensor(data=_.numpy(), dtype='float32') for _ in inputs['Ks']]
            perm_mat = paddle.to_tensor(inputs['gt_perm_mat'].numpy(), dtype='float32')

            iter_num = iter_num + 1

            # zero the parameter gradients
            # forward
            with paddle.set_grad_enabled(True):
                s_pred, d_pred = \
                    model(data1, data2, P1_gt, P2_gt, G1_gt, G2_gt, H1_gt, H2_gt, n1_gt, n2_gt, KG, KH, inp_type)

                multi_loss = []
                if cfg.TRAIN.LOSS_FUNC == 'offset':
                    d_gt, grad_mask = displacement(perm_mat, P1_gt, P2_gt, n1_gt)
                    loss = criterion(d_pred, d_gt, grad_mask)
                elif cfg.TRAIN.LOSS_FUNC == 'perm':
                    loss = criterion(s_pred, perm_mat, n1_gt, n2_gt)
                else:
                    raise ValueError('Unknown loss function {}'.format(cfg.TRAIN.LOSS_FUNC))

                # backward + optimize
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
                """
                if cfg.MODULE == 'NGM.hypermodel':
                    writer.add_scalars(
                        'weight',
                        {'w2': model.module.weight2, 'w3': model.module.weight3},
                        epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                    )
                """

                # training accuracy statistic
                acc, _, __ = matching_accuracy(lap_solver(s_pred, n1_gt, n2_gt), perm_mat, n1_gt)

                """
                # tfboard writer
                loss_dict = {'loss_{}'.format(i): l.item() for i, l in enumerate(multi_loss)}
                loss_dict['loss'] = loss.item()
                tfboard_writer.add_scalars('loss', loss_dict, epoch * cfg.TRAIN.EPOCH_ITERS + iter_num)
                accdict = dict()
                accdict['matching accuracy'] = acc
                tfboard_writer.add_scalars(
                    'training accuracy',
                    accdict,
                    epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                )
                """

                # statistics
                running_loss += loss.item() * perm_mat.shape[0]
                epoch_loss += loss.item() * perm_mat.shape[0]

                if iter_num % cfg.STATISTIC_STEP == 0:
                    running_speed = cfg.STATISTIC_STEP * perm_mat.shape[0]/ (time.time() - running_since)
                    print('Epoch {:<4} Iteration {:<4} {:>4.2f}sample/s Loss={:<8.4f}'
                          .format(epoch, iter_num, running_speed, running_loss / cfg.STATISTIC_STEP / perm_mat.shape[0]))
                    """
                    tfboard_writer.add_scalars(
                        'speed',
                        {'speed': running_speed},
                        epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                    )
                    """
                    running_loss = 0.0
                    running_since = time.time()

        epoch_loss = epoch_loss / dataset_size

        save_model(model, str(checkpoint_path / 'params_{:04}_wz_pretrain.pdparams'.format(epoch + 1)))
        paddle.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}_wz_pretrain.pdopt'.format(epoch + 1)))

        print('Epoch {:<4} Loss: {:.4f}'.format(epoch, epoch_loss))
        print()
        # Eval in each epoch
        if (epoch % 3 == 1) :
            accs = eval_model(model, dataloader['test'])
            acc_dict = {"{}".format(cls): single_acc for cls, single_acc in zip(dataloader['train'].dataset.classes, accs)}
            acc_dict['average'] = paddle.mean(accs)
            """
            tfboard_writer.add_scalars(
                'Eval acc',
                acc_dict,
                (epoch + 1) * cfg.TRAIN.EPOCH_ITERS
            )
            """

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

    #torch.manual_seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    paddle.seed(cfg.RANDOM_SEED)

    dataset_len = {'train': cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, 'test': cfg.EVAL.SAMPLES}
    image_dataset = {
        x: GMDataset(cfg.DATASET_FULL_NAME,
                    
                     dataset_len[x],
                     cfg.TRAIN.CLASS if x == 'train' else None,
		     sets=x,
                     obj_resize=cfg.PAIR.RESCALE)
        for x in ('train', 'test')}
    dataloader = {x: get_dataloader(image_dataset[x], fix_seed=(x == 'test'))
        for x in ('train', 'test')}

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Since Paddle has independent gpu and cpu version. Here by default use GPU
    # and now Paddle gpu ONLY support ONE gpu
    # so we recommend specifying GPU at Command Line
    paddle.set_device('gpu:0')

    model = Net()
    # model = model.cuda()
    # load pretrained params in torch
    #load_model(model, 'pretrained_vgg16_pca_voc.pdparams')

    if cfg.TRAIN.LOSS_FUNC == 'offset':
        criterion = RobustLoss(norm=cfg.TRAIN.RLOSS_NORM)
    elif cfg.TRAIN.LOSS_FUNC == 'perm':
        criterion = CrossEntropyLoss()
    else:
        raise ValueError('Unknown loss function {}'.format(cfg.TRAIN.LOSS_FUNC))

    scheduler = optim.lr.MultiStepDecay(      learning_rate=cfg.TRAIN.LR,
                                               milestones=cfg.TRAIN.LR_STEP,
                                               gamma=cfg.TRAIN.LR_DECAY,
                                               last_epoch=cfg.TRAIN.START_EPOCH - 1)

    optimizer = optim.Momentum(learning_rate=scheduler, momentum=cfg.TRAIN.MOMENTUM, parameters=model.parameters(),  use_nesterov=True)

    # model = DataParallel(model, device_ids=cfg.GPUS)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)

    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    boardwriter = LogWriter(logdir=str(Path(cfg.OUTPUT_PATH) / 'visualdl' / 'training_{}'.format(now_time)))

    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('train_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        model = train_eval_model(model, criterion, optimizer, scheduler, dataloader, boardwriter,
                                 num_epochs=cfg.TRAIN.NUM_EPOCHS,
                                 resume=cfg.TRAIN.START_EPOCH != 0,
                                 start_epoch=cfg.TRAIN.START_EPOCH)
