import paddle
import time
from datetime import datetime
from pathlib import Path

from src.lap_solvers_pdl.hungarian import hungarian
from data.data_loader_pdl import GMDataset, get_dataloader
from src.utils_pdl.evaluation_metric import matching_accuracy
#from parallel import DataParallel
from src.utils_pdl.model_sl import load_model

from src.utils.config import cfg


def eval_model(model, dataloader, eval_epoch=None, verbose=False):
    print('Start evaluation...')
    since = time.time()

    paddle.set_device('gpu:0')

    '''
    if eval_epoch is not None:
        model_path = str(Path(cfg.OUTPUT_PATH) / 'params' / 'params_{:04}.pdparams'.format(eval_epoch))
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)
    '''

    model.eval()

    ds = dataloader.dataset
    classes = ds.classes
    cls_cache = ds.cls

    lap_solver = hungarian

    accs = paddle.zeros([len(classes)])

    for i, cls in enumerate(classes):
        if verbose:
            print('Evaluating class {}: {}/{}'.format(cls, i, len(classes)))

        running_since = time.time()
        no_dataloader_time = 0
        iter_num = 0

        ds.cls = cls
        acc_match_num = paddle.zeros([1])
        acc_total_num = paddle.zeros([1])
        for inputs in dataloader:
            if 'images' in inputs:
                data1, data2 = [paddle.to_tensor(data=_) for _ in inputs['images']]
                inp_type = 'img'
            elif 'features' in inputs:
                data1, data2 = [paddle.to_tensor(data=_) for _ in inputs['features']]
                inp_type = 'feat'
            else:
                raise ValueError('no valid data key (\'images\' or \'features\') found from dataloader!')
            P1_gt, P2_gt = [paddle.to_tensor(data=_, dtype='float32') for _ in inputs['Ps']]
            n1_gt, n2_gt = [paddle.to_tensor(data=_, dtype='int32') for _ in inputs['ns']]
            e1_gt, e2_gt = [paddle.to_tensor(data=_, dtype='float32') for _ in inputs['es']]
            G1_gt, G2_gt = [paddle.to_tensor(data=_, dtype='float32') for _ in inputs['Gs']]
            H1_gt, H2_gt = [paddle.to_tensor(data=_, dtype='float32') for _ in inputs['Hs']]
            KG, KH = [paddle.to_tensor(data=_.numpy(), dtype='float32') for _ in inputs['Ks']]
            perm_mat = paddle.to_tensor(inputs['gt_perm_mat'], dtype='float32')

            batch_num = data1.shape[0]

            iter_num = iter_num + 1

            infer_start_time = time.time()
            with paddle.set_grad_enabled(False):
                s_pred, pred = \
                    model(data1, data2, P1_gt, P2_gt, G1_gt, G2_gt, H1_gt, H2_gt, n1_gt, n2_gt, KG, KH, inp_type)

                s_pred_perm = lap_solver(s_pred, n1_gt, n2_gt)

                _, _acc_match_num, _acc_total_num = matching_accuracy(s_pred_perm, perm_mat, n1_gt)
                acc_match_num += _acc_match_num
                acc_total_num += _acc_total_num

                infer_end_time = time.time()
                no_dataloader_time += infer_end_time - infer_start_time

                if iter_num % cfg.STATISTIC_STEP == 0 and verbose:
                    running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
                    print('Class {:<8} Iteration {:<4} {:>4.2f}sample/s'.format(cls, iter_num, running_speed))
                    running_since = time.time()

        accs[i] = acc_match_num / acc_total_num
        if verbose:
            print('Class {} acc = {:.4f}'.format(cls, float(accs[i].numpy())))
            print('Which takes {:.0f}m {:.0f}s'.format(no_dataloader_time // 60, no_dataloader_time %60))
        

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    model.train()
    ds.cls = cls_cache

    print('Matching accuracy')
    for cls, single_acc in zip(classes, accs):
        print('{} = {}'.format(cls, float(single_acc.numpy())))
    print('average = {}'.format(float(paddle.mean(accs).numpy())))

    return accs


if __name__ == '__main__':
    from src.utils.dup_stdout_manager import DupStdoutFileManager
    from src.utils.parse_args import parse_args
    from src.utils.print_easydict import print_easydict

    args = parse_args('Deep learning of graph matching evaluation code.')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    paddle.seed(cfg.RANDOM_SEED)
    paddle.set_device(device='gpu:0')

    image_dataset = GMDataset(cfg.DATASET_FULL_NAME,
                              sets='test',
                              length=cfg.EVAL.SAMPLES,
                              obj_resize=cfg.PAIR.RESCALE)
    dataloader = get_dataloader(image_dataset)

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    load_model(model, 'src/utils_pdl/pca.pdparams')
    #load_model(model, 'pretrained_vgg16_pca_voc.pdparams')
    #model = model.to(device)
    #model = DataParallel(model, device_ids=cfg.GPUS)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('eval_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        classes = dataloader.dataset.classes
        pcks = eval_model(model, dataloader,
                          eval_epoch=cfg.EVAL.EPOCH if cfg.EVAL.EPOCH != 0 else None,
                          verbose=True)


