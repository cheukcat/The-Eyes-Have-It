import argparse
import numpy as np
import os
import os.path as osp
import sys
import time
import warnings

import torch
import torch.distributed as dist
from mmcv import Config
from mmcv.runner import build_optimizer
from mmseg.utils import get_root_logger
from pathlib import Path
from timm.scheduler import CosineLRScheduler

CUR_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(CUR_DIR))
from occnet import build_loss, build_model, build_dataloader
from occnet.core import Runner
from occnet.dataloader import get_nus_label_name
from occnet.utils import MeanIoU, revise_ckpt

warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass


def main(local_rank, args):
    # global settings
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    dataset_config = cfg.dataset_params
    ignore_label = dataset_config['ignore_label']
    version = dataset_config['version']
    train_dataloader_config = cfg.train_data_loader
    val_dataloader_config = cfg.val_data_loader

    max_num_epochs = cfg.max_epochs
    grid_size = cfg.grid_size

    # init DDP
    distributed = True
    ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = os.environ.get("MASTER_PORT", "20506")
    hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
    rank = int(os.environ.get("RANK", 0))  # node id
    gpus = torch.cuda.device_count()  # gpus per node
    print(f"tcp://{ip}:{port}")
    dist.init_process_group(
        backend="nccl", init_method=f"tcp://{ip}:{port}",
        world_size=hosts * gpus, rank=rank * gpus + local_rank
    )
    world_size = dist.get_world_size()
    cfg.gpu_ids = range(world_size)
    torch.cuda.set_device(local_rank)

    if dist.get_rank() != 0:
        import builtins
        builtins.print = pass_print

    # configure logger
    if dist.get_rank() == 0:
        os.makedirs(cfg.work_dir, exist_ok=True)
        cfg.dump(osp.join(cfg.work_dir, osp.basename(args.py_config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    model = build_model(cfg.model)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        model = ddp_model_module(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = model.cuda()
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location='cpu')
        model.load_state_dict(revise_ckpt(ckpt['state_dict']), strict=True)
    print('done ddp model')

    # generate datasets
    SemKITTI_label_name = get_nus_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(cfg.unique_label)
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label]

    # build dataloader
    train_dataloader, val_dataloader = \
        build_dataloader(
            dataset_config,
            train_dataloader_config,
            val_dataloader_config,
            grid_size=grid_size,
            version=version,
            dist=distributed,
            scale_rate=cfg.get('scale_rate', 1)
        )

    # build optimizer, loss_func, scheduler, evaluator
    optimizer = build_optimizer(model, cfg.optimizer)
    loss_func = build_loss(ignore_label=ignore_label)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=len(train_dataloader) * max_num_epochs,
        lr_min=1e-6,
        warmup_t=500,
        warmup_lr_init=1e-5,
        t_in_epochs=False
    )
    evaluator = MeanIoU(unique_label, ignore_label, unique_label_str)

    # build runner
    runner = Runner(cfg=cfg,
                    model=model,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    evaluator=evaluator,
                    loss_func=loss_func,
                    logger=logger,
                    rank=dist.get_rank())
    # run
    runner.eval_epoch(0)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/vanilla_occ/vanilla_occ_r50_fpn_1x_nus-trainval.py')
    parser.add_argument('--work-dir', type=str, default='work_dirs/occnet/')
    parser.add_argument('--resume-from', type=str, default='')

    args = parser.parse_args()

    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
