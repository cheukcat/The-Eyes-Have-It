import os
import torch
import time
import mmcv
import numpy as np
import os.path as osp

from occnet.utils import revise_ckpt, revise_ckpt_2


class Runner:
    def __init__(self, cfg,
                 model,
                 train_dataloader,
                 val_dataloader,
                 optimizer,
                 scheduler,
                 evaluator,
                 loss_func,
                 logger,
                 rank=0
                 ):
        assert isinstance(loss_func, (tuple, list))
        self.cfg = cfg
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.loss_func = loss_func
        self.ignore_label = cfg.dataset_params.get('ignore_label', None)
        self.logger = logger
        self.work_dir = cfg.work_dir
        self.rank = rank
        self.best_val_miou = 0
        self.global_iter = 0

    def train_epoch(self, epoch):
        self.model.train()
        if hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(epoch)
        loss_list = []
        time.sleep(10)
        data_time_s = time.time()
        time_s = time.time()
        loss_ce, loss_lovasz = self.loss_func
        for i_iter, (imgs, img_metas, train_vox_label, train_grid, train_pt_labs) \
                in enumerate(self.train_dataloader):
            imgs = imgs.cuda()
            train_grid = train_grid.to(torch.float32).cuda()
            # train_pt_labs is not needed in training
            labels = train_vox_label.type(torch.LongTensor).cuda()
            # forward + backward + optimize
            data_time_e = time.time()
            output = self.model(img=imgs,
                                img_metas=img_metas,
                                points=train_grid)
            # compute loss
            loss = loss_lovasz(
                torch.nn.functional.softmax(output, dim=1),
                labels, ignore=self.ignore_label
            ) + loss_ce(output, labels)

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       self.cfg.grad_max_norm)
            self.optimizer.step()
            loss_list.append(loss.item())
            self.scheduler.step_update(self.global_iter)
            time_e = time.time()

            self.global_iter += 1
            if i_iter % self.cfg.print_freq == 0 and self.rank == 0:
                lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    '[TRAIN] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f), grad_norm: %.1f, lr: %.7f, time: %.3f (%.3f)' % (
                        epoch, i_iter, len(self.train_dataloader),
                        loss.item(), np.mean(loss_list), grad_norm, lr,
                        time_e - time_s, data_time_e - data_time_s
                    ))
            data_time_s = time.time()
            time_s = time.time()
        # save checkpoint
        if self.rank == 0:
            dict_to_save = {
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': epoch + 1,
                'global_iter': self.global_iter,
                'best_val_miou': self.best_val_miou
            }
            save_file_name = os.path.join(os.path.abspath(self.work_dir), f'epoch_{epoch + 1}.pth')
            torch.save(dict_to_save, save_file_name)
            dst_file = osp.join(self.work_dir, 'latest.pth')
            mmcv.symlink(save_file_name, dst_file)

    @torch.no_grad()
    def eval_epoch(self, epoch):
        self.model.eval()
        val_loss_list = []
        self.evaluator.reset()
        loss_ce, loss_lovasz = self.loss_func
        for i_iter_val, (imgs, img_metas, val_vox_label, val_grid, val_pts_label) \
                in enumerate(self.val_dataloader):
            imgs = imgs.cuda()
            labels = val_vox_label.cuda()
            val_grid_float = val_grid.to(torch.float32).cuda()
            val_grid_int = val_grid.to(torch.long).cuda()
            val_pts_label = val_pts_label.squeeze(-1).cpu()
            output = self.model(img=imgs,
                                img_metas=img_metas,
                                points=val_grid_float)
            # compute loss
            loss = loss_lovasz(
                torch.nn.functional.softmax(output, dim=1).detach(),
                labels, ignore=self.ignore_label
            ) + loss_ce(output.detach(), labels)

            output = torch.argmax(output, dim=1)
            output = output.detach().cpu()
            for count in range(len(val_grid_int)):
                self.evaluator.after_step(
                    output[count,
                           val_grid_int[count][:, 0],
                           val_grid_int[count][:, 1],
                           val_grid_int[count][:, 2]].flatten(),
                    val_pts_label[count])
            val_loss_list.append(loss.detach().cpu().numpy())
            if i_iter_val % self.cfg.print_freq == 0 and self.rank == 0:
                self.logger.info('[EVAL] Epoch %d Iter %5d: Loss: %.3f (%.3f)' % (
                    epoch, i_iter_val, loss.item(), np.mean(val_loss_list)))

        val_miou = self.evaluator.after_epoch()
        if self.best_val_miou < val_miou:
            self.best_val_miou = val_miou

        self.logger.info('Current val miou is %.3f while the best val miou vox is %.3f' %
                         (val_miou, self.best_val_miou))
        self.logger.info('Current val loss is %.3f' %
                         (np.mean(val_loss_list)))

    def run(self, max_epoch, resume_from=None):
        # resume and load
        epoch = 0
        self.cfg.resume_from = ''
        if osp.exists(osp.join(self.work_dir, 'latest.pth')):
            self.cfg.resume_from = osp.join(self.work_dir, 'latest.pth')
        if resume_from:
            self.cfg.resume_from = resume_from

        print('resume from: ', self.cfg.resume_from)
        print('work dir: ', self.work_dir)

        if self.cfg.resume_from and osp.exists(self.cfg.resume_from):
            map_location = 'cpu'
            ckpt = torch.load(self.cfg.resume_from, map_location=map_location)
            print(self.model.load_state_dict(revise_ckpt(ckpt['state_dict']), strict=False))
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            epoch = ckpt['epoch']
            if 'best_val_miou' in ckpt:
                self.best_val_miou = ckpt['best_val_miou']
            self.global_iter = ckpt['global_iter']
            print(f'successfully resumed from epoch {epoch}')
        elif self.cfg.load_from:
            ckpt = torch.load(self.cfg.load_from, map_location='cpu')
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt
            state_dict = revise_ckpt(state_dict)
            try:
                print(self.model.load_state_dict(state_dict, strict=False))
            except:
                state_dict = revise_ckpt_2(state_dict)
                print(self.model.load_state_dict(state_dict, strict=False))

        # run
        self.eval_epoch()
        while epoch < max_epoch:
            self.train_epoch(epoch)
            self.eval_epoch(epoch)
            epoch += 1
