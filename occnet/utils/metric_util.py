import torch
import torch.distributed as dist
import numpy as np
from mmseg.utils import get_root_logger


class MeanIoU:

    def __init__(self,
                 class_indices,
                 ignore_label: int,
                 label_str
                 # empty_class: int
                 ):
        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        self.ignore_label = ignore_label
        self.label_str = label_str

    def reset(self) -> None:
        self.total_seen = torch.zeros(self.num_classes).cuda()
        self.total_correct = torch.zeros(self.num_classes).cuda()
        self.total_positive = torch.zeros(self.num_classes).cuda()

    def after_step(self, outputs, targets):
        outputs = outputs[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]

        for i, c in enumerate(self.class_indices):
            self.total_seen[i] += torch.sum(torch.isclose(targets, c)).item()
            self.total_correct[i] += torch.sum(torch.isclose(targets, c) &
                                               torch.isclose(outputs, c)).item()
            self.total_positive[i] += torch.sum(torch.isclose(outputs, c)).item()

    def after_epoch(self):
        dist.all_reduce(self.total_seen)
        dist.all_reduce(self.total_correct)
        dist.all_reduce(self.total_positive)

        ious = []

        for i in range(self.num_classes):
            if self.total_seen[i] == 0:
                ious.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i] +
                                                   self.total_positive[i] -
                                                   self.total_correct[i])
                ious.append(cur_iou.item())

        miou = np.mean(ious)
        logger = get_root_logger()
        logger.info(f'Validation per class iou:')
        for iou, label_str in zip(ious, self.label_str):
            logger.info('%s : %.2f%%' % (label_str, iou * 100))

        return miou * 100
