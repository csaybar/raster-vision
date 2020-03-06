import warnings
warnings.filterwarnings('ignore')  # noqa
from os.path import join, isdir, basename
from collections import defaultdict
import logging
import glob

import numpy as np
import matplotlib
import matplotlib.patches as patches
matplotlib.use('Agg')  # noqa
import torch
from torch.utils.data import ConcatDataset
from torchvision import models
from PIL import Image

from rastervision2.pytorch_learner.learner import Learner
from rastervision2.pytorch_learner.utils import (compute_conf_mat_metrics,
                                                 compute_conf_mat)
from rastervision2.core.data.utils import color_to_triple
from rastervision2.pipeline.filesystem import file_to_json
from rastervision2.pytorch_learner.object_detection_utils import (
    MyFasterRCNN, CocoDataset, compute_class_f1, compute_coco_eval)

log = logging.getLogger(__name__)


class ObjectDetectionLearner(Learner):
    def build_model(self):
        # TODO we shouldn't need to pass the image size here
        model = MyFasterRCNN(
            self.cfg.model.backbone, len(self.cfg.data.class_names),
            self.cfg.data.img_sz, pretrained=True)
        return model

    def get_datasets(self):
        cfg = self.cfg

        if cfg.data.data_format == 'default':
            data_dirs = self.unzip_data()

        transform, aug_transform = self.get_data_transforms()

        train_ds, valid_ds, test_ds = [], [], []
        for data_dir in data_dirs:
            train_dir = join(data_dir, 'train')
            valid_dir = join(data_dir, 'valid')

            if isdir(train_dir):
                img_dir = join(train_dir, 'img')
                annotation_uris = [join(train_dir, 'labels.json')]
                if cfg.overfit_mode:
                    train_ds.append(CocoDataset(img_dir, annotation_uris, transform=transform))
                else:
                    train_ds.append(CocoDataset(img_dir, annotation_uris, transform=aug_transform))

            if isdir(valid_dir):
                img_dir = join(valid_dir, 'img')
                annotation_uris = [join(valid_dir, 'labels.json')]
                valid_ds.append(CocoDataset(img_dir, annotation_uris, transform=transform))
                test_ds.append(CocoDataset(img_dir, annotation_uris, transform=transform))

        train_ds, valid_ds, test_ds = \
            ConcatDataset(train_ds), ConcatDataset(valid_ds), ConcatDataset(test_ds)

        return train_ds, valid_ds, test_ds

    def train_step(self, batch, batch_nb):
        x, y = batch
        loss_dict = self.model(x, y)
        return {'train_loss': loss_dict['total_loss']}

    def validate_step(self, batch, batch_nb):
        x, y = batch
        out = self.model(x)
        ys = [_y.cpu() for _y in y]
        outs = [_out.cpu() for _out in out]
        
        return {'ys': ys, 'outs': outs}

    def validate_end(self, outputs, num_samples):
        outs = torch.cat([o['outs'] for o in outputs])
        ys = torch.cat([o['ys'] for o in outputs])
        num_labels = len(self.cfg.data.class_names)
        coco_eval = compute_coco_eval(outs, ys, num_labels)

        metrics = {
            'map': 0.0,
            'map50': 0.0,
            'mean_f1': 0.0,
            'mean_score_thresh': 0.5
        }
        if coco_eval is not None:
            coco_metrics = coco_eval.stats
            best_f1s, best_scores = compute_class_f1(coco_eval)
            mean_f1 = np.mean(best_f1s[1:])
            mean_score_thresh = np.mean(best_scores[1:])
            metrics = {
                'map': coco_metrics[0],
                'map50': coco_metrics[1],
                'mean_f1': mean_f1,
                'mean_score_thresh': mean_score_thresh
            }
        return metrics

    def post_forward(self, x):
        # TODO
        return x['out']

    def prob_to_pred(self, x):
        # TODO
        return x.argmax(1)

    def plot_xyz(self, ax, x, y, z=None):
        ax.imshow(x.permute(1, 2, 0))
        y = y if z is None else z

        scores = y.get_field('scores')
        for box_ind, (box, class_id) in enumerate(
                zip(y.boxes, y.get_field('labels'))):
            rect = patches.Rectangle(
                (box[1], box[0]),
                box[3] - box[1],
                box[2] - box[0],
                linewidth=1,
                edgecolor='cyan',
                facecolor='none')
            ax.add_patch(rect)

            class_name = self.cfg.data.class_names[class_id]
            if scores is not None:
                score = scores[box_ind]
                label_name += ' {:.2f}'.format(score)

            h, w = x.shape[1:]
            label_height = h * 0.03
            label_width = w * 0.15
            rect = patches.Rectangle(
                (box[1], box[0] - label_height),
                label_width,
                label_height,
                color='cyan')
            ax.add_patch(rect)

            ax.text(
                box[1] + w * 0.003, box[0] - h * 0.003, class_name, fontsize=7)

        ax.axis('off')
