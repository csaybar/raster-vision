import warnings
warnings.filterwarnings('ignore')  # noqa
from os.path import join, isdir, basename
from collections import defaultdict
import logging
import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')  # noqa
import torch
from torch.utils.data import Dataset, ConcatDataset
import torch.nn.functional as F
from torchvision import models
from PIL import Image

from rastervision2.pytorch_learner.learner import Learner
from rastervision2.pytorch_learner.utils import (compute_conf_mat_metrics,
                                                 compute_conf_mat)
from rastervision2.core.data.utils import color_to_triple
from rastervision.backend.torch_utils.object_detection.model import (
    MyFasterRCNN)

log = logging.getLogger(__name__)


def collate_fn(data):
    x = [d[0].unsqueeze(0) for d in data]
    y = [d[1] for d in data]
    return (torch.cat(x), y)


class CocoDataset(Dataset):
    def __init__(self, img_dir, annotation_uris, transforms=None):
        self.img_dir = img_dir
        self.annotation_uris = annotation_uris
        self.transforms = transforms

        self.imgs = []
        self.img2id = {}
        self.id2img = {}
        self.id2boxes = defaultdict(lambda: [])
        self.id2labels = defaultdict(lambda: [])
        self.label2name = {}
        for annotation_uri in annotation_uris:
            ann_json = file_to_json(annotation_uri)
            for img in ann_json['images']:
                self.imgs.append(img['file_name'])
                self.img2id[img['file_name']] = img['id']
                self.id2img[img['id']] = img['file_name']
            for ann in ann_json['annotations']:
                img_id = ann['image_id']
                box = ann['bbox']
                label = ann['category_id']
                box = torch.tensor(
                    [[box[1], box[0], box[1] + box[3], box[0] + box[2]]])
                self.id2boxes[img_id].append(box)
                self.id2labels[img_id].append(label)
        self.id2boxes = dict([(id, torch.cat(boxes).float())
                              for id, boxes in self.id2boxes.items()])
        self.id2labels = dict([(id, torch.tensor(labels))
                               for id, labels in self.id2labels.items()])

    def __getitem__(self, ind):
        img_fn = self.imgs[ind]
        img_id = self.img2id[img_fn]
        img = Image.open(join(self.img_dir, img_fn))

        if img_id in self.id2boxes:
            boxes, labels = self.id2boxes[img_id], self.id2labels[img_id]
            boxlist = BoxList(boxes, labels=labels)
        else:
            boxlist = BoxList(
                torch.empty((0, 4)), labels=torch.empty((0, )).long())
        if self.transforms:
            return self.transforms(img, boxlist)
        return (img, boxlist)

    def __len__(self):
        return len(self.imgs)


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

            # build datasets
            if isdir(train_dir):
                if cfg.overfit_mode:
                    train_ds.append(
                        ObjectDetectionDataset(
                            train_dir, transform=transform))
                else:
                    train_ds.append(
                        ObjectDetectionDataset(
                            train_dir, transform=aug_transform))

            if isdir(valid_dir):
                valid_ds.append(
                    ObjectDetectionDataset(
                        valid_dir, transform=transform))
                test_ds.append(
                    ObjectDetectionDataset(
                        valid_dir, transform=transform))

        train_ds, valid_ds, test_ds = \
            ConcatDataset(train_ds), ConcatDataset(valid_ds), ConcatDataset(test_ds)

        return train_ds, valid_ds, test_ds

    def train_step(self, batch, batch_nb):
        x, y = batch
        out = self.post_forward(self.model(x))
        return {'train_loss': F.cross_entropy(out, y)}

    def validate_step(self, batch, batch_nb):
        x, y = batch
        out = self.post_forward(self.model(x))
        val_loss = F.cross_entropy(out, y)

        num_labels = len(self.cfg.data.class_names)
        y = y.view(-1)
        out = self.prob_to_pred(out).view(-1)
        conf_mat = compute_conf_mat(out, y, num_labels)

        return {'val_loss': val_loss, 'conf_mat': conf_mat}

    def validate_end(self, outputs, num_samples):
        conf_mat = sum([o['conf_mat'] for o in outputs])
        val_loss = torch.stack([o['val_loss']
                                for o in outputs]).sum() / num_samples
        conf_mat_metrics = compute_conf_mat_metrics(conf_mat,
                                                    self.cfg.data.class_names)

        metrics = {'val_loss': val_loss.item()}
        metrics.update(conf_mat_metrics)

        return metrics

    def post_forward(self, x):
        return x['out']

    def prob_to_pred(self, x):
        return x.argmax(1)

    def plot_xyz(self, ax, x, y, z=None):
        x = x.permute(1, 2, 0)
        if x.shape[2] == 1:
            x = torch.cat([x for _ in range(3)], dim=2)
        ax.imshow(x)
        ax.axis('off')

        labels = z if z is not None else y
        colors = [color_to_triple(c) for c in self.cfg.data.class_colors]
        colors = [tuple([_c / 255 for _c in c]) for c in colors]
        cmap = matplotlib.colors.ListedColormap(colors)
        labels = labels.numpy()
        ax.imshow(labels, alpha=0.4, vmin=0, vmax=len(colors), cmap=cmap)
