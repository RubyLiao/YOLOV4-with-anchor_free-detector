# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 15:07
@Author        : Tianxiaomo
@File          : train.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter
import logging
import os, sys, math
from tqdm import tqdm
from dataset import Yolo_dataset
from cfg import Cfg
from models import Yolov4
import argparse
from easydict import EasyDict as edict
from torch.nn import functional as F
from tool.darknet2pytorch import Darknet
from config_utils import CONFIG
from tool.utils import truncate_radius, draw_truncate_gaussian
from tool.vis_utils import vis



import numpy as np


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True, GIoU=False, DIoU=False, CIoU=False):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    https://github.com/ultralytics/yolov3/blob/eca5b9c1d36e4f73bf2f94e141d864f1c2739e23/utils/utils.py#L262-L282
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        # intersection top left
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # intersection bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[:, :2])
        con_br = torch.max(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, 0] + bboxes_a[:, None, 2]) - (bboxes_b[:, 0] + bboxes_b[:, 2])) ** 2 / 4 + (
                (bboxes_a[:, None, 1] + bboxes_a[:, None, 3]) - (bboxes_b[:, 1] + bboxes_b[:, 3])) ** 2 / 4

        w1 = bboxes_a[:, 2] - bboxes_a[:, 0]
        h1 = bboxes_a[:, 3] - bboxes_a[:, 1]
        w2 = bboxes_b[:, 2] - bboxes_b[:, 0]
        h2 = bboxes_b[:, 3] - bboxes_b[:, 1]

        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # intersection top left
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # intersection bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        con_br = torch.max((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, :2] - bboxes_b[:, :2]) ** 2 / 4).sum(dim=-1)

        w1 = bboxes_a[:, 2]
        h1 = bboxes_a[:, 3]
        w2 = bboxes_b[:, 2]
        h2 = bboxes_b[:, 3]

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    area_u = area_a[:, None] + area_b - area_i
    iou = area_i / area_u

    if GIoU or DIoU or CIoU:
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            area_c = torch.prod(con_br - con_tl, 2)  # convex area
            return iou - (area_c - area_u) / area_c  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w1 / h1).unsqueeze(1) - torch.atan(w2 / h2), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    return iou


class Yolo_loss(nn.Module):
    def __init__(self, n_classes=80, n_anchors=3, device=None, batch=2):
        super(Yolo_loss, self).__init__()
        self.device = device
        self.strides = [8, 16, 32]
        image_size = 608
        self.n_classes = n_classes
        self.n_anchors = n_anchors
        self.all_n_anchors = n_anchors

        self.anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.ignore_thre = 0.5

        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []

        for i in range(self.n_anchors):
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]
            masked_anchors = np.array([all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32)
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)
            # calculate pred - xywh obj cls
            fsize = image_size // self.strides[i]
            grid_x = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).to(device)
            grid_y = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).permute(0, 1, 3, 2).to(device)
            anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)
            anchor_h = torch.from_numpy(masked_anchors[:, 1]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def build_center_target(self, label_box, batch_ind, fsize, output_id, tgt_mask, tgt_scale, target):


        b = batch_ind
        #这个地方应该改一下，先确认哪些gt应该标注在本层

        x1, y1, x2, y2, cls = label_box[0], label_box[1], label_box[2], label_box[3], int(label_box[4])
        w_source, h_source = x2 - x1, y2 - y1
        x_center, y_center = (x1+x2)/(self.strides[output_id]*2), (y1+y2)/(self.strides[output_id]*2)
        x_center_int, y_center_int = int(x_center), int(y_center)

        target[b, 0, y_center_int, x_center_int, 0] = float(x_center - x_center_int)
        target[b, 0, y_center_int, x_center_int, 1] = float(y_center - y_center_int)
        target[b, 0, y_center_int, x_center_int, 2] = float(torch.log(w_source/self.strides[output_id]/4 + 1e-16))
        target[b, 0, y_center_int, x_center_int, 3] = float(torch.log(h_source/self.strides[output_id]/4 + 1e-16))

        h_radius, w_radius = truncate_radius((math.ceil(h_source), math.ceil(w_source)),
                                             stride=self.strides[output_id])  # 根据box计算radius
        h_radius, w_radius = max(1, int(h_radius)), max(1, int(w_radius))                       #obj_mask不标也没关系。target小圈外的都是0

        draw_truncate_gaussian(target[b, 0, :, :, 4], (x_center_int, y_center_int), h_radius, w_radius, k=1)  # 将中心点和radius标记到confidence中
        draw_truncate_gaussian(target[b, 0, :, :, 5+cls], (x_center_int, y_center_int), h_radius, w_radius,
                               k=1)  # 将中心点和radius标记到confidence中


        tgt_mask[b, 0, y_center_int, x_center_int, :] = 1.

        tgt_scale[b, 0, y_center_int, x_center_int, :] = torch.sqrt(2 - (w_source/self.strides[output_id] * h_source/self.strides[output_id]) / fsize / fsize)


    def build_target(self, pred, labels, batchsize, fsize, n_ch, output_id):
        # target assignment
        tgt_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes).to(device=self.device)
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).to(device=self.device)
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).to(self.device)
        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).to(self.device)       #n_ch=5+n_classes

        # labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (self.strides[output_id] * 2)    #x_center
        truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (self.strides[output_id] * 2)    #y_center
        truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / self.strides[output_id]
        truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / self.strides[output_id]
        truth_i_all = truth_x_all.to(torch.int16).cpu().numpy()
        truth_j_all = truth_y_all.to(torch.int16).cpu().numpy()

        tgt_mask_free = torch.zeros(batchsize, 1, fsize, fsize, 4 + self.n_classes).to(device=self.device)  # tgt_mask才是xywh, 和cls的mask
        obj_mask_free = torch.ones(batchsize, 1, fsize, fsize).to(device=self.device)  # 那么obj_mask这样标,只应用于confidence，与作者同
        tgt_scale_free = torch.zeros(batchsize, 1, fsize, fsize, 2).to(self.device)
        target_free = torch.zeros(batchsize, 1, fsize, fsize, n_ch).to(self.device).cpu().numpy()  # n_ch=5+n_classes

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = torch.zeros(n, 4).to(self.device)
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors[output_id], CIoU=True)

            # temp = bbox_iou(truth_box.cpu(), self.ref_anchors[output_id])

            best_n_all = anchor_ious_all.argmax(dim=1)
            best_n = best_n_all % 3
            best_n_mask = ((best_n_all == self.anch_masks[output_id][0]) |
                           (best_n_all == self.anch_masks[output_id][1]) |
                           (best_n_all == self.anch_masks[output_id][2]))

            if sum(best_n_mask) == 0:
                continue

            truth_box[:n, 0] = truth_x_all[b, :n]      #x_center赋给truth_box
            truth_box[:n, 1] = truth_y_all[b, :n]

            pred_ious = bboxes_iou(pred[b].view(-1, 4), truth_box, xyxy=False)    #pred和所有的truth box的交集
            pred_best_iou, _ = pred_ious.max(dim=1)                               #交集最大的
            pred_best_iou = (pred_best_iou > self.ignore_thre)                    #交集大于阈值的
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth                      #不管是多大的pred和哪一层的truth_box相交，只要是交集是与truth_box iou最大且分数高，这个anchor就记为True
            obj_mask[b] = ~ pred_best_iou                                          #取反。也就是完全没有iou的，记为1.iou符合要求的，记为0

            for ti in range(best_n.shape[0]):  #ti是box在label中的序号
                if best_n_mask[ti] == 1:       #如果box属于本层
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1                         #符合本层要求的，也就是铁定，唯一在本层要被标记为正样本的，记为1. 这一个位置不为pred的结果所动。哪怕这个点的pred飞出天际，也记为1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).cpu().numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)

                    label_box = labels[b, ti]
                    self.build_center_target(label_box, b, fsize, output_id, tgt_mask_free, tgt_scale_free, target_free)

        # tgt_mask = torch.cat([tgt_mask, tgt_mask_free], dim=1)
        # obj_mask = torch.cat([obj_mask, obj_mask_free], dim=1)
        # tgt_scale = torch.cat([tgt_scale, tgt_scale_free], dim=1)
        # target = torch.cat([target, torch.tensor(target_free).cuda()], dim=1)
        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, xin, labels=None, images=None):
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0
        for output_id, output in enumerate(xin):
            batchsize = output.shape[0]
            fsize = output.shape[2]
            n_ch = 5 + self.n_classes

            output = output.view(batchsize, self.all_n_anchors, n_ch, fsize, fsize)
            output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

            # logistic activation for xy, obj, cls
            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])

            pred = output[:, :3, :, :, :4].clone()

            pred[..., 0] += self.grid_x[output_id]
            pred[..., 1] += self.grid_y[output_id]
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[output_id]
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[output_id]

            obj_mask, tgt_mask, tgt_scale, target = self.build_target(pred, labels, batchsize, fsize, n_ch, output_id)

            conf_pred = output[..., 4]
            conf_gt = target[..., 4]
            vis(conf_pred, conf_gt, images)

            # loss calculation
            output[..., 4] *= obj_mask                     #confidence先抛弃了，既不是命定的正样本anchor, 又不是完全没交集的。也就是抛弃了中间结果
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask    #xywh和cls通道仅保留了命定的正样本anchor
            output[..., 2:4] *= tgt_scale                  #这是一个神奇的scale。feature map上的box的2- w*h/f_size*f_size 再开方     w, h乘上它有什么意义？

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale                   #它也乘上了。

            loss_xy += F.binary_cross_entropy(input=output[..., :2], target=target[..., :2],            #然后开始计算了
                                              weight=tgt_scale * tgt_scale, size_average=False)
            loss_wh += F.mse_loss(input=output[..., 2:4], target=target[..., 2:4], size_average=False) / 2
            loss_obj += F.binary_cross_entropy(input=output[..., 4], target=target[..., 4], size_average=False)
            loss_cls += F.binary_cross_entropy(input=output[..., 5:], target=target[..., 5:], size_average=False)
            loss_l2 += F.mse_loss(input=output, target=target, size_average=False)

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2


def collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append([img])
        bboxes.append([box])
    images = np.concatenate(images, axis=0)
    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images).div(255.0)
    bboxes = np.concatenate(bboxes, axis=0)
    bboxes = torch.from_numpy(bboxes)
    return images, bboxes


def train(model, device, config, epochs=5, log_step=20, img_scale=0.5):
    train_dataset = Yolo_dataset(config.train_label, config)

    n_train = len(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=config.batch, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True, collate_fn=collate)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {config.batch}
        Subdivisions:    {config.subdivisions}
        Learning rate:   {config.learning_rate}
        Training size:   {n_train}
        Device:          {device.type}
        Images size:     {config.width}
        Optimizer:       {config.TRAIN_OPTIMIZER}
        Dataset classes: {config.classes}
        Train label path:{config.train_label}
        Pretrained:
    ''')

    # learning rate setup
    def adjust_learning_rate(optimizer, iter, lr):
        """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
        if iter < burn_in:
            factor = pow(iter / burn_in, 4)
        elif iter < steps[0]:
            factor = 1.0
        elif iter < steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        lr = lr*factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr



    lr_start = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr_start, betas=(0.9, 0.999), eps=1e-08)
    # optimizer = optim.SGD(model.parameters(), lr=lr_start, momentum=0.9, dampening=0)

    criterion = Yolo_loss(device=device, batch=config.batch // config.subdivisions, n_classes=config.classes)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', verbose=True, patience=6, min_lr=1e-7)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, 0.001, 1e-6, 20)

    model.train()
    start_epoch = 0
    burn_in = 10
    epochs = 6
    batch_num = len(train_loader) * epochs
    steps = [int(0.5*batch_num), int(0.8*batch_num)]
    global_iter = start_epoch * len(train_loader)
    for epoch in range(start_epoch, epochs):
        torch.save(model.state_dict(), os.path.join(config.checkpoints, f'Yolov4_epoch{epoch + 1}.pth'))

        # with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', ncols=50) as pbar:
        for i, batch in enumerate(train_loader):
            global_iter += 1
            images = batch[0]
            bboxes = batch[1]

            images = images.to(device=device, dtype=torch.float32)
            bboxes = bboxes.to(device=device)

            bboxes_pred = model(images)
            loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(bboxes_pred, bboxes, images)
            # loss = loss / config.subdivisions

            loss.backward()
            optimizer.step()
            model.zero_grad()

            adjust_learning_rate(optimizer, global_iter, lr_start)
            lr = optimizer.param_groups[0]['lr']

            if global_iter % (log_step) == 0:
                part = i/len(train_loader)
                logging.info('Epoch:%.2f/%d  Loss:%.4f. loss_xy:%.4f  loss_wh:%.4f  loss_obj:%.4f  loss_cls:%.4f  loss_l2:%.4f  lr:%.6f'%
                             (epoch+part, epochs, loss.item(), loss_xy.item(), loss_wh.item(), loss_obj.item(), loss_cls.item(), loss_l2.item(),
                              lr))

        try:
            os.mkdir(config.checkpoints)
            logging.info('Created checkpoint directory')
        except OSError:
            pass
        torch.save(model.state_dict(), os.path.join(config.checkpoints, f'Yolov4_epoch{epoch + 1}.pth'))
        logging.info(f'Checkpoint {epoch + 1} saved !')



def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
    #                     help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='learning_rate')
    parser.add_argument('-f', '--load', dest='load', type=str, default=r'D:\DeepBlueProjects\chem-lab\pytorch-YOLOv4-master\checkpoints\Yolov4_epoch6.pth',
                        help='Load model from a .pth file')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='0',
                        help='GPU', dest='gpu')
    parser.add_argument('-dir', '--data-dir', type=str, default=r'D:\Data\chem-yolov4\dataset\all\JPEGImages',
                        help='dataset dir', dest='dataset_dir')
    parser.add_argument('-weight', type=str, default=r'D:\DeepBlueProjects\chem-lab\pytorch-YOLOv4-master\weights\yolov4.conv.137.pth', help='pretrained yolov4.conv.137')
    parser.add_argument('-classes', type=int, default=len(CONFIG.cls_id4.keys()), help='dataset classes')
    parser.add_argument('-train_label_path', dest='train_label', type=str, default=r'D:\Data\chem-yolov4\dataset\all\top_data.txt',      #r'D:\Data\tianping-ljr-laboratory\v3\tiny_data.txt',
                        help="train label path")
    args = vars(parser.parse_args())

    for k in args.keys():
        cfg[k] = args.get(k)
    return edict(cfg)


def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    import datetime
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging


if __name__ == "__main__":
    logging = init_logger(log_dir='log')
    cfg = get_args(**Cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if cfg.use_darknet_cfg:
        model = Darknet(cfg.cfgfile)
    else:
        model = Yolov4(cfg.weight, n_classes=cfg.classes)
        # pretrained_dict = torch.load(cfg.load)
        # model_dict = model.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if
        #                    k in model_dict}  # pretrained_dict只保留了model_dict中存在的键。为什么直接load它会报错。要先给model_dict更新。
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=device)

    try:
        train(model=model,
              config=cfg,
              epochs=cfg.TRAIN_EPOCHS,
              device=device, )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
