# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 21:05
@Author        : Tianxiaomo
@File          : Cfg.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
from easydict import EasyDict

Cfg = EasyDict()

Cfg.use_darknet_cfg = False
Cfg.cfgfile = 'cfg/yolov4.cfg'

Cfg.batch = 1
Cfg.subdivisions = 1
Cfg.width = 608
Cfg.height = 608
Cfg.channels = 3
Cfg.momentum = 0.949
Cfg.decay = 0.0005
Cfg.angle = 0
Cfg.saturation = 1.5
Cfg.exposure = 1.5
Cfg.hue = .1

Cfg.learning_rate = 0.00261
Cfg.burn_in = 200
Cfg.max_batches = 500500
Cfg.steps = [2000, 8000]
Cfg.policy = Cfg.steps
Cfg.scales = .1, .1

Cfg.cutmix = 0
Cfg.mosaic = 1

Cfg.letter_box = 0
Cfg.jitter = 0.2
Cfg.classes = 18
Cfg.track = 0
Cfg.w = Cfg.width
Cfg.h = Cfg.height
Cfg.flip = 1
Cfg.blur = 0
Cfg.gaussian = 0
Cfg.boxes = 60  # box num
Cfg.TRAIN_EPOCHS = 20
Cfg.train_label = r'D:\Data\chem-yolov4\dataset\all\front_data.txt'
Cfg.val_label = r'D:\Data\CMS01_single-end\front_data.txt'
Cfg.TRAIN_OPTIMIZER = 'adam'
'''
image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
...
'''

if Cfg.mosaic and Cfg.cutmix:
    Cfg.mixup = 4
elif Cfg.cutmix:
    Cfg.mixup = 2
elif Cfg.mosaic:
    Cfg.mixup = 3
else:
    Cfg.mixup = 0

Cfg.checkpoints = 'checkpoints'
Cfg.TRAIN_TENSORBOARD_DIR = 'log'

Cfg.iou_type = 'iou'  # 'giou', 'diou', 'ciou'
Cfg.keep_checkpoint_max = 10