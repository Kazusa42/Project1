import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import (YOLOLoss, get_lr_scheduler, set_optimizer_lr)
from utils.callbacks import LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_anchors, get_classes, weights_init, show_config
from utils.utils_fit import fit_one_epoch

from configure import *


if __name__ == "__main__":
    Cuda = IF_CUDA
    classes_path = CLASSES_PATH
    anchors_path = ANCHOR_PATH
    anchors_mask = ANCHOR_MASK
    model_path = MODEL_PATH
    input_shape = INPUT_SHAPE
    backbone = BACKBONE

    pretrained = PRE_TRAINED
    mosaic = MOSAIC
    label_smoothing = 0

    Init_Epoch = INIT_EPOCH
    Freeze_Epoch = FREEZE_EPOCH
    Freeze_batch_size = FREEZE_BATCH_SIZE
    UnFreeze_Epoch = UNFREZZEZ_EPOCH
    Unfreeze_batch_size = UNFREEZE_BATCH_SIZE
    Freeze_Train = FREEZE_TRAIN

    Init_lr = INIT_LR
    Min_lr = MIN_LR

    optimizer_type = OPT_TYPE
    momentum = MOMENTUM
    weight_decay = WEIGHT_DECAY

    lr_decay_type = LR_DECAY_TYPE

    save_period = 1
    num_workers = 4  # thread number

    train_annotation_path = 'train.txt'
    val_annotation_path = 'val.txt'

    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)

    model = YoloBody(anchors_mask, num_classes, backbone=backbone, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask, label_smoothing)
    loss_history = LossHistory("logs/", model, input_shape=input_shape)
    
    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    show_config(classes_path=classes_path, model_path=model_path, input_shape=input_shape, Init_Epoch=Init_Epoch,
                Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch, Freeze_batch_size=Freeze_batch_size,
                Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train,
                Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
                lr_decay_type=lr_decay_type, num_workers=num_workers, num_train=num_train, num_val=num_val,
                pretrained=PRE_TRAINED, backbone=BACKBONE, attentionneck=IF_ATTENTIONNECK, anchor_shape=ANCHOR_PATH)

    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        nbs = 64
        Init_lr_fit = max(batch_size / nbs * Init_lr, 1e-4)
        Min_lr_fit = max(batch_size / nbs * Min_lr, 1e-6)

        pg0, pg1, pg2 = [], [], []  
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)    
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)    
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)   
        optimizer = {
            'adam': optim.Adam(pg0, Init_lr_fit, betas=(momentum, 0.999)),
            'sgd': optim.SGD(pg0, Init_lr_fit, momentum=momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Dataset is too small to train.")

        train_dataset = YoloDataset(train_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch,
                                    mosaic=mosaic, train=True)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch,
                                  mosaic=False, train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)

        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs = 64
                Init_lr_fit = max(batch_size / nbs * Init_lr, 1e-4)
                Min_lr_fit = max(batch_size / nbs * Min_lr, 1e-6)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("Dataset is too small to train.")

                gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
                gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)

                UnFreeze_flag = True

            gen.dataset.epoch_now = epoch
            gen_val.dataset.epoch_now = epoch
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step,
                          epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, save_period)
