"""
Global variable configure
Aiming to simplify the trainging, predicting and evaluating process between different dataset and weights
"""

"""  Basic configure for model """
CLASSES_PATH = r'model_data/DOTA_classes.txt'
DATASET_PATH = r'DOTA'
MODEL_PATH = r'model_data/ep266-loss0.226-val_loss0.296.pth'
BACKBONE = r'resnet50'  # resnet50, mobilenetv1, mobilenetv2, mobilenetv3, densenet121, densenet169, densenet201
IF_CUDA = r'True'
ANCHOR_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
ANCHOR_PATH = r'model_data/yolo_anchors.txt'
INPUT_SHAPE = [416, 416]
CONFIDENCE = 0.5
NMS_SCORE = 0.3


""" Training setting """
PRE_TRAINED = False  # If MODEL_PATH is not None, the value will not work.
MOSAIC = True

INIT_EPOCH = 0
FREEZE_EPOCH, UNFREZZEZ_EPOCH = 50, 100
FREEZE_BATCH_SIZE, UNFREEZE_BATCH_SIZE = 16, 8
FREEZE_TRAIN = True

OPT_TYPE = r'sgd'  # sgd or adam
INIT_LR = 0.01  # If you use adam, set this value to 0.001
MIN_LR = INIT_LR * 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 5e-4
LR_DECAY_TYPE = r'cos'


""" Others """
FONT_TYPE = r'model_data/monoMMM_5.ttf'

