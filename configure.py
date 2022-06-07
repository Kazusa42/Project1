"""
Global variable configure
Aiming to simplify the trainging, predicting and evaluating process between different dataset and weights
"""

BACKBONE_LIST = [r'resneet50',  # 0
                 r'convnext_tiny', r'convnext_small'  # 1, 2
                 r'CSPDarknet53']  # 3

"""  Basic configure for model """
CLASSES_PATH = r'model_data/DOTA_classes.txt'
DATASET_PATH = r'DOTA'
MODEL_PATH = r''
BACKBONE = BACKBONE_LIST[1]
IF_CUDA = r'True'
ANCHOR_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
ANCHOR_PATH = r'model_data/kmeans_anchors.txt'
CONFIDENCE = 0.5
NMS_SCORE = 0.3
INPUT_SHAPE = [640, 640]
RESOLUTION = [INPUT_SHAPE[0] // 32, INPUT_SHAPE[1] // 32]

""" Training setting """
PRE_TRAINED = False  # If MODEL_PATH is not None, the value will not work.
MOSAIC = True

INIT_EPOCH = 0
FREEZE_EPOCH, UNFREZZEZ_EPOCH = 0, 500
FREEZE_BATCH_SIZE, UNFREEZE_BATCH_SIZE = 16, 6
FREEZE_TRAIN = False

OPT_TYPE = r'adam'  # sgd or adam
INIT_LR = 0.001  # If you use adam, set this value to 0.001
MIN_LR = INIT_LR * 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 5e-4
LR_DECAY_TYPE = r'cos'


""" Others """
FONT_TYPE = r'model_data/font_style_1.ttf'

