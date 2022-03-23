"""
Global variable configure
Aiming to simplify the trainging, predicting and evaluating process between different dataset
"""

# frcnn basic configure
CLASSES_PATH = r'model_data/DOTA_classes.txt'
MODEL_PATH = r'model_data/ep266-loss0.226-val_loss0.296.pth'
# ANCHOR_SIZE = [4, 16, 32]  # [4, 16, 32] for tiny object, [8, 16, 32] for normal type
ANCHOR_PATH = r'model_data/yolo_anchors.txt'
IF_CUDA = r'True'
FONT_TYPE = r'model_data/monoMMM_5.ttf'
#  get_map
pass

#  Dataset
DATASET_PATH = r'DOTA'
