import torch
from sunn_models.config.config import CfgNode as CN


_C = CN()

###################################################################################
_C.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
_C.MODEL.WEIGHTS = None
_C.MODEL.RESUME = False
_C.MODEL.MODEL_DIR = ''
_C.MODEL.RSU_DICT = {
    'RSU-E7': (3, 32, 64),
    'RSU-E6': (64, 32, 128),
    'RSU-E5': (128, 64, 256),
    'RSU-E4': (256, 128, 512),
    'RSU-4F-1': (512, 256, 512),
    'RSU-4F-2': (512, 256, 512),
    'RSU-4F-3': (1024, 256, 512),
    'RSU-D4': (1024, 128, 256),
    'RSU-D5': (512, 64, 128),
    'RSU-D6': (256, 32, 64),
    'RSU-D7': (128, 16, 64)
}

###################################################################################
_C.DATA.IMAGE_DIR = '/mnt/dms/makeface/DUTS-TE/DUTS-TE-Image'
_C.DATA.MASK_DIR = '/mnt/dms/makeface/DUTS-TE/DUTS-TE-Mask'
_C.DATA.OUTPUT_DIR = '/mnt/dms/makeface'
_C.DATA.TRAIN_PERCENT = 0.8
_C.DATA.SHUFFLE = True
_C.DATA.RESIZE = 320
_C.DATA.AUGMENTATION = True
_C.DATA.DROP_LAST = True

###################################################################################
_C.OPTIMIZER.INITIAL_LEARNING_RATE = 1e-3
_C.OPTIMIZER.BETAS = (0.9, 0.999)
_C.OPTIMIZER.EPS = 1e-8
_C.OPTIMIZER.WEIGHT_DECAY = 0

###################################################################################
_C.TRAIN.EPOCHS = 100
_C.TRAIN.BATCH_SIZE = 32

###################################################################################
_C.LOSS.FUNCTION = 'BCE'
