from torch.nn import *

from torch_loss_functions import KLDivWithLogitsLoss, WeightedKLDivWithLogitsLoss, LabelSmoothingBCEWithLogitsLoss
from eegnet import EEGNet
from wavenet import WaveNet
from eeg_2d_models import EfficientNet
