from hparams import hyperparams
print(hyperparams.path_dataset_common)

from depen import *
from model import Multi_Synth_pl
from datasets import Common_pl_dataset

from pytorch_model_summary import summary
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

