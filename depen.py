#can be deleted, just made for myself to make evrything clean
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import torch
from torch import nn
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import random
import torch.nn.functional as F
import os
import pandas as pd
from typing import List, Dict, Tuple

from torchaudio.datasets.utils import download_url, extract_archive, unicode_csv_reader
from torch import Tensor

from hparams import hyperparams #!

from allennlp.data.tokenizers import SpacyTokenizer

from allennlp.data import Token, Vocabulary
from allennlp.data.fields import ListField, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer

from allennlp.nn import util as nn_util

import math, copy, time
from torch.autograd import Variable


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#seeding can be used when training
def seed_e(seed_value):
  pl.seed_everything(seed_value)
  random.seed(seed_value)
  np.random.seed(seed_value) 
  torch.manual_seed(seed_value)
  torch.cuda.manual_seed(seed_value)
  torch.cuda.manual_seed_all(seed_value)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

train_audio_transforms = nn.Sequential(
    #torchaudio.transforms.Resample(hyperparams.orig_sample_rate, hyperparams.sampling_rate),
    torchaudio.transforms.MelSpectrogram(sample_rate=hyperparams.sampling_rate, n_mels=hyperparams.n_mels,),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=hyperparams.freq_mask_param),
    torchaudio.transforms.TimeMasking(time_mask_param=hyperparams.time_mask_param)
)

test_audio_transforms = nn.Sequential(
    #torchaudio.transforms.Resample(hyperparams.orig_sample_rate, hyperparams.sampling_rate),
    torchaudio.transforms.MelSpectrogram(sample_rate=hyperparams.sampling_rate, n_mels=hyperparams.n_mels),
    #torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    #torchaudio.transforms.TimeMasking(time_mask_param=35)
)



alphabet = ['p', 'ɹ', 'ɪ', 'n', 't', 'ŋ', 'ð', 'oʊ', 'l', 'i', 's', 'ɛ', 'w', 'tʃ', 'iː', 
              'ɑː', 'æ', 'z', 'ə', 'k', 'ɜː', 'd', 'f', 'ɚ', 'ʌ', 'm', 'ɔː', 'ɑːɹ', 'ᵻ', 'b', 'ʃ', 'v', 'aɪ', 
              'ʊ', 'ɡ', 'eɪ', 'ɔːɹ', 'oːɹ', 'ɾ', 'ɐ', 'uː', 'əl', 'θ', 'dʒ', 'j', 'aʊ', 'h', 'ɔ', 'ɛɹ', 'ʔ', 'n̩', 
              'ɪɹ', 'ʊɹ', 'aɪɚ', 'ʒ', 'oː', 'iə', 'r', 'ɔɪ', 'aɪə', 'õ', '(gn)', 'a', 'ɣ', 'x',] + list("-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}") + [" ", "  "]
#
tokenizer = SpacyTokenizer(pos_tags=True)
vocab = Vocabulary()
vocab.add_tokens_to_namespace(
  alphabet,
  namespace='token_vocab')
token_indexer = {
    'tokens': SingleIdTokenIndexer(namespace='token_vocab'),
}