from depen import *
from depen import tokenizer, vocab, token_indexer
import phonemizer
from phonemizer import phonemize
from phonemizer import separator
from dataset_download import download_dataset_and_extract
from datasets import create_mask_pad


def load_ljspeech_item(line: List[str], path: str, ext_audio: str):
    fileid, _, normalized_transcript, _, _ = line
    fileid_audio = fileid #+ ext_audio
    fileid_audio = os.path.join(path, fileid_audio)

    # Load audio
    waveform, sample_rate = torchaudio.load(fileid_audio)
    client_id = 1

    return waveform, sample_rate, client_id, normalized_transcript
    


class LJSPEECH(Dataset):
    """Create a Dataset for LJSpeech-1.1.

    Args:
        root (str): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from.
            (default: ``"https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"wavs"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    _ext_audio = ".wav"
    _ext_archive = '.tar.bz2'

    FOLDER_IN_ARCHIVE = "wavs"

    def __init__(self,
                 root: str,
                 folder_in_archive: str = FOLDER_IN_ARCHIVE,) -> None:

        self._path = os.path.join(root, folder_in_archive)
        self._metadata_path = os.path.join(root,'data.tsv')


        with open(self._metadata_path, "r") as metadata:
            walker = unicode_csv_reader(metadata, delimiter=",")
            self._header = next(walker)
            self._walker = list(walker)


    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, transcript, normalized_transcript)``
        """
        line = self._walker[n]
        return load_ljspeech_item(line, self._path, self._ext_audio)


    def __len__(self) -> int:
        return len(self._walker)


def collate_fn_lj(batch, data_type, max_len_mel = 1200, reconstructed_phoneme = True):
  #

  batch_size = len(batch)
  parts = 6
  final_list = []
  example_id = [0]

  for n in range(parts):
    final_list.append([None]*(batch_size))

  for i in range(batch_size):
    waveform, sample_rate, client_id, sentence = batch[i]
    part = [waveform, waveform.shape[1], sample_rate, client_id, sentence, len(sentence)]
    for j in range(parts):
        final_list[j][i] = part[j]


  waveforms, waveform_l, sample_rates, client_ids, sentences, sentences_l = final_list

  if not reconstructed_phoneme:
    sentences = phonemize(sentences, backend='espeak',with_stress=False, separator=separator.Separator(phone=' ', syllable='', word='- '))
  for i in range(len(sentences_l)):
    sentences_l[i] = len(sentences[i])
  biggest_l_index = sentences_l.index(max(sentences_l))
  
  token = tokenizer.tokenize(sentences[biggest_l_index])
  text_field = TextField(token, token_indexer)
  text_field.index(vocab)
  padding_lengths = text_field.get_padding_lengths()

  list_tokens = []
  mel_list = [None] * len(sample_rates)
  mel_list_l = [None] * len(sample_rates)
  #

  for i in range(len(sentences_l)):
    token = tokenizer.tokenize(sentences[i])
    text_field = TextField(token, token_indexer)
    text_field.index(vocab)
    tensor_dict = text_field.as_tensor(padding_lengths)
    list_tokens.append(tensor_dict)
    #
    if data_type == "train":
      mel_list[i] = train_audio_transforms(waveforms[i]).squeeze(0).transpose(0, 1)
    else:
      mel_list[i] = test_audio_transforms(waveforms[i]).squeeze(0).transpose(0, 1)
    mel_list_l[i] = mel_list[i].shape[0]
    waveforms[i] = waveforms[i].squeeze(0)
    #
  waveforms = nn.utils.rnn.pad_sequence(waveforms, batch_first=True).unsqueeze(1)
  #
  mel_list.append(torch.zeros((max_len_mel, mel_list[0].shape[1])))
  spectrograms = nn.utils.rnn.pad_sequence(mel_list, batch_first=True).unsqueeze(1).transpose(2, 3)
  spectrograms = spectrograms[1:]
  highest_mel_l = spectrograms[0].shape[2]
  mel_mask = create_mask_pad(highest_mel_l, mel_list_l)
  #
  text_field_tensors = text_field.batch_tensors(list_tokens)
  #
  sentences_tensor = nn_util.get_token_ids_from_text_field_tensors(text_field_tensors)
  sentences_mask = nn_util.get_text_field_mask(text_field_tensors) == False

  return sentences_tensor, sentences_mask, spectrograms, mel_mask, waveforms, waveform_l, client_ids, example_id


class Lj_pl_dataset(pl.LightningDataModule):
  def __init__(self, hparams):
    super().__init__()
    self.hparams = hparams
    self.path = self.hparams.path_dataset_common
    if not self.hparams.reconstructed_phoneme:
      print("!!!!!!!!!!!!!!!!!!!!!!!!!! prepare long time !!!!!!!!!!!!!!!!!!!!")


  def prepare_data(self):
    #here you download your dataset, from some site for example
    #or pytorch torchaudio etc.
    #or call torch.utils.data.Dataset type
    download_dataset_and_extract("lj")

  def setup(self): 
    dataset = LJSPEECH(self.path)
    len_t = int(len(dataset) * 0.8)
    len_v = int((len(dataset) - len_t)/2)
    len_test = len(dataset) - len_t - len_v
    self.dataset_train, self.dataset_val, self.dataset_test = torch.utils.data.random_split(dataset, [len_t, len_v, len_test], generator=torch.Generator().manual_seed(42))
    
  def train_dataloader(self):
    train_loader = DataLoader(dataset=self.dataset_train,
                                batch_size=self.hparams.batch_size,
                                shuffle=True,
                                collate_fn=lambda x: collate_fn_lj(x, 'train', self.hparams.mel_limit, reconstructed_phoneme = self.hparams.reconstructed_phoneme),
                                num_workers = self.hparams.num_workers)
    return train_loader

  def val_dataloader(self):
    val_loader = DataLoader(dataset=self.dataset_val,
                                batch_size=self.hparams.batch_size,
                                shuffle=False,
                                collate_fn=lambda x: collate_fn_lj(x, 'valid', self.hparams.mel_limit, reconstructed_phoneme = self.hparams.reconstructed_phoneme),
                                num_workers = self.hparams.num_workers)
    return val_loader

  def test_dataloader(self):
    test_loader = DataLoader(dataset=self.dataset_test,
                                batch_size=self.hparams.batch_size,
                                shuffle=False,
                                collate_fn=lambda x: collate_fn_lj(x, 'valid', self.hparams.mel_limit, reconstructed_phoneme = self.hparams.reconstructed_phoneme),
                                num_workers = self.hparams.num_workers)
    return test_loader