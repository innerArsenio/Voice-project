from depen import *
from hparams import hyperparams
from datasets import COMMONVOICE, collate_fn_common
from modules import Model_Check
import shutil


"""

path = hyperparams.path_dataset_common
TSV = "my_train.tsv"
folder = "clips"
new_folder = "new_clips"

tsvm = os.path.join(path, TSV)
folder = os.path.join(path, folder)
new_folder = os.path.join(path, new_folder)

df = pd.read_csv(tsvm, sep="\t")

s = df["path"].values.tolist()
for i in range(len(s)):
    s[i] = str(s[i])

for f in s:
    f = os.path.join(folder, f)
    shutil.move(f, new_folder)
    

"""




print("############################### Test")

TSV = "ph_end_new_my_test.tsv"
dataset = COMMONVOICE(hyperparams.path_dataset_common, TSV, 2, mel_limit=2500, reconstructed=True, dias_ph=True)

everything, truefalse = dataset[1]
waveforms, waveform_l, sample_rates, client_ids, sentences, sentences_l = everything
print(waveforms, "\n", sample_rates, "\n", client_ids, "\n", sentences, "\n", truefalse)


print("############################### Collate Test")
listt = [1, 2]
listt[0] = dataset[0]
listt[1] = dataset[1]

sentences_tensor, sentences_mask, spectrograms, mel_mask, waveforms, waveform_l, client_ids, example_id = collate_fn_common(listt, "test", reconstructed_phoneme=True)

print(sentences_tensor.shape, spectrograms.shape)
model = Model_Check(hyperparams)
x = model.forward(sentences_tensor, sentences_mask, spectrograms, mel_mask)
print(x.shape)

"""
print("Sentence tensor \n", sentences_tensor, "\n", sentences_tensor.shape)
print("###############################")
print("Sentence mask \n", sentences_mask, "\n", sentences_mask.shape)
print("###############################")
print("spectrograms \n", spectrograms, "\n", spectrograms.shape)
print("###############################")
print("mel_mask \n", mel_mask, "\n", mel_mask.shape)
print("###############################")
print("waveforms \n", waveforms, "\n", waveforms.shape)
print("###############################")
print("waveform_l \n", waveform_l, "\n", len(waveform_l))
print("###############################")
print("client_ids \n", client_ids, "\n", len(client_ids))
print("###############################")
print("example_id \n", example_id, "\n", len(example_id))
print("###############################")
"""


"""


print("###############################  Train")



TSV = "train.tsv"
dataset = COMMONVOICE(hyperparams.path_dataset_common, TSV, 10)
print("############################### Train numbers")
everything, truefalse = dataset[1]
waveforms, waveform_l, sample_rates, client_ids, sentences, sentences_l = everything
print(waveforms, "\n", sample_rates, "\n", client_ids, "\n", sentences, "\n")
print("############################### Collate Train")

listt = [1, 2]
listt[0] = dataset[0]
listt[1] = dataset[1]

sentences_tensor, sentences_mask, spectrograms, mel_mask, waveforms, waveform_l, client_ids, example_id = collate_fn_common(listt, "train")

print("Sentence tensor \n", sentences_tensor, "\n", sentences_tensor.shape)
print("###############################")
print("Sentence mask \n", sentences_mask, "\n", sentences_mask.shape)
print("###############################")
print("spectrograms \n", spectrograms, "\n", spectrograms.shape)
print("###############################")
print("last spectrograms \n", spectrograms[19], "\n", spectrograms[19][0][120:128],"\n", spectrograms[19][0][120],"\n", spectrograms[19][0][2][1500:1600],"\n",mel_mask[19])
print("###############################")
print("mel_mask \n", mel_mask, "\n", mel_mask.shape)
print("###############################")
print("waveforms \n", waveforms, "\n", waveforms.shape)
print("###############################")
print("waveform_l \n", waveform_l, "\n", len(waveform_l))
print("###############################")
print("client_ids \n", client_ids, "\n", len(client_ids))
print("###############################")
print("example_id \n", example_id, "\n", len(example_id))
print("###############################")


numpy_array = np.array(everything)
print(numpy_array)
print(numpy_array.shape)

print(numpy_array[0][0])

print(type(numpy_array[0][0]))

"""