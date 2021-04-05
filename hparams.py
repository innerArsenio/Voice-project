from argparse import Namespace

re_dict = {
    "path_dataset_common": "/home/aldeka/senior_sound/COMMONVOICE/gg/",
    "orig_sample_rate": 48000,
    "sampling_rate": 48000, #22050,
    "freq_mask_param": 15,
    "time_mask_param": 35,
    "batch_size": 2,
    "drow_train_examples": 10, # if do first time then reconstructed False do
    "num_workers": 4, 
    #
    #'pin_memory': True,
    "vocab": 100,
    "d_model_emb": 128,
    "d_ff":256,
    "heads": 4,
    "pe_max_len": 300,
    "dropout": 0.1,
    "encoder_number": 8,
    #!
    "local_heads": 2, #more that 0 enable local heads
    "mask_reverse": True, #usually False and True for padded, reversing make False for padded
    "separate_example": False, #if False will directly feed to audio encoder the output needed
    "add_sch": False,
    "off_mask": False,
    #!
    "n_mels": 128,
    "n_mels_ff": 256, 
    "pe_mels_max_len": 2500, 
    "mel_limit":2500,
    #
    "decoder_number": 4,
    #AUDIO
    "attention_type_audio_encoder": "performer", #performer, selfatt, linear
    "feedforward_type_audio_encoder": "glu", # classic, glu
    "local_window_size": 256,
    #TEXT
    "attention_type_text_encoder": "selfatt", #performer, selfatt, linear
    "feedforward_type_text_encoder": "glu", # classic, glu
    #DECODER
    "attention_type_decoder": "selfatt", #performer, selfatt, linear
    "feedforward_type_decoder": "glu", # classic, glu
    #
    "learning_rate": 3e-4,
    "epochs": 100, 
    #
    "reconstructed_phoneme": True, # False and do phonemize
    "reconstructed": True, #if final true, if need to clean false
    "dias_ph": True, #dias reconstructed formating
    "train_path": "ph_end_new_my_train.tsv", 
    "dev_path": "ph_end_new_my_dev.tsv", 
    "test_path": "ph_end_new_my_test.tsv",
}

re_dict_lj = {
    "path_dataset_common": "/content/lj",
    "orig_sample_rate": 22050,
    "sampling_rate": 22050, #22050,
    "freq_mask_param": 15,
    "time_mask_param": 35,
    "batch_size": 32,
    "num_workers": 4, 
    #
    #'pin_memory': True,
    "vocab": 100,
    "d_model_emb": 128,
    "d_ff":256,
    "heads": 4,
    "pe_max_len": 300,
    "dropout": 0.1,
    "encoder_number": 4,
    #!
    "local_heads": 0, #more that 0 enable local heads
    "mask_reverse": False, #usually False and True for padded, reversing make False for padded
    "separate_example": True, #if False will directly feed to audio encoder the output needed
    "add_sch": False,
    "off_mask": False, #off mask for check
    #!
    "n_mels": 80,
    "n_mels_ff": 160, 
    "pe_mels_max_len": 1200, 
    "mel_limit":1200,
    #
    "decoder_number": 3,
    #AUDIO
    "attention_type_audio_encoder": "selfatt", #performer, selfatt, linear
    "feedforward_type_audio_encoder": "glu", # classic, glu
    "local_window_size": 256,
    #TEXT
    "attention_type_text_encoder": "selfatt", #performer, selfatt, linear
    "feedforward_type_text_encoder": "glu", # classic, glu
    #DECODER
    "attention_type_decoder": "selfatt", #performer, selfatt, linear
    "feedforward_type_decoder": "glu", # classic, glu
    #
    "learning_rate": 3e-4,
    "epochs": 100, 
    #
    "reconstructed_phoneme": True, # False and do phonemize

}



hyperparams = Namespace(**re_dict)
