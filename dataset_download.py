from depen import *

def download_dataset_and_extract(dataset_type):
    #directory = os.path.join(hyperparams.path_dataset_common, dataset_type)
    #if not os.path.exists(directory):
    #    os.makedirs(directory)
    
    if dataset_type == "lj":
        print("download this lj") #then to the folder created")

    if dataset_type == "VCTK":
        print("download this version \n version 0.92 \n  extract files") #then to the folder created")
    
    if dataset_type == "COMMONVOICE":
        print("download this version \n en_1932h_2020-06-22 \n  extract files")# then to the folder created")
    return
