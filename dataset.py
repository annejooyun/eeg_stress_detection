import scipy
import os
import numpy as np
import pandas as pd
import variables as v


def load_dataset(data_type="ica2", test_type="Arithmetic"):
    '''
    Loads data from the SAM 40 Dataset with the test specified by test_type.
    The data_type parameter specifies which of the datasets to load. Possible values
    are raw, filtered, ica_filtered.
    Returns a Numpy Array with shape (120, 32, 3200).
    '''
    assert (test_type in v.TEST_TYPES)

    assert (data_type in v.DATA_TYPES)

    if data_type == "ica2" and test_type != "Arithmetic":
        print("Data of type", data_type, "does not have test type", test_type)
        return 0

    if data_type == "raw":
        dir = v.DIR_RAW
        data_key = 'Data'
    elif data_type == "filtered":
        dir = v.DIR_FILTERED
        data_key = 'Clean_data'
    elif data_type == "ica1":
        dir = v.DIR_ICA_FILTERED
        data_key = 'Clean_data'
    elif data_type == "ica2":
        dir = v.DIR_ICA_FILTERED_2
        data_key = 'Clean_data'
    else:
        print("No files matching data type found")
        return 0

    dataset = np.empty((120, 32, 3200))

    counter = 0
    for filename in os.listdir(dir):
        if test_type not in filename:
            continue

        f = os.path.join(dir, filename)
        data = scipy.io.loadmat(f)[data_key]
        dataset[counter] = data
        counter += 1
    return dataset


def load_labels():
    '''
    Loads labels from the dataset and transforms the label values to binary values.
    Values larger than 5 are set to 1 and values lower than or equal to 5 are set to zero.
    '''
    labels = pd.read_excel(v.LABELS_PATH)
    labels = labels.rename(columns=v.COLUMNS_TO_RENAME)
    labels = labels[1:]
    labels = labels.astype("int")
    labels = labels > 5
    return labels


def convert_to_epochs(dataset, channels, sfreq):
    '''
    Splits EEG data into epochs with length 1 sec
    '''
    epoched_dataset = np.empty((dataset.shape[0], dataset.shape[2]//sfreq, channels, sfreq))
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[2]//sfreq):
            epoched_dataset[i, j] = dataset[i, :, j*sfreq:(j+1)*sfreq]
    return epoched_dataset

def load_channels():
        root = 'Data'
        coordinates_file = os.path.join(root,"Coordinates.locs") 

        channel_names = []

        with open(coordinates_file, "r") as file:
            for line in file:
                elements = line.split()
                channel = elements[-1]
                channel_names.append(channel)
                
        return channel_names