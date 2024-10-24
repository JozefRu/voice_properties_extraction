import os
import time


import numpy as np
import pandas as pd
import glob
from glob import iglob

import scipy.io.wavfile
import spafe.utils.vis as vis
from spafe.utils.preprocessing import SlidingWindow
from spafe.utils.vis import show_features
import matplotlib.pyplot as plt

from spafe.features.mfcc import mfcc
from spafe.features.rplp import plp
from spafe.features.lfcc import lfcc
from spafe.features.lpc import lpc

####################################################################################
####################################################################################


def calculate_statistics(coeff_data, stress_level, coeff_type, singlefilename):
    # Initialize lists to store statistics for each coefficient dimension
    mean_values = []
    median_values = []
    min_values = []
    max_values = []

    # Loop through each coefficient dimension (first 13 columns)
    for i in range(coeff_data.shape[1]):
        # Calculate statistics for the current dimension
        mean_values.append(np.mean(coeff_data[:, i]))
        median_values.append(np.median(coeff_data[:, i]))
        min_values.append(np.min(coeff_data[:, i]))
        max_values.append(np.max(coeff_data[:, i]))

    # Create a DataFrame for the current file
    current_file_df = pd.DataFrame({
        'Stress Level': [stress_level] * len(mean_values),
        'Coefficient Type': [coeff_type] * len(mean_values),
        'File Name': [singlefilename] * len(mean_values),
        'Mean': mean_values,
        'Median': median_values,
        'Min': min_values,
        'Max': max_values
    })

    # Append the current file results to the aggregated DataFrame
    global aggregated_results_df
    aggregated_results_df = pd.concat([aggregated_results_df, current_file_df], ignore_index=True)


# Function to find all .wav files recursively
def find_wav_files(root_dir):
    wav_files = []
    filename: str
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.wav'):
                relative_path = os.path.relpath(os.path.join(dirpath, filename), root_dir)
                wav_files.append((os.path.join(dirpath, filename), relative_path))
    return wav_files

####################################################################################
####################################################################################

# Initialize a DataFrame to store aggregated results
aggregated_results_df = pd.DataFrame()

# extract input folder name
rootdir_glob = 'D:/Vedecky/DP/CRISIS_1/'
input_folder_name = os.path.basename(os.path.dirname(rootdir_glob))
# output folder 
output_folder = 'output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

file_list = find_wav_files(rootdir_glob)
 

for file, relative_path in file_list:
    
    # Extract stress level from the folder name
    stress_level = os.path.basename(os.path.dirname(file))

    # Extract the file name
    singlefilename = os.path.basename(file)

    # Create the corresponding subdirectory structure within the output folder
    output_subdir = os.path.join(output_folder, os.path.dirname(relative_path))
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    print("processing file ... ", file)
    reader = scipy.io.wavfile.read(file)
    fs, sig = reader
    

    # subfolder of coefficient
    for coeff_type in ["mfccs", "lfccs", "plps", "lpcs"]:
        coeff_output_folder = os.path.join(output_subdir, coeff_type)
        if not os.path.exists(coeff_output_folder):
            os.makedirs(coeff_output_folder)

    ##################
        if coeff_type == "mfccs":
            coeff_data = mfcc(sig,
                fs=fs,
                pre_emph=1,
                pre_emph_coeff=0.97,
                window=SlidingWindow(0.03, 0.015, "hamming"),
                nfilts=128,
                nfft=2048,
                low_freq=0,
                high_freq=8000,
                normalize="mvn")   

        elif coeff_type == "lfccs":
            coeff_data = lfcc(sig,
                fs=fs,
                pre_emph=1,
                pre_emph_coeff=0.97,
                window=SlidingWindow(0.03, 0.015, "hamming"),
                nfilts=128,
                nfft=2048,
                low_freq=0,
                high_freq=8000,
                normalize="mvn")

        elif coeff_type == "plps":
            coeff_data = plp(sig,
                fs=fs,
                pre_emph=0,
                pre_emph_coeff=0.97,
                window=SlidingWindow(0.03, 0.015, "hamming"),
                nfilts=128,
                nfft=1024,
                low_freq=0,
                high_freq=fs/2,
                lifter=0.9,
                normalize="mvn") 

        elif coeff_type == "lpcs":
            coeff_data, _ = lpc(sig,
                    fs=fs,
                    pre_emph=0,
                    pre_emph_coeff=0.97,
                    window=SlidingWindow(0.030, 0.015, "hamming"))

        coeff_data = np.round(coeff_data, decimals=8)
        # stress level + coeff_type + name of file
        stress_col = np.array([stress_level] * len(coeff_data)).reshape(-1, 1)
        coeff_type_col = np.array([coeff_type] * len(coeff_data)).reshape(-1, 1)
        file_name_col = np.array([singlefilename] * len(coeff_data)).reshape(-1, 1)
        coeff_data_with_info = np.hstack((coeff_data, stress_col, coeff_type_col, file_name_col))

        # Save data
        coeff_filename_with_info = os.path.join(coeff_output_folder, os.path.splitext(os.path.basename(file))[0] + f"_{coeff_type}.csv")
        with open(coeff_filename_with_info, "wb") as f:
            np.savetxt(f, coeff_data_with_info, delimiter=',', fmt='%s')

        # Calculate statistics for each column and accumulate results
        calculate_statistics(coeff_data, stress_level, coeff_type, singlefilename)

# Save aggregated results to a single CSV file
aggregated_results_filename = os.path.join(output_folder, 'aggregated_statistics.csv')
aggregated_results_df.to_csv(aggregated_results_filename, index=False)

    # visualize features
    #show_features(mfccs, "Mel Frequency Cepstral Coefﬁcients", "MFCCs Index", "Frame Index")
    #plt.show()  # Add this line to explicitly display the plot

    #show_features(lfccs, "Linear Frequency Cepstral Coefﬁcients", "LFCCs Index", "Frame Index")
    #plt.show()  # Add this line to explicitly display the plot

    #show_features(plps, "Perceptual linear predictions", "PLPs Index", "Frame Index")
    #plt.show()  # Add this line to explicitly display the plot

    #show_features(lpcs, "Linear prediction coefficents", "LPCs Index", "Frame Index")
    #plt.show()  # Add this line to explicitly display the plot




