import os
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from mtsa.utils import files_train_test_split, Wav2Array ,get_files_from_path
from mtsa.features.mel import Array2Mfcc
from sklearn.model_selection import train_test_split

def preprocess_mimii_data(data_dir, output_dir):
    """
    Preprocess the MIMII dataset by converting .wav files to MFCC arrays,
    normalizing the data, and generating cycles for training and testing.

    Args:
        data_dir (str): Path to the MIMII dataset directory.
        output_dir (str): Path to save the preprocessed data.
        cycle_size (int): Number of samples per cycle.
        cycle_stride (int): Stride for generating cycles.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Wav2Array and Array2Mfcc
    wav2array = Wav2Array(16000)
    array2mfcc = Array2Mfcc(16000)

    # Iterate through machine types
    for machine_type in os.listdir(data_dir):
        machine_path = os.path.join(data_dir, machine_type)
        if not os.path.isdir(machine_path):
            continue

        # Iterate through machine IDs (e.g., id_00, id_01, etc.)
        for machine_id in os.listdir(machine_path):
            id_path = os.path.join(machine_path, machine_id)
            if not os.path.isdir(id_path):
                continue

            print(f"Processing {machine_type}/{machine_id}...")

            # Split train and test data
            # X_train, X_test , y_train , y_test = files_train_test_split(id_path)

            
            normal = get_files_from_path(os.path.join(id_path, "normal"))
            abnormal = get_files_from_path(os.path.join(id_path, "abnormal"))

            X_train, X_test  =  train_test_split(normal, test_size=0.2, random_state=42)

            y_normal_test = [0] * len(X_test)  # Normal samples labeled as 1
            y_abnormal_test = [1] * len(abnormal)  # Abnormal samples labeled as 0

            y_test = y_normal_test + y_abnormal_test

            X_test = list(X_test) + list(abnormal)

          

            # Convert .wav files to arrays
            train_arrays = wav2array.transform(X_train)
            test_arrays = wav2array.transform(X_test)
            


            # Convert arrays to MFCC
            train_mfcc = array2mfcc.transform(train_arrays)
            test_mfcc = array2mfcc.transform(test_arrays)



            # Normalize the data
            scaler = MinMaxScaler()
            train_mfcc_transposed = np.transpose(train_mfcc, (0, 2, 1))
            train_mfcc_reshaped = train_mfcc_transposed.reshape(-1, train_mfcc.shape[1])

            scaler.fit(train_mfcc_reshaped)

            train_mfcc_scaled_reshaped = scaler.transform(train_mfcc_reshaped)

            train_mfcc_scaled_3d = train_mfcc_scaled_reshaped.reshape(train_mfcc.shape[0], train_mfcc.shape[2], train_mfcc.shape[1])
            # Transpor de volta para o formato original (N, M, F)
            train_mfcc_normalized = np.transpose(train_mfcc_scaled_3d, (0, 2, 1)) # Shape: (712, 20, 313)

            # Normalizar dados de teste com o scaler treinado
            test_mfcc_transposed = np.transpose(test_mfcc, (0, 2, 1))
            test_mfcc_reshaped = test_mfcc_transposed.reshape(-1, test_mfcc.shape[1])

            test_mfcc_scaled_reshaped = scaler.transform(test_mfcc_reshaped)

            test_mfcc_scaled_3d = test_mfcc_scaled_reshaped.reshape(test_mfcc.shape[0], test_mfcc.shape[2], test_mfcc.shape[1])
            test_mfcc_normalized = np.transpose(test_mfcc_scaled_3d, (0, 2, 1))

            # Agora train_mfcc_normalized e test_mfcc_normalized estão normalizados por coeficiente MFCC
            # e têm o shape original (N, M, F)
            train_mfcc = train_mfcc_normalized
            test_mfcc = test_mfcc_normalized
            test_labels = y_test
            

           
            
            
            # Generate cycles
            
            def generate_cycles(data): # Pass parameters
                cycles = []
                # Corrected loop range to handle edge cases and include last possible cycle
                
                for i in range(len(data)):
                    cycles.append(data[i:i + 1, : , : ])
               
                return cycles  

            train_cycles = generate_cycles(train_mfcc)
            test_cycles = generate_cycles(test_mfcc)

            # Save cycles
            train_output_file = os.path.join(output_dir, f"{machine_type}_{machine_id}_train_cycles.pkl")
            test_output_file = os.path.join(output_dir, f"{machine_type}_{machine_id}_test_cycles.pkl")
            test_labels_output_file = os.path.join(output_dir, f"{machine_type}_{machine_id}_test_labels.pkl")

            with open(train_output_file, "wb") as f:
                pkl.dump(train_cycles, f)
            with open(test_output_file, "wb") as f:
                pkl.dump(test_cycles, f)
            with open(test_labels_output_file, "wb") as f:
                pkl.dump(test_labels, f)

            print(f"Saved {len(train_cycles)} train cycles to {train_output_file}")
            print(f"Saved {len(test_cycles)} test cycles to {test_output_file}")
            print(f"Saved {len(test_labels)} test labels to {test_labels_output_file}")
           

if __name__ == "__main__":
    data_directory = os.path.join(os.getcwd(),  "Data", "MIMII")  #"Data\MIMII"   Path to the MIMII dataset
    output_directory = "Data\preprocessed_mimii"  # Path to save preprocessed data
    preprocess_mimii_data(data_directory, output_directory)
