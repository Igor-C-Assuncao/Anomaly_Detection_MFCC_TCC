import os
import pickle as pkl
import numpy as np
from sklearn.preprocessing import StandardScaler
from mtsa.utils import files_train_test_split, Wav2Array
from mtsa.features.mel import Array2Mfcc

def preprocess_mimii_data(data_dir, output_dir, cycle_size=1800, cycle_stride=600):
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
    wav2array = Wav2Array(sampling_rate=16000)
    array2mfcc = Array2Mfcc(sampling_rate=16000)

    # Iterate through machine types
    for machine_type in os.listdir(data_dir):
        machine_path = os.path.join(data_dir, machine_type)
        if not os.path.isdir(machine_path):
            continue

        # Process only `id_00` for training
        id_00_path = os.path.join(machine_path, "id_00")
        if os.path.isdir(id_00_path):
            print(f"Processing {machine_type}/id_00 for training...")

            # Split train and test data
            X_train, X_test, y_train, y_test = files_train_test_split(id_00_path)

            # Convert .wav files to arrays
            train_arrays = wav2array.transform(X_train)
            test_arrays = wav2array.transform(X_test)

            # Convert arrays to MFCC
            train_mfcc = array2mfcc.transform(train_arrays)
            test_mfcc = array2mfcc.transform(test_arrays)

            # Normalize the data
            scaler = StandardScaler()
            train_mfcc = scaler.fit_transform(train_mfcc.reshape(-1, train_mfcc.shape[-1])).reshape(train_mfcc.shape)
            test_mfcc = scaler.transform(test_mfcc.reshape(-1, test_mfcc.shape[-1])).reshape(test_mfcc.shape)

            # Generate cycles
            def generate_cycles(data):
                cycles = []
                for i in range(0, len(data) - cycle_size, cycle_stride):
                    cycles.append(data[i:i + cycle_size])
                return np.array(cycles)

            train_cycles = generate_cycles(train_mfcc)
            test_cycles = generate_cycles(test_mfcc)

            # Save cycles
            train_output_file = os.path.join(output_dir, f"{machine_type}_id_00_train_cycles.pkl")
            test_output_file = os.path.join(output_dir, f"{machine_type}_id_00_test_cycles.pkl")

            with open(train_output_file, "wb") as f:
                pkl.dump(train_cycles, f)
            with open(test_output_file, "wb") as f:
                pkl.dump(test_cycles, f)

            print(f"Saved {len(train_cycles)} train cycles to {train_output_file}")
            print(f"Saved {len(test_cycles)} test cycles to {test_output_file}")

        # Process `id_01` for a separate test set
        id_01_path = os.path.join(machine_path, "id_01")
        if os.path.isdir(id_01_path):
            print(f"Processing {machine_type}/id_01 for separate testing...")

            # Load test data
            X_test, _, y_test, _ = files_train_test_split(id_01_path)

            # Convert .wav files to arrays
            test_arrays = wav2array.transform(X_test)

            # Convert arrays to MFCC
            test_mfcc = array2mfcc.transform(test_arrays)

            # Normalize the data
            test_mfcc = scaler.transform(test_mfcc.reshape(-1, test_mfcc.shape[-1])).reshape(test_mfcc.shape)

            # Generate cycles
            test_cycles = generate_cycles(test_mfcc)

            # Save cycles
            test_output_file = os.path.join(output_dir, f"{machine_type}_id_01_test_cycles.pkl")

            with open(test_output_file, "wb") as f:
                pkl.dump(test_cycles, f)

            print(f"Saved {len(test_cycles)} test cycles to {test_output_file}")

if __name__ == "__main__":
    data_directory = "Data"  # Path to the MIMII dataset
    output_directory = "Data/preprocessed_mimii"  # Path to save preprocessed data
    preprocess_mimii_data(data_directory, output_directory)
