import os
import pickle as pkl
import numpy as np
from mtsa.models.mfccmix import MFCCMix
from mtsa.utils import files_train_test_split

def preprocess_mimii_data(data_dir, output_dir, chunk_size=1800, chunk_stride=600):
    """
    Preprocess the MIMII dataset by converting .wav files to MFCC arrays,
    and generating chunks for training and testing.

    Args:
        data_dir (str): Path to the MIMII dataset directory.
        output_dir (str): Path to save the preprocessed data.
        chunk_size (int): Number of samples per chunk.
        chunk_stride (int): Stride for generating chunks.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize MFCCMix
    mfcc_transformer = MFCCMix(sampling_rate=16000)

    # Iterate through machine types and IDs
    for machine_type in os.listdir(data_dir):
        machine_path = os.path.join(data_dir, machine_type)
        if not os.path.isdir(machine_path):
            continue

        for machine_id in os.listdir(machine_path):
            id_path = os.path.join(machine_path, machine_id)
            if not os.path.isdir(id_path):
                continue

            print(f"Processing {machine_type}/{machine_id}...")

            # Split train and test data
            X_train, X_test, y_train, y_test = files_train_test_split(id_path)

            # Transform train and test data to MFCC
            train_mfcc = mfcc_transformer.transform(X_train)
            test_mfcc = mfcc_transformer.transform(X_test)

            # Generate chunks
            def generate_chunks(data):
                chunks = []
                for i in range(0, len(data) - chunk_size, chunk_stride):
                    chunks.append(data[i:i + chunk_size])
                return np.array(chunks)

            train_chunks = generate_chunks(train_mfcc)
            test_chunks = generate_chunks(test_mfcc)

            # Save chunks
            train_output_file = os.path.join(output_dir, f"{machine_type}_{machine_id}_train_chunks.pkl")
            test_output_file = os.path.join(output_dir, f"{machine_type}_{machine_id}_test_chunks.pkl")

            with open(train_output_file, "wb") as f:
                pkl.dump(train_chunks, f)
            with open(test_output_file, "wb") as f:
                pkl.dump(test_chunks, f)

            print(f"Saved {len(train_chunks)} train chunks to {train_output_file}")
            print(f"Saved {len(test_chunks)} test chunks to {test_output_file}")

if __name__ == "__main__":
    data_directory = "Data/"  # Path to the MIMII dataset
    output_directory = "Data/preprocessed_mimii"  # Path to save preprocessed data
    preprocess_mimii_data(data_directory, output_directory)
