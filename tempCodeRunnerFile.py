from mtsa.features.mel import Array2Mfcc

def preprocess_mimii_data(data_dir, output_dir, cycle_size=1800, cycle_stride=600):
    """
    Preprocess the MIMII dataset by converting .wav files to MFCC arrays,
    normalizing the data, and generating cycles for training and testing.

    Args:
        data_dir (str): Path to the MIMII dataset directory.
        output_dir (str): Path to save the preprocess