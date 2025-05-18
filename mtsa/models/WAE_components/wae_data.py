# WAE_components/wae_data.py
import torch
from torch.utils.data import Dataset
import numpy as np

class WAEData(Dataset):
    """
    Dataset customizado para carregar os ciclos de MFCC para o WAE.
    Assume que os dados de entrada (X) são uma lista de arrays numpy,
    onde cada array representa um ciclo/janela de MFCCs.
    """
    def __init__(self, mfcc_cycles_list, transform=None):
        """
        Args:
            mfcc_cycles_list (list of np.ndarray): Lista de ciclos de MFCC.
                Cada ciclo deve ter shape (num_mfcc_coefficients, sequence_length).
            transform (callable, optional): Transformações opcionais a serem aplicadas
                                            em uma amostra.
        """
        self.mfcc_cycles = mfcc_cycles_list
        self.transform = transform

    def __len__(self):
        return len(self.mfcc_cycles)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_mfcc = self.mfcc_cycles[idx] # Shape: (num_mfcc_coefficients, sequence_length)

        # Converter para tensor PyTorch
        sample_mfcc_tensor = torch.from_numpy(sample_mfcc).float()

        # Garante shape (num_mfcc_coefficients, sequence_length)
        if sample_mfcc_tensor.shape[0] > sample_mfcc_tensor.shape[1]:
            sample_mfcc_tensor = sample_mfcc_tensor.t()

        if self.transform:
            sample_mfcc_tensor = self.transform(sample_mfcc_tensor)

        return sample_mfcc_tensor