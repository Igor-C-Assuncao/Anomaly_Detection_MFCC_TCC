# WAE_components/tcn_decoder.py
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm # Mantendo o padrão do TCN_AAE
from .tcn_encoder import TemporalBlock # Reutilizando o TemporalBlock

class TCNDecoder(nn.Module):
    """
    Decoder com Redes de Convolução Temporal (TCN) para WAE.
    Recebe um vetor latente e reconstrói a sequência de MFCCs.
    A arquitetura pode ser um desafio, pois a TCN é naturalmente para mapear sequência para sequência
    ou sequência para um vetor. Para decodificar de um vetor para uma sequência,
    precisamos "expandir" o vetor latente ou usar uma arquitetura TCN transposta.

    Uma abordagem mais simples, inspirada em autoencoders com TCNs, é primeiro
    expandir o vetor latente para o comprimento da sequência desejada e depois aplicar TCNs.
    """
    def __init__(self, embedding_dim, output_size_mfcc, sequence_length, num_channels, kernel_size=3, dropout=0.2):
        """
        Args:
            embedding_dim (int): Dimensão do espaço latente de entrada.
            output_size_mfcc (int): Número de coeficientes MFCC a serem reconstruídos.
            sequence_length (int): Comprimento da sequência de MFCC a ser reconstruída.
            num_channels (list): Lista contendo o número de canais para cada camada TCN.
                                 Ex: [hidden_size] * num_layers.
            kernel_size (int): Tamanho do kernel das convoluções.
            dropout (float): Taxa de dropout.
        """
        super(TCNDecoder, self).__init__()
        self.sequence_length = sequence_length
        self.output_size_mfcc = output_size_mfcc
        self.embedding_dim = embedding_dim

        # Camada para expandir o vetor latente para a dimensão inicial da TCN e o comprimento da sequência
        # A primeira camada da TCN no decoder receberá num_channels[0] como canais de entrada.
        self.initial_linear = nn.Linear(embedding_dim, num_channels[0]) #num_channels[0] * sequence_length)

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            # A primeira camada da TCN recebe num_channels[0] (após o linear)
            # As camadas subsequentes recebem os canais da camada anterior
            in_channels = num_channels[i-1] if i > 0 else num_channels[0]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.tcn_network = nn.Sequential(*layers)

        # Camada final para mapear a saída da TCN para a dimensão dos MFCCs
        self.final_conv = nn.Conv1d(num_channels[-1], output_size_mfcc, kernel_size=1)


    def forward(self, latent_z):
        """
        Args:
            latent_z (torch.Tensor): Vetor latente de entrada.
                                     Shape: (batch_size, embedding_dim)
        Returns:
            torch.Tensor: Sequência de MFCCs reconstruída.
                          Shape: (batch_size, num_mfcc_coefficients, sequence_length)
        """
        # (batch_size, embedding_dim) -> (batch_size, num_channels[0])
        expanded_z = self.initial_linear(latent_z)

        # (batch_size, num_channels[0]) -> (batch_size, num_channels[0], sequence_length)
        # Repetir o vetor ao longo da dimensão do tempo
        # O .unsqueeze(2) adiciona uma dimensão para sequence_length
        # .repeat replica ao longo dessa dimensão
        sequence_input = expanded_z.unsqueeze(2).repeat(1, 1, self.sequence_length)

        # (batch_size, num_channels[0], sequence_length) -> (batch_size, num_channels[-1], sequence_length)
        tcn_out = self.tcn_network(sequence_input)

        # (batch_size, num_channels[-1], sequence_length) -> (batch_size, output_size_mfcc, sequence_length)
        reconstructed_mfcc = self.final_conv(tcn_out)

        return reconstructed_mfcc