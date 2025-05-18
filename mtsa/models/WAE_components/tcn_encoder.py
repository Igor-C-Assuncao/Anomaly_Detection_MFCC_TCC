# WAE_components/tcn_encoder.py
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm # Mantendo o padrão do TCN_AAE

# Inspirado em WAE/models/TCN_AAE.py e WAE/models/TCN_AE.py


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding) # Chomp to remove padding
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding) # Chomp to remove padding
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    """
    Encoder com Redes de Convolução Temporal (TCN) para WAE.
    Recebe sequências de MFCCs e retorna um vetor latente.
    """
    def __init__(self, input_size_mfcc, embedding_dim, num_channels, kernel_size=3, dropout=0.2):
        """
        Args:
            input_size_mfcc (int): Número de coeficientes MFCC (dimensão da entrada em cada time step).
            embedding_dim (int): Dimensão do espaço latente de saída.
            num_channels (list): Lista contendo o número de canais para cada camada TCN.
                                 Ex: [hidden_size] * num_layers.
            kernel_size (int): Tamanho do kernel das convoluções.
            dropout (float): Taxa de dropout.
        """
        super(TCNEncoder, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size_mfcc if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.tcn_network = nn.Sequential(*layers)
        # A saída da TCN ainda é uma sequência. Precisamos de uma camada para mapear para o embedding_dim.
        # Uma abordagem comum é usar a saída do último time step ou aplicar um Global Average Pooling.
        # Por simplicidade, vamos usar a saída do último time step após a TCN.
        # A dimensão exata da saída do self.tcn_network dependerá do comprimento da sequência de entrada.
        # Para obter um vetor latente de tamanho fixo 'embedding_dim', podemos precisar de uma camada Linear adicional.
        # Esta camada linear irá operar sobre a saída achatada ou a saída do último passo da TCN.
        # A referência TCN_AAE.py usa uma nn.Linear no final, vamos seguir essa ideia.
        self.final_linear = nn.Linear(num_channels[-1], embedding_dim)


    def forward(self, x_mfcc):
        """
        Args:
            x_mfcc (torch.Tensor): Tensor de entrada com MFCCs.
                                   Shape: (batch_size, num_mfcc_coefficients, sequence_length)
                                   A TCN espera (batch_size, channels, sequence_length).
        """
        # A entrada para Conv1d deve ser (batch_size, channels, seq_length)
        # x_mfcc geralmente é (batch_size, seq_len, num_features) ou (batch_size, num_features, seq_len)
        # Vamos assumir que a entrada x_mfcc já está como (batch_size, num_mfcc_coefficients, sequence_length)
        # Se vier como (batch_size, sequence_length, num_mfcc_coefficients), precisa de x_mfcc.permute(0, 2, 1)

        # Passa pela rede TCN
        tcn_out = self.tcn_network(x_mfcc) # Saída: (batch_size, num_channels[-1], sequence_length)

        # Pega a saída do último time step para representar a sequência
        # Ou poderia ser Global Average Pooling: tcn_out.mean(dim=2)
        last_time_step_out = tcn_out[:, :, -1] # Saída: (batch_size, num_channels[-1])

        # Camada linear final para mapear para o espaço latente
        latent_z = self.final_linear(last_time_step_out) # Saída: (batch_size, embedding_dim)
        return latent_z