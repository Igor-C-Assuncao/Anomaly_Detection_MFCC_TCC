# WAE_components/lstm_discriminator.py
import torch
import torch.nn as nn

class LSTMDiscriminator(nn.Module):
    """
    Discriminador com LSTM para WAE.
    Recebe amostras do espaço latente e retorna um score (crítico de Wasserstein).
    """
    def __init__(self, embedding_dim, hidden_dim, num_layers=1, dropout=0.0, bidirectional=False):
        """
        Args:
            embedding_dim (int): Dimensão do vetor latente de entrada.
            hidden_dim (int): Dimensão do estado oculto da LSTM.
            num_layers (int): Número de camadas da LSTM.
            dropout (float): Taxa de dropout (aplicada entre as camadas LSTM se num_layers > 1).
            bidirectional (bool): Se a LSTM será bidirecional.
        """
        super(LSTMDiscriminator, self).__init__()
        self.embedding_dim = embedding_dim # A entrada da LSTM é o vetor latente Z

        # Para que a LSTM processe Z como uma "sequência", podemos considerá-lo uma sequência de comprimento 1.
        # Ou, se Z já tem uma estrutura sequencial (o que não é o caso comum para o output de um encoder TCN que usamos),
        # poderíamos usar essa estrutura. Assumindo Z é (batch_size, embedding_dim).
        # A LSTM espera (batch_size, seq_len, input_size). Então, Z se torna (batch_size, 1, embedding_dim).
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True, # Importante: (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        self.num_directions = 2 if bidirectional else 1
        # A saída da LSTM será (batch_size, seq_len, num_directions * hidden_dim)
        # Pegamos a saída do último time step: (batch_size, num_directions * hidden_dim)
        self.final_linear = nn.Linear(self.num_directions * hidden_dim, 1) # Saída é um score escalar

    def forward(self, latent_z):
        """
        Args:
            latent_z (torch.Tensor): Amostras do espaço latente.
                                     Shape: (batch_size, embedding_dim)
        Returns:
            torch.Tensor: Score do discriminador para cada amostra.
                          Shape: (batch_size, 1)
        """
        # Transforma latent_z em uma sequência de comprimento 1 para a LSTM
        # (batch_size, embedding_dim) -> (batch_size, 1, embedding_dim)
        lstm_input = latent_z.unsqueeze(1)

        # Passa pela LSTM
        # lstm_out shape: (batch_size, seq_len=1, num_directions * hidden_dim)
        # hidden_state shape: (num_layers * num_directions, batch_size, hidden_dim)
        lstm_out, (hidden_state, cell_state) = self.lstm(lstm_input)

        # Pega a saída do último time step (que é o único, neste caso)
        # lstm_out[:, -1, :] tem shape (batch_size, num_directions * hidden_dim)
        last_time_step_out = lstm_out[:, -1, :]

        # Passa pela camada linear final para obter o score
        score = self.final_linear(last_time_step_out) # Shape: (batch_size, 1)

        # Para WGAN, o output do crítico não passa por sigmoide.
        return score