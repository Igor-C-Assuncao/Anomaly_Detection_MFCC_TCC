# WAE_components/wae_base_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Importando os componentes que criamos na Fase 1 e o WAEData
from .tcn_encoder import TCNEncoder
from .tcn_decoder import TCNDecoder
from .lstm_discriminator import LSTMDiscriminator
from .wae_data import WAEData
# Se NetworkLearnerModel for relevante para observadores ou outras funcionalidades
# from mtsa.models.networkAnalysis.networkLearnerModel import NetworkLearnerModel

# Por enquanto, vamos herdar diretamente de nn.Module.
# A integração com NetworkLearnerModel pode ser feita depois, se necessário.
class WAEBaseModel(nn.Module):
    """
    Modelo base para o Wasserstein Autoencoder (WAE) com TCN Encoder/Decoder
    e LSTM Discriminator, focado em detecção de anomalias em sequências de MFCC.
    """
    def __init__(self,
                 # Parâmetros de entrada/saída e dimensionais
                 input_size_mfcc: int,         # Número de coeficientes MFCC
                 sequence_length: int,         # Comprimento da sequência de MFCC
                 embedding_dim: int,           # Dimensão do espaço latente Z

                 # Parâmetros do Encoder TCN
                 tcn_encoder_channels: list,
                 tcn_decoder_channels: list,
                 lstm_discriminator_hidden_dim: int,
            
              # Lista de canais para as camadas TCN do encoder
                 tcn_encoder_kernel_size: int = 3,
                 tcn_encoder_dropout: float = 0.2,

                 # Parâmetros do Decoder TCN
                 # Lista de canais para as camadas TCN do decoder
                 tcn_decoder_kernel_size: int = 3,
                 tcn_decoder_dropout: float = 0.2,

                 # Parâmetros do Discriminador LSTM
                 
                 lstm_discriminator_num_layers: int = 1,
                 lstm_discriminator_dropout: float = 0.0,
                 lstm_discriminator_bidirectional: bool = False,

                 # Parâmetros de Treinamento
                 lr_autoencoder: float = 1e-3,
                 lr_discriminator: float = 1e-4,
                 weight_decay_autoencoder: float = 1e-5,
                 weight_decay_discriminator: float = 1e-5,
                 lambda_reconstruction: float = 1.0, # Peso da perda de reconstrução
                 lambda_wasserstein: float = 10.0,   # Peso da penalidade de Wasserstein (para o gerador)
                                                     # ou multiplicador para o gradiente penalty no WGAN-GP
                 
                 # Parâmetros do WGAN-GP (se for usar)
                 gradient_penalty_weight: float = 10.0, # lambda_gp no WGAN-GP

                 # Outros
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
                 ):
        super(WAEBaseModel, self).__init__()

        self.input_size_mfcc = input_size_mfcc
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.device = device

        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_wasserstein = lambda_wasserstein # Usado como 'sigma' no paper original do WAE
        self.gradient_penalty_weight = gradient_penalty_weight


        # --- Instanciar Componentes ---
        self.encoder = TCNEncoder(
            input_size_mfcc=input_size_mfcc,
            embedding_dim=embedding_dim,
            num_channels=tcn_encoder_channels,
            kernel_size=tcn_encoder_kernel_size,
            dropout=tcn_encoder_dropout
        ).to(self.device)

        self.decoder = TCNDecoder(
            embedding_dim=embedding_dim,
            output_size_mfcc=input_size_mfcc,
            sequence_length=sequence_length,
            num_channels=tcn_decoder_channels,
            kernel_size=tcn_decoder_kernel_size,
            dropout=tcn_decoder_dropout
        ).to(self.device)

        self.discriminator = LSTMDiscriminator(
            embedding_dim=embedding_dim, # O discriminador opera no espaço latente Z
            hidden_dim=lstm_discriminator_hidden_dim,
            num_layers=lstm_discriminator_num_layers,
            dropout=lstm_discriminator_dropout,
            bidirectional=lstm_discriminator_bidirectional
        ).to(self.device)

        # --- Otimizadores ---
        # Parâmetros do autoencoder (encoder + decoder)
        self.optimizer_autoencoder = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr_autoencoder,
            weight_decay=weight_decay_autoencoder
        )
        # Parâmetros do discriminador
        self.optimizer_discriminator = optim.Adam(
            self.discriminator.parameters(),
            lr=lr_discriminator,
            weight_decay=weight_decay_discriminator
        )

    def _forward_autoencoder(self, mfcc_input_sequence):
        """
        Passa os dados pelo encoder e decoder.
        Args:
            mfcc_input_sequence (torch.Tensor): Shape (batch_size, num_mfcc_coefficients, sequence_length)
        Returns:
            reconstructed_mfcc (torch.Tensor): Shape (batch_size, num_mfcc_coefficients, sequence_length)
            latent_z (torch.Tensor): Shape (batch_size, embedding_dim)
        """
        latent_z = self.encoder(mfcc_input_sequence)
        reconstructed_mfcc = self.decoder(latent_z)
        return reconstructed_mfcc, latent_z

    def _calculate_reconstruction_loss(self, original_mfcc, reconstructed_mfcc):
        """Calcula a perda de reconstrução (MSE por padrão)."""
        return F.mse_loss(reconstructed_mfcc, original_mfcc, reduction='mean')

    def _calculate_gradient_penalty(self, real_samples_prior, fake_samples_latent_z):
        """Calcula o Gradient Penalty para WGAN-GP."""
        batch_size = real_samples_prior.size(0)
        # Alpha varia uniformemente entre 0 e 1 para cada amostra no batch
        alpha = torch.rand(batch_size, 1, device=self.device)
        alpha = alpha.expand_as(real_samples_prior) # (batch_size, embedding_dim)

        # Interpolação entre amostras reais (da prior) e falsas (do encoder)
        interpolated_samples = alpha * real_samples_prior + (1 - alpha) * fake_samples_latent_z
        interpolated_samples.requires_grad_(True)

        # Passa as amostras interpoladas pelo discriminador
        prob_interpolated = self.discriminator(interpolated_samples) # (batch_size, 1)

        # Calcula os gradientes da saída do discriminador em relação às amostras interpoladas
        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated_samples,
            grad_outputs=torch.ones_like(prob_interpolated, device=self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0] # Pegar o primeiro (e único) tensor de gradientes

        # gradients tem shape (batch_size, embedding_dim)
        # Calcular a norma L2 dos gradientes para cada amostra
        gradients_norm = gradients.norm(2, dim=1) # (batch_size)
        
        # Penalidade é a média do quadrado da diferença entre a norma e 1
        gradient_penalty = ((gradients_norm - 1) ** 2).mean()
        return gradient_penalty

    def _train_step_discriminator(self, real_mfcc_batch):
        """Um passo de treinamento para o discriminador (crítico no WGAN)."""
        with torch.backends.cudnn.flags(enabled=False):
            self.optimizer_discriminator.zero_grad()

            # Amostras "reais" para o espaço latente Z vêm de uma distribuição prior (Gaussiana)
            # O tamanho do batch deve ser o mesmo que o das amostras "falsas"
            current_batch_size = real_mfcc_batch.size(0)
            real_samples_prior_z = torch.randn(current_batch_size, self.embedding_dim, device=self.device)

            # Amostras "falsas" para o espaço latente Z vêm do encoder
            with torch.no_grad(): # Não precisamos de gradientes para o encoder aqui
                fake_samples_latent_z = self.encoder(real_mfcc_batch) # (batch_size, embedding_dim)

            # Scores do discriminador para amostras reais e falsas
            d_real_scores = self.discriminator(real_samples_prior_z) # (batch_size, 1)
            d_fake_scores = self.discriminator(fake_samples_latent_z) # (batch_size, 1)

            # Perda de Wasserstein para o discriminador: maximizar E[D(x_real)] - E[D(x_fake)]
            # Ou, minimizar -(E[D(x_real)] - E[D(x_fake)]) = E[D(x_fake)] - E[D(x_real)]
            loss_d = d_fake_scores.mean() - d_real_scores.mean()

            # Adicionar Gradient Penalty (WGAN-GP)
            gradient_penalty = self._calculate_gradient_penalty(real_samples_prior_z.detach(), fake_samples_latent_z.detach())
            loss_d += self.gradient_penalty_weight * gradient_penalty
            
            loss_d.backward()
            self.optimizer_discriminator.step()

            # Para WGAN original (sem GP), seria necessário fazer "weight clipping" nos parâmetros do discriminador.
            # Ex: for p in self.discriminator.parameters(): p.data.clamp_(-0.01, 0.01)
            # Mas WGAN-GP é geralmente preferido.

            return loss_d.item()


    def _train_step_autoencoder(self, real_mfcc_batch):
        """Um passo de treinamento para o autoencoder (encoder + decoder / gerador)."""
        self.optimizer_autoencoder.zero_grad()

        # Forward pass pelo autoencoder
        reconstructed_mfcc, latent_z_from_encoder = self._forward_autoencoder(real_mfcc_batch)

        # 1. Perda de Reconstrução
        loss_reconstruction = self._calculate_reconstruction_loss(real_mfcc_batch, reconstructed_mfcc)

        # 2. Perda Adversarial (Wasserstein) para o Gerador (Encoder)
        # O encoder quer gerar latent_z que o discriminador pense que veio da prior.
        # Queremos maximizar E[D(encoder(x_real))], ou minimizar -E[D(encoder(x_real))]
        d_scores_on_generated_z = self.discriminator(latent_z_from_encoder) # (batch_size, 1)
        loss_generator_wasserstein = -d_scores_on_generated_z.mean()
        
        # Perda total do autoencoder
        # O paper WAE original usa: L_recons + sigma * L_MMD ou L_recons + sigma * L_adv_discriminator
        # No WGAN, a perda do gerador é tipicamente -E[D(G(z_prior))].
        # Aqui, G é o encoder, e ele mapeia de x para Z, enquanto o discriminador opera em Z.
        # A perda adversarial visa fazer Q(Z) (distribuição de latent_z_from_encoder)
        # ser similar a P(Z) (distribuição da prior gaussiana).
        total_loss_ae = (self.lambda_reconstruction * loss_reconstruction +
                         self.lambda_wasserstein * loss_generator_wasserstein)
        
        total_loss_ae.backward()
        self.optimizer_autoencoder.step()

        return loss_reconstruction.item(), loss_generator_wasserstein.item()

    def fit(self,
            mfcc_cycles_list_train: list, # Lista de ciclos de MFCC para treino
            epochs: int = 10,
            batch_size: int = 32,
            n_critic_steps: int = 5 # Número de passos de treino do discriminador por passo do AE
            ):
        """
        Treina o modelo WAE-GAN.
        Args:
            mfcc_cycles_list_train: Lista de arrays numpy, cada um sendo um ciclo de MFCC.
            epochs: Número de épocas de treinamento.
            batch_size: Tamanho do batch.
            n_critic_steps: Número de vezes que o discriminador é treinado por cada vez que o AE é treinado.
        """
        self.train() # Coloca o módulo em modo de treinamento (afeta Dropout, BatchNorm, etc.)

        train_dataset = WAEData(mfcc_cycles_list=mfcc_cycles_list_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        print(f"Iniciando treinamento WAEBaseModel por {epochs} épocas...")
        print(f"Dispositivo: {self.device}")

        for epoch in range(epochs):
            epoch_loss_d_avg = 0.0
            epoch_loss_reconstruction_avg = 0.0
            epoch_loss_g_wasserstein_avg = 0.0
            
            # Usar tqdm para barra de progresso no dataloader
            batch_iterator = tqdm(train_dataloader, desc=f"Época {epoch+1}/{epochs}", unit="batch")

            for i, real_mfcc_batch in enumerate(batch_iterator):
                real_mfcc_batch = real_mfcc_batch.to(self.device)

                # --- Treinar Discriminador (Crítico) ---
                # No WGAN, o crítico é treinado mais vezes que o gerador
                current_loss_d = 0
                for _ in range(n_critic_steps):
                    loss_d_step = self._train_step_discriminator(real_mfcc_batch)
                    current_loss_d += loss_d_step
                epoch_loss_d_avg += (current_loss_d / n_critic_steps)


                # --- Treinar Autoencoder (Gerador) ---
                # Treina o AE/Gerador com menos frequência
                # if i % n_critic_steps == 0: # Ou pode treinar a cada batch
                loss_reconstruction, loss_g_wasserstein = self._train_step_autoencoder(real_mfcc_batch)
                epoch_loss_reconstruction_avg += loss_reconstruction
                epoch_loss_g_wasserstein_avg += loss_g_wasserstein
                
                # Atualizar a descrição da barra de progresso
                batch_iterator.set_postfix({
                    "L_D": f"{(current_loss_d / n_critic_steps):.4f}",
                    "L_Rec": f"{loss_reconstruction:.4f}",
                    "L_G_Wass": f"{loss_g_wasserstein:.4f}"
                })

            num_batches = len(train_dataloader)
            epoch_loss_d_avg /= num_batches
            epoch_loss_reconstruction_avg /= num_batches
            epoch_loss_g_wasserstein_avg /= num_batches
            
            print(f"Fim da Época {epoch+1}: "
                  f"Perda Discriminador Média: {epoch_loss_d_avg:.4f}, "
                  f"Perda Reconstrução Média: {epoch_loss_reconstruction_avg:.4f}, "
                  f"Perda Gerador (Wass) Média: {epoch_loss_g_wasserstein_avg:.4f}")
            
        print("Treinamento concluído.")


    @torch.no_grad() # Desabilitar cálculo de gradientes para inferência
    def predict_anomaly_score(self, mfcc_input_sequence):
        """
        Calcula a pontuação de anomalia para uma sequência de MFCC de entrada.
        Args:
            mfcc_input_sequence (torch.Tensor): Shape (batch_size, num_mfcc_coefficients, sequence_length)
                                               ou (num_mfcc_coefficients, sequence_length) para uma única amostra.
        Returns:
            final_anomaly_score (torch.Tensor): Pontuação de anomalia. Shape (batch_size,)
        """
        self.eval() # Coloca o módulo em modo de avaliação

        if mfcc_input_sequence.ndim == 2: # Se for uma única amostra
            mfcc_input_sequence = mfcc_input_sequence.unsqueeze(0) # Adiciona dimensão do batch

        mfcc_input_sequence = mfcc_input_sequence.to(self.device)

        reconstructed_mfcc, latent_z = self._forward_autoencoder(mfcc_input_sequence)

        # 1. Erro de Reconstrução (por amostra no batch)
        reconstruction_error = F.mse_loss(reconstructed_mfcc, mfcc_input_sequence, reduction='none')
        reconstruction_error_per_sample = reconstruction_error.mean(dim=[1, 2]) # Média sobre canais e tempo

        # 2. Saída do Discriminador no espaço latente Z
        # No WAE, um Z "ruim" (anômalo) pode ter um score do discriminador muito negativo
        # (se o discriminador foi treinado para dar scores altos para a prior e baixos para o encoder).
        # Ou, se a tarefa é distinguir prior de encoder_output, um Z anômalo pode não se parecer
        # nem com a prior nem com o que o encoder tipicamente produz para dados normais.
        # A ideia do seu TCC é usar o Z-score da saída do discriminador.
        # Para isso, precisaríamos da média e desvio padrão das saídas do discriminador
        # em dados de treinamento (normais). Vamos assumir que isso é pré-calculado ou
        # que o score direto do discriminador já é informativo.

        # Por agora, vamos retornar o erro de reconstrução e o score do discriminador separadamente.
        # A combinação pode ser feita fora ou aqui se tivermos `mean_disc_score_train` e `std_disc_score_train`.

        # Para o WAE, um score de anomalia pode ser simplesmente o erro de reconstrução,
        # e a regularização de Wasserstein no espaço latente ajuda a tornar essa reconstrução mais significativa.
        # Ou pode ser uma combinação.
        # O artigo ECML23.pdf (Silva et al.) combina o erro de reconstrução com o Z-score da saída do discriminador. 
        # Score_final_anomalia = Erro_Reconstrução * |Z-score(Saida_Discriminador(Z))|

        # Vamos retornar o erro de reconstrução por enquanto. A pontuação combinada pode ser uma variação.
        # Se quisermos implementar a pontuação do ECML23:
        #   - Precisaríamos treinar e depois calcular média/std das saídas do discriminador em dados normais de treino.
        #   - `discriminator_output_on_z = self.discriminator(latent_z)`
        #   - `z_score = (discriminator_output_on_z - self.mean_disc_train) / self.std_disc_train`
        #   - `final_anomaly_score = reconstruction_error_per_sample * torch.abs(z_score.squeeze())`

        # Para este esqueleto, vamos focar no erro de reconstrução como score de anomalia primário.
        # O WAE em si já força o espaço latente a ser "bom".
        
        # Um score de anomalia pode ser o erro de reconstrução.
        # Ou, o quão "fora da distribuição prior" está o latent_z, medido pelo discriminador.
        # Se D(z) é alto para z da prior, e baixo para z "ruim", então -D(encoder(x_anomalo)) seria alto.
        discriminator_scores_on_z = self.discriminator(latent_z).squeeze() # (batch_size)

        # Retornamos ambos para flexibilidade
        return reconstruction_error_per_sample, discriminator_scores_on_z


    @torch.no_grad()
    def score_samples(self, mfcc_cycles_list_test: list, batch_size: int = 32, 
                      mean_disc_score_train: float = 0.0, std_disc_score_train: float = 1.0):
        """
        Calcula as pontuações de anomalia para uma lista de ciclos de MFCC.
        Implementa a combinação de erro de reconstrução com Z-score da saída do discriminador.
        Args:
            mfcc_cycles_list_test: Lista de arrays numpy (ciclos de MFCC).
            batch_size: Tamanho do batch para inferência.
            mean_disc_score_train: Média das saídas do discriminador em dados normais de treino.
            std_disc_score_train: Desvio padrão das saídas do discriminador em dados normais de treino.

        Returns:
            np.ndarray: Array com as pontuações de anomalia finais.
        """
        self.eval()
        test_dataset = WAEData(mfcc_cycles_list=mfcc_cycles_list_test)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        all_final_scores = []

        for mfcc_batch in test_dataloader:
            mfcc_batch = mfcc_batch.to(self.device)
            
            reconstruction_errors, disc_scores_on_z = self.predict_anomaly_score(mfcc_batch)
            
            # Calcular Z-score para as saídas do discriminador
            # Evitar divisão por zero se std_disc_score_train for muito pequeno
            if std_disc_score_train < 1e-6: # Um pequeno epsilon
                z_scores_disc = torch.zeros_like(disc_scores_on_z)
            else:
                z_scores_disc = (disc_scores_on_z - mean_disc_score_train) / std_disc_score_train
            
            # Pontuação final combinada
            final_scores_batch = reconstruction_errors * torch.abs(z_scores_disc)
            all_final_scores.append(final_scores_batch.cpu().numpy())
            
        return np.concatenate(all_final_scores)

    def calculate_discriminator_score_stats_on_training(self, mfcc_cycles_list_train: list, batch_size: int = 32):
        """
        Calcula a média e o desvio padrão das saídas do discriminador (crítico)
        nos dados de treinamento (que se espera serem normais).
        Isso é necessário para o cálculo do Z-score na pontuação de anomalia combinada.
        """
        self.eval()
        train_dataset = WAEData(mfcc_cycles_list=mfcc_cycles_list_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        all_disc_scores = []
        with torch.no_grad():
            for mfcc_batch in train_dataloader:
                mfcc_batch = mfcc_batch.to(self.device)
                latent_z = self.encoder(mfcc_batch)
                disc_scores = self.discriminator(latent_z).squeeze()
                all_disc_scores.append(disc_scores.cpu().numpy())
        
        all_disc_scores_np = np.concatenate(all_disc_scores)
        mean_disc_score = float(np.mean(all_disc_scores_np))
        std_disc_score = float(np.std(all_disc_scores_np))
        
        print(f"Estatísticas das saídas do Discriminador no treino (normal): Média={mean_disc_score:.4f}, Std={std_disc_score:.4f}")
        return mean_disc_score, std_disc_score