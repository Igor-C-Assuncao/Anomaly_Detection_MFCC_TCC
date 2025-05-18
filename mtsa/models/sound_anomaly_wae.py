import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler # Para normalização Z-score
from tqdm import tqdm # Para a barra de progresso, se necessário aqui também

# Componentes de pré-processamento (você já os tem no seu projeto mtsa)
from mtsa.utils import Wav2Array # Se for processar .wav diretamente
from mtsa.features.mel import Array2Mfcc

# Nosso WAEBaseModel e WAEData
from mtsa.models.WAE_components.wae_base_model import WAEBaseModel
from mtsa.models.WAE_components.wae_data import WAEData # Embora WAEData seja usado internamente por WAEBaseModel

# Classe auxiliar para permitir que StandardScaler trabalhe com listas de arrays 2D (MFCCs)
# ou para aplicar a normalização dentro do WAEBaseModel ou WAEData.
# Por simplicidade inicial, vamos assumir que o pré-processamento (incluindo normalização)
# dos MFCCs já foi feito antes de chegar ao WAEBaseModel.
# No entanto, um StandardScaler no pipeline é mais elegante.

class MfccStandardScaler(BaseEstimator, OutlierMixin):
    """
    Um StandardScaler customizado para normalizar listas de arrays MFCC.
    Cada array MFCC (amostra) é normalizado individualmente ou globalmente.
    Para detecção de anomalias, geralmente ajustamos o scaler em dados normais de treino.
    """
    def __init__(self, global_scaling=True):
        self.global_scaling = global_scaling
        if self.global_scaling:
            self.scaler = MinMaxScaler()
        else:
            self.scalers = [] # Um scaler por característica MFCC, se necessário

    def fit(self, X_mfcc_list, y=None):
        """
        Ajusta o scaler.
        X_mfcc_list: Lista de arrays numpy, onde cada array é (num_coeffs, seq_len).
        """
        if self.global_scaling:
            # Concatenar todos os MFCCs ao longo da dimensão dos coeficientes (ou achatá-los)
            # para ajustar um único scaler. Cuidado com a memória.
            # Uma abordagem mais robusta seria ajustar nas features achatadas (coef_i, tempo_j).
            # Para simplificar, podemos ajustar em cada coeficiente MFCC independentemente.
            # Ex: Tratar cada coeficiente MFCC como uma feature e todos os time steps como amostras.
            
            # Shape de entrada X_mfcc_list[0]: (num_coeffs, seq_len)
            # Vamos achatar e concatenar para (N_total_timesteps_coeffs, 1) e ajustar
            # Ou melhor: (N_samples * seq_len, num_coeffs) para ajustar por coeficiente.
            if len(X_mfcc_list) == 0:
                return self
            
            num_coeffs = X_mfcc_list[0].shape[0]
            all_coeffs_data = [[] for _ in range(num_coeffs)]

            for mfcc_array in X_mfcc_list: # mfcc_array é (num_coeffs, seq_len)
                for i in range(num_coeffs):
                    all_coeffs_data[i].extend(mfcc_array[i, :].tolist())
            
            # Agora temos uma lista de listas, onde cada sublista são todos os valores de um coef.
            # Precisamos de um scaler por coeficiente se não for global sobre tudo.
            # Se global_scaling for True e quisermos um scaler único para todas as features e tempos:
            #   flat_data = np.concatenate([arr.flatten() for arr in X_mfcc_list]).reshape(-1, 1)
            #   self.scaler.fit(flat_data)
            # Mas isso perde a estrutura. Melhor normalizar por coeficiente ou por (coef, tempo).
            # Seu TCC menciona Z-Score para garantir média zero e desvio padrão um para *todas as características*.
            # Isso sugere que cada (coeficiente MFCC no tempo t) é uma característica.
            # O WAE/generate_cycles.py usa StandardScaler em cada ciclo:
            #   df_slice[df_slice.columns] = scaler.fit_transform(df_slice[df_slice.columns])
            #   Isso normaliza cada sensor (coluna) independentemente.
            #   No nosso caso, cada coeficiente MFCC pode ser uma "coluna".

            # Vamos seguir uma abordagem onde cada coeficiente MFCC é normalizado através do tempo e das amostras.
            # (N_amostras * N_tempos, N_coeficientes_MFCC)
            # Transpor MFCCs para (seq_len, num_coeffs) e depois concatenar.
            concatenated_mfcc_frames = np.concatenate([np.squeeze(mfcc).T for mfcc in X_mfcc_list], axis=0)
            # concatenated_mfcc_frames tem shape (total_frames, num_coeffs)
            self.scaler.fit(concatenated_mfcc_frames)
        else:
            # Implementar lógica para múltiplos scalers se necessário
            pass
        return self

    def transform(self, X_mfcc_list):
        if len(X_mfcc_list) == 0:
            return []
        
        transformed_list = []
        if self.global_scaling:
            for mfcc_array in X_mfcc_list: # mfcc_array é (num_coeffs, seq_len)
                # Transpor para (seq_len, num_coeffs) para aplicar o scaler
                mfcc_T = mfcc_array.T
                # Garante que mfcc_T é 2D (seq_len, num_coeffs)
                mfcc_T = np.squeeze(mfcc_T)
                if mfcc_T.ndim == 1:
                    mfcc_T = mfcc_T.reshape(-1, 1)
                scaled_mfcc_T = self.scaler.transform(mfcc_T)
                # Transpor de volta para (num_coeffs, seq_len)
                transformed_list.append(scaled_mfcc_T.T)
        else:
            # Implementar lógica para múltiplos scalers
            pass
        return transformed_list

    def fit_transform(self, X_mfcc_list, y=None):
        self.fit(X_mfcc_list, y)
        return self.transform(X_mfcc_list)


class SoundAnomalyWAE(nn.Module, BaseEstimator, OutlierMixin):
    def __init__(self,
                 # Parâmetros de pré-processamento
                 sampling_rate: int = 16000, # Taxa de amostragem do áudio
                 num_mfcc_coeffs: int = 20,  # Número de coeficientes MFCC a extrair
                 mfcc_fft_window_size: int = 512, # Ex: n_fft para librosa.feature.mfcc
                 mfcc_hop_length: int = 256,    # Ex: hop_length para librosa.feature.mfcc
                 # (Outros parâmetros de MFCC podem ser adicionados aqui)

                 # Parâmetros do WAEBaseModel (passados diretamente)
                 sequence_length: int = 313, # Comprimento da sequência de MFCC (número de frames)
                                             # Este valor precisa ser consistente com a saída do Array2Mfcc
                 embedding_dim: int = 64,    # Exemplo
                 tcn_encoder_channels: list = [64, 128], # Exemplo
                 tcn_decoder_channels: list = [128, 64], # Exemplo
                 lstm_discriminator_hidden_dim: int = 128, # Exemplo
                 # (Outros parâmetros do WAEBaseModel)
                 
                 # Parâmetros de treinamento para o método fit desta classe
                 epochs: int = 10,
                 batch_size: int = 32,
                 n_critic_steps: int = 5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 # Adicione outros parâmetros do WAEBaseModel aqui se quiser expô-los no construtor principal
                 lr_autoencoder: float = 1e-3,
                 lr_discriminator: float = 1e-4,
                 lambda_reconstruction: float = 1.0,
                 lambda_wasserstein: float = 10.0,
                 gradient_penalty_weight: float = 10.0
                 ):
        super(SoundAnomalyWAE, self).__init__()

        self.sampling_rate = sampling_rate
        self.num_mfcc_coeffs = num_mfcc_coeffs
        self.mfcc_fft_window_size = mfcc_fft_window_size
        self.mfcc_hop_length = mfcc_hop_length
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_critic_steps = n_critic_steps
        self.device_name = device # Renomeado para evitar conflito com self.device de nn.Module

        # Guardar os parâmetros do WAEBaseModel para passá-los no _build_model
        self.wae_params = {
            "input_size_mfcc": num_mfcc_coeffs,
            "sequence_length": sequence_length,
            "embedding_dim": embedding_dim,
            "tcn_encoder_channels": tcn_encoder_channels,
            "tcn_decoder_channels": tcn_decoder_channels,
            "lstm_discriminator_hidden_dim": lstm_discriminator_hidden_dim,
            "lr_autoencoder": lr_autoencoder,
            "lr_discriminator": lr_discriminator,
            "lambda_reconstruction": lambda_reconstruction,
            "lambda_wasserstein": lambda_wasserstein,
            "gradient_penalty_weight": gradient_penalty_weight,
            "device": self.device_name
            # Adicionar outros parâmetros do WAEBaseModel aqui
        }
        
        # Para armazenar estatísticas do discriminador após o treino
        self.mean_disc_score_train_ = 0.0
        self.std_disc_score_train_ = 1.0

        # O pipeline é construído no _build_model e atribuído a self.model_pipeline
        self.model_pipeline_ = self._build_model()


    def _build_model(self):
        # Etapas do Pipeline
        pipeline_steps = []

        # 1. Conversão de Áudio para Array (se a entrada for caminhos de arquivo .wav)
        # Se a entrada já for uma lista de arrays numpy de áudio, esta etapa pode ser pulada
        # ou adaptada. Seu preprocessing_mimii.py parece já fazer isso.
        # pipeline_steps.append(('wav2array', Wav2Array(sampling_rate=self.sampling_rate, mono=True)))

        # 2. Extração de MFCC
        # A classe Array2Mfcc no seu projeto mtsa já existe.
        # Ela espera uma lista de arrays de áudio.
        pipeline_steps.append(('array2mfcc', Array2Mfcc(
            sampling_rate=self.sampling_rate,
            # Passe outros parâmetros do MFCC se a classe Array2Mfcc os aceitar,
            # como n_mfcc=self.num_mfcc_coeffs, n_fft, hop_length.
            # Atualmente, Array2Mfcc em mtsa/features/mel.py só pega sampling_rate.
            # Você pode precisar modificá-la ou criar uma nova que aceite mais params.
            # Por enquanto, o número de coeficientes é fixo em 20 por librosa.
            # O comprimento da sequência (sequence_length) será determinado pelo Array2Mfcc.
            # É CRUCIAL que self.wae_params["sequence_length"] corresponda ao que Array2Mfcc produz.
        )))

        # 3. Normalização Z-Score dos MFCCs
        # Usaremos nosso MfccStandardScaler customizado.
        pipeline_steps.append(('mfcc_scaler', MfccStandardScaler(global_scaling=True)))

        # 4. Modelo WAEBaseModel
        # **Importante**: WAEBaseModel espera uma lista de tensores MFCC já processados.
        # O pipeline do Scikit-learn passará a saída da etapa anterior.
        # No entanto, WAEBaseModel é um nn.Module, não um transformador Scikit-learn.
        # Para integrá-lo diretamente em um pipeline sklearn, precisaríamos de um wrapper.
        # Uma abordagem mais simples é ter WAEBaseModel como o componente principal
        # e o pipeline sklearn cuidar apenas do pré-processamento.
        # A classe SoundAnomalyWAE então orquestraria isso.

        # Alternativa: Não usar pipeline sklearn para o modelo final, mas sim chamá-lo diretamente.
        # O pipeline aqui seria só para pré-processamento.
        
        # Vamos manter o WAEBaseModel como o atributo principal e o pipeline para pré-proc.
        self.preprocessing_pipeline_ = Pipeline(steps=pipeline_steps)
        
        # O WAEBaseModel é instanciado separadamente
        self.wae_model_ = WAEBaseModel(**self.wae_params).to(self.device_name)
        
        return self.preprocessing_pipeline_ # Retorna o pipeline de pré-processamento


    def fit(self, X_audio_list, y=None):
        """
        Treina o modelo WAE.
        Args:
            X_audio_list (list): Lista de arrays numpy, cada um representando um sinal de áudio bruto.
                                OU lista de caminhos para arquivos .wav, se Wav2Array for usado.
                                Assumindo que preprocessing_mimii.py já criou os ciclos de MFCC.
                                Portanto, X_audio_list aqui deve ser a lista de ciclos de MFCC.
            y: Ignorado (treinamento não supervisionado para o WAE).
        """
        print("Iniciando pré-processamento dos dados de treinamento...")
        # Se X_audio_list já são os MFCCs (lista de arrays numpy)
        # mfcc_cycles_processed = self.preprocessing_pipeline_.fit_transform(X_audio_list)
        
        # Se X_audio_list são os dados brutos que precisam passar por Array2Mfcc e Scaler
        # Ex: X_audio_list = [audio_array1, audio_array2, ...]
        # OU, se preprocessing_mimii.py já gerou os arquivos .pkl com os ciclos de MFCC:
        #   Nesse caso, X_audio_list seria a lista de MFCCs carregada desses arquivos.
        
        # Assumindo que X_audio_list é uma lista de ciclos de MFCC (numpy arrays)
        # cada um com shape (num_coeffs, seq_len_original_do_ciclo)
        
        # 1. Ajustar e transformar com o pipeline de pré-processamento
        #   Isto irá ajustar o MfccStandardScaler e normalizar os dados.
        print("Ajustando e transformando dados de treino com o pipeline de pré-processamento...")
        mfcc_cycles_normalized = self.preprocessing_pipeline_.fit_transform(X_audio_list)
        # mfcc_cycles_normalized é uma lista de arrays numpy normalizados.

        print(f"Pré-processamento concluído. {len(mfcc_cycles_normalized)} amostras processadas.")
        if not mfcc_cycles_normalized:
            print("Nenhuma amostra após o pré-processamento. Verifique os dados de entrada.")
            return self

        # Verificar se sequence_length corresponde após Array2Mfcc
        # Esta lógica precisaria ser mais robusta se Array2Mfcc não garantir um seq_len fixo.
        # Se Array2Mfcc (ou seu pré-processamento anterior) já garante sequence_length fixo:
        actual_seq_len = mfcc_cycles_normalized[0].shape[1]
        if actual_seq_len != self.wae_params["sequence_length"]:
            print(f"AVISO: sequence_length esperado pelo WAE ({self.wae_params['sequence_length']}) "
                  f"difere do sequence_length real dos MFCCs processados ({actual_seq_len}). "
                  f"Ajustando sequence_length do WAE.")
            self.wae_params["sequence_length"] = actual_seq_len
            # Re-instanciar o WAEBaseModel com o sequence_length correto
            self.wae_model_ = WAEBaseModel(**self.wae_params).to(self.device_name)
            # O WAEBaseModel deve ser re-instanciado aqui com o novo sequence_length
            # Isso é complicado se já estivermos dentro do fit. Idealmente, sequence_length
            # deve ser conhecido ou determinado antes de instanciar WAEBaseModel.
            # Por agora, vamos assumir que o sequence_length fornecido é o correto.

        # 2. Treinar o WAEBaseModel
        print("Iniciando treinamento do WAEBaseModel...")
        self.wae_model_.fit(
            mfcc_cycles_list_train=mfcc_cycles_normalized,
            epochs=self.epochs,
            batch_size=self.batch_size,
            n_critic_steps=self.n_critic_steps
        )

        # 3. Calcular e armazenar estatísticas do discriminador nos dados de treinamento (normais)
        print("Calculando estatísticas do discriminador nos dados de treinamento...")
        self.mean_disc_score_train_, self.std_disc_score_train_ = \
            self.wae_model_.calculate_discriminator_score_stats_on_training(
                mfcc_cycles_list_train=mfcc_cycles_normalized,
                batch_size=self.batch_size
            )
        
        print("Treinamento e cálculo de estatísticas concluídos.")
        return self

    def score_samples(self, X_audio_list_test):
        """
        Calcula a pontuação de anomalia para novas amostras.
        Args:
            X_audio_list_test (list): Lista de arrays numpy de áudio bruto para teste,
                                     OU lista de ciclos de MFCC já extraídos.
        Returns:
            np.ndarray: Array de pontuações de anomalia.
        """
        self.wae_model_.eval() # Garantir que o modelo base está em modo de avaliação

        print("Iniciando pré-processamento dos dados de teste...")
        # Aplicar apenas 'transform' do pipeline de pré-processamento (não 'fit_transform')
        # O MfccStandardScaler usará a média e std aprendidas no treino.
        mfcc_cycles_test_normalized = self.preprocessing_pipeline_.transform(X_audio_list_test)
        
        print(f"Pré-processamento de teste concluído. {len(mfcc_cycles_test_normalized)} amostras processadas.")
        if not mfcc_cycles_test_normalized:
            print("Nenhuma amostra de teste após o pré-processamento.")
            return np.array([])

        print("Calculando pontuações de anomalia...")
        anomaly_scores = self.wae_model_.score_samples(
            mfcc_cycles_list_test=mfcc_cycles_test_normalized,
            batch_size=self.batch_size,
            mean_disc_score_train=self.mean_disc_score_train_,
            std_disc_score_train=self.std_disc_score_train_
        )
        return anomaly_scores # Já é um array numpy

    def predict(self, X_audio_list_test, threshold=None):
        """
        Prediz se as amostras são anomalias com base em um limiar.
        Args:
            X_audio_list_test (list): Dados de teste.
            threshold (float, optional): Limiar para classificar como anomalia.
                                         Se None, um valor padrão pode ser usado ou um erro lançado.
                                         O paper ECML23 usa o método boxplot para definir o limiar.
        Returns:
            np.ndarray: Array de rótulos (0 para normal, 1 para anomalia).
        """
        if threshold is None:
            # Você precisará definir uma estratégia para o threshold.
            # Pode ser com base na distribuição dos scores_samples nos dados de treino,
            # por exemplo, Q3 + 1.5 * IQR, ou Q3 + 3 * IQR como no ECML23.pdf 
            # Este cálculo do threshold deveria ser feito após o fit, nos scores dos dados de treino.
            # Por enquanto, vamos usar um placeholder.
            raise ValueError("Threshold deve ser fornecido para predição.")

        anomaly_scores = self.score_samples(X_audio_list_test)
        
        # Pontuações mais altas indicam maior anomalia
        predictions = (anomaly_scores > threshold).astype(int)
        return predictions

    # OutlierMixin do Scikit-learn espera um método fit_predict, mas não é estritamente necessário
    # se você usar fit e depois predict.
    # Se quiser, pode adicionar:
    # def fit_predict(self, X, y=None, threshold=None):
    #     self.fit(X, y)
    #     return self.predict(X, threshold)