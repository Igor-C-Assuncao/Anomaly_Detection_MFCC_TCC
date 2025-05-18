# test_wae_script.py
import torch
import numpy as np
import pickle # Para carregar os ciclos de MFCC do MIMII pré-processados

from mtsa.models.sound_anomaly_wae import SoundAnomalyWAE # Nossa classe principal

def load_mfcc_cycles_from_pkl(pkl_file_path):
    """ Carrega uma lista de ciclos de MFCC de um arquivo .pkl. """
    with open(pkl_file_path, 'rb') as f:
        mfcc_cycles = pickle.load(f)
    # Assegurar que cada ciclo é um numpy array
    return [np.array(cycle) for cycle in mfcc_cycles]

if __name__ == "__main__":
    print("Iniciando script de teste para SoundAnomalyWAE...")

    # --- Parâmetros de Exemplo ---
    # Pré-processamento e Dados
    SAMPLING_RATE = 16000
    NUM_MFCC_COEFFS = 20 # librosa.feature.mfcc default, mas Array2Mfcc precisa ser ajustado se diferente
    
    # O sequence_length precisa ser o número de frames que seu Array2Mfcc produz.
    # Vamos supor que seu preprocessing_mimii.py gere ciclos com um comprimento fixo de frames MFCC.
    # Por exemplo, se um ciclo de áudio de 10s a 16kHz com hop_length=256 resulta em ~625 frames.
    # O seu train_mfcc no notebook MTSA.ipynb tinha shape (712, 20, 313), então seq_len = 313.
    SEQUENCE_LENGTH = 313 # Ajuste este valor crucialmente!
    
    # WAEBaseModel
    EMBEDDING_DIM = 32
    TCN_CHANNELS = [32, 64] # Canais para encoder e decoder (pode ser diferente para cada)
    LSTM_HIDDEN_DIM = 64
    
    # Treinamento
    EPOCHS = 3 # Poucas épocas para teste rápido
    BATCH_SIZE = 16 # Batch size menor para teste
    N_CRITIC_STEPS = 2

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {DEVICE}")

    # --- Carregar Dados de Treinamento (Ciclos de MFCC pré-processados) ---
    # Substitua pelo caminho correto para SEU arquivo .pkl gerado por preprocessing_mimii.py
    # Este arquivo deve conter uma lista de numpy arrays, cada um sendo um ciclo MFCC.
    # Ex: (num_coeffs, seq_len)
    try:
        # Tentar carregar os dados de treinamento pré-processados
        # Certifique-se que este arquivo existe e contém dados no formato esperado
        # (lista de numpy arrays, cada um com shape (NUM_MFCC_COEFFS, SEQUENCE_LENGTH))
        train_mfcc_cycles_path = "Data/preprocessed_mimii/slider_id_00_train_cycles.pkl" # EXEMPLO DE CAMINHO
        print(f"Carregando ciclos de MFCC de: {train_mfcc_cycles_path}")
        # Seu preprocessing_mimii.py parece salvar os ciclos em um formato um pouco diferente
        # pkl.dump(train_cycles, f) onde train_cycles = generate_cycles(train_mfcc)
        # e generate_cycles faz: cycles.append(data[i:i + 1, : , : ])
        # Isso significa que cada "ciclo" no pkl é na verdade (1, num_coeffs, seq_len)
        # Precisamos ajustar o carregamento ou o WAEData para lidar com isso.
        
        raw_loaded_cycles = load_mfcc_cycles_from_pkl(train_mfcc_cycles_path)
        if not raw_loaded_cycles:
            raise ValueError("Arquivo PKL de treino está vazio ou não foi carregado.")

        # Ajustar o formato se necessário: remover a dimensão extra '1'
        # Se cada ciclo no PKL é (1, NUM_MFCC_COEFFS, SEQUENCE_LENGTH)
        train_mfcc_data = [cycle.squeeze(0) for cycle in raw_loaded_cycles if cycle.shape[0] == 1]
        print(f"Carregados {len(train_mfcc_data)} ciclos de treino.")

        if not train_mfcc_data:
            print("Nenhum ciclo de treino carregado. Verifique o formato dos dados no PKL.")
            exit()
            
        # Verificar as dimensões do primeiro ciclo carregado
        print(f"Shape do primeiro ciclo de treino carregado (após squeeze): {train_mfcc_data[0].shape}")
        if train_mfcc_data[0].shape[0] != NUM_MFCC_COEFFS or train_mfcc_data[0].shape[1] != SEQUENCE_LENGTH:
            print(f"ALERTA: Dimensões do MFCC carregado ({train_mfcc_data[0].shape}) não correspondem "
                  f"aos parâmetros esperados (coeffs: {NUM_MFCC_COEFFS}, seq_len: {SEQUENCE_LENGTH}).")
            print("Por favor, ajuste NUM_MFCC_COEFFS e SEQUENCE_LENGTH no script de teste ou verifique seu pré-processamento.")
            # Você pode querer que o script pare aqui se as dimensões não baterem.
            # exit()

    except FileNotFoundError:
        print(f"Arquivo de dados de treino não encontrado: {train_mfcc_cycles_path}")
        print("Gerando dados dummy para teste...")
        # Gerar dados dummy se o arquivo não for encontrado (APENAS PARA TESTE DE ESTRUTURA)
        num_dummy_samples = 50
        train_mfcc_data = [
            np.random.rand(NUM_MFCC_COEFFS, SEQUENCE_LENGTH).astype(np.float32)
            for _ in range(num_dummy_samples)
        ]
        print(f"Gerados {len(train_mfcc_data)} ciclos de treino dummy.")


    # --- Instanciar o Modelo SoundAnomalyWAE ---
    print("Instanciando SoundAnomalyWAE...")
    wae_detector = SoundAnomalyWAE(
        sampling_rate=SAMPLING_RATE,
        num_mfcc_coeffs=NUM_MFCC_COEFFS,
        # mfcc_fft_window_size, mfcc_hop_length (se Array2Mfcc for modificado para usá-los)
        
        sequence_length=SEQUENCE_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        tcn_encoder_channels=TCN_CHANNELS,
        tcn_decoder_channels=TCN_CHANNELS, # Pode ser diferente
        lstm_discriminator_hidden_dim=LSTM_HIDDEN_DIM,
        
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        n_critic_steps=N_CRITIC_STEPS,
        device=DEVICE
    )
    print("Modelo instanciado.")

    # --- Treinar o Modelo ---
    print("Iniciando treinamento (fit)...")
    wae_detector.fit(train_mfcc_data) # Passa a lista de ciclos de MFCC
    print("Treinamento concluído.")

    # --- Testar score_samples (com dados de treino por enquanto) ---
    print("Testando score_samples...")
    # Para um teste real, você usaria dados de teste aqui.
    # E também passaria a média e std real do discriminador calculadas no treino.
    # O método fit já calcula e armazena self.mean_disc_score_train_ e self.std_disc_score_train_
    
    # Usar um subconjunto dos dados de treino para testar o score_samples
    test_data_subset = train_mfcc_data[:max(5, BATCH_SIZE)] # Pegar algumas amostras para teste rápido
    
    anomaly_scores = wae_detector.score_samples(test_data_subset)
    print(f"Pontuações de anomalia para {len(test_data_subset)} amostras de teste (usando dados de treino):")
    print(anomaly_scores)
    
    # --- Testar predict (exemplo) ---
    # Primeiro, precisamos de um limiar. Vamos calcular um limiar simples dos scores de treino.
    # Em um cenário real, você faria isso com mais cuidado.
    if hasattr(wae_detector, 'mean_disc_score_train_'): # Verifica se o fit completou essa parte
        print("Calculando scores nos dados de treino para definir um limiar de exemplo...")
        training_scores_for_threshold = wae_detector.score_samples(train_mfcc_data)
        
        if training_scores_for_threshold.size > 0:
            # Exemplo de limiar: média + 2 * desvio padrão dos scores de treino
            # (assumindo que scores mais altos são mais anômalos)
            example_threshold = np.mean(training_scores_for_threshold) + 2 * np.std(training_scores_for_threshold)
            print(f"Limiar de exemplo definido como: {example_threshold:.4f}")

            print("Testando predict...")
            predictions = wae_detector.predict(test_data_subset, threshold=example_threshold)
            print(f"Predições para {len(test_data_subset)} amostras de teste:")
            print(predictions)
        else:
            print("Não foi possível calcular scores de treino para definir o limiar.")
    else:
        print("Atributos de score do discriminador não encontrados; pulando teste de predict.")

    print("Script de teste concluído.")