# WAE/train_cycles_adversarial.py

import os
import numpy as np
import torch as th
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import pickle as pkl
import tqdm # Certifique-se de que tqdm está instalado: pip install tqdm
import logging # Para melhor feedback

# Importar os modelos e o parser de argumentos
# Ajuste o caminho se a estrutura do seu projeto for diferente
# Ex: from .ArgumentParser import parse_arguments
# Ex: from .models.LSTM_AAE import Encoder, Decoder, SimpleDiscriminator, LSTMDiscriminator, ConvDiscriminator
# Ou se estiverem no mesmo nível:
from ArgumentParser import parse_arguments
from models.LSTM_AAE import Encoder, Decoder, SimpleDiscriminator, LSTMDiscriminator, ConvDiscriminator
# Se estiver usando TCN, importe de models.TCN_AAE
# from models.TCN_AAE import Encoder_TCN, Decoder_TCN, ConvDiscriminator_TCN # Exemplo

# --- Configuração do Logging ---
# Coloque no início do seu script ou na função principal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# --- Constante N_MFCC (deve ser consistente com generate_cv_folds_data.py) ---
# Idealmente, isso viria de uma configuração ou seria um argumento.
# Se args.FEATS = "mimii_mfcc" é usado, NUMBER_FEATURES é definido em load_parameters
N_MFCC = 20 # Usado para FEATS_TO_NUMBER

# --- Funções de Treinamento (semelhantes às suas originais) ---

def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

def train_discriminator(optimizer_discriminator, train_tensors_fold, multivariate_normal, epoch, args):
    frozen_params(args.encoder)
    frozen_params(args.decoder)
    free_params(args.discriminator)

    losses = []
    # A barra de progresso original usava "unit=cycles", o que pode ser confuso se os tensores são batches.
    # Vamos usar unit="batch" se train_tensors_fold for um DataLoader, ou "cycle" se for uma lista de ciclos individuais.
    # Seus scripts parecem iterar sobre uma lista de ciclos, processando um por vez.
    unit_desc = "cycle" if isinstance(train_tensors_fold, list) else "batch"

    with tqdm.tqdm(train_tensors_fold, unit=unit_desc, disable=args.disable_tqdm) as tqdm_epoch:
        for train_tensor_cycle in tqdm_epoch:
            tqdm_epoch.set_description(f"Fold {args.fold_id} - Disc Epoch {epoch + 1}")
            optimizer_discriminator.zero_grad()

            # train_tensor_cycle já deve estar no device correto
            # train_tensor_cycle = train_tensor_cycle.to(args.device) # Se não estiver

            real_latent_space = args.encoder(train_tensor_cycle) # (batch_size=1, embedding_dim)
            
            # Ajustar o sample do MVN para o shape do latent space do WAE (que não tem dimensão de batch no meio como AAE)
            # O real_latent_space para WAE (LSTM_AAE) é (1, embedding_dim)
            # Para o AAE original do repo schelotto, o encoder retorna (batch_size, seq_len, embedding_dim)
            # e eles pegam o último hidden state, resultando em (batch_size, embedding_dim)
            # Seus modelos LSTM_AAE.Encoder retornam hidden[-1], que é (num_layers * num_directions, batch_size, hidden_size)
            # e você pega hidden[-1] que seria (batch_size, hidden_size) - o que está correto.
            # O batch_size aqui é 1 porque você itera sobre os ciclos.
            random_latent_space = multivariate_normal.sample(sample_shape=real_latent_space.shape[:1]).to(args.device) # sample_shape=(1,)

            discriminator_real = args.discriminator(real_latent_space)
            discriminator_random = args.discriminator(random_latent_space)

            loss_random_term = th.log(discriminator_random + 1e-9) # Adicionar epsilon para estabilidade
            loss_real_term = th.log(1 - discriminator_real + 1e-9) # Adicionar epsilon

            loss = args.WAE_regularization_term * -th.mean(loss_real_term + loss_random_term)
            loss.backward()

            nn.utils.clip_grad_norm_(args.discriminator.parameters(), args.clip_norm if hasattr(args, 'clip_norm') else 1.0)
            optimizer_discriminator.step()
            losses.append(loss.item())
    return np.mean(losses) if losses else 0.0


def train_reconstruction(optimizer_encoder, optimizer_decoder, train_tensors_fold, epoch, args):
    free_params(args.encoder)
    free_params(args.decoder)
    frozen_params(args.discriminator)

    losses = []
    unit_desc = "cycle" if isinstance(train_tensors_fold, list) else "batch"

    with tqdm.tqdm(train_tensors_fold, unit=unit_desc, disable=args.disable_tqdm) as tqdm_epoch:
        for train_tensor_cycle in tqdm_epoch:
            tqdm_epoch.set_description(f"Fold {args.fold_id} - Enc/Dec Epoch {epoch + 1}")
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()

            # train_tensor_cycle = train_tensor_cycle.to(args.device) # Se não estiver

            real_latent_space = args.encoder(train_tensor_cycle) # (1, embedding_dim)

            # Para o decoder LSTM (bidirecional no seu LSTM_AAE), ele espera (batch, seq_len, input_size)
            # O input_size é embedding_dim. seq_len é o número de "passos" para gerar.
            # O train_tensor_cycle original é (1, seq_len_original, n_features)
            # O stacked_LV deve ser (1, seq_len_original, embedding_dim)
            seq_len_original = train_tensor_cycle.shape[1]
            stacked_lv = real_latent_space.unsqueeze(1).repeat(1, seq_len_original, 1) # (1, seq_len_original, embedding_dim)

            reconstructed_input = args.decoder(stacked_lv) # (1, seq_len_original, n_features)
            discriminator_real_latent = args.discriminator(real_latent_space) # (1, 1)

            reconstruction_loss = F.mse_loss(reconstructed_input, train_tensor_cycle)
            # A perda do discriminador para o gerador/encoder é -log(D(G(z))) no GAN original.
            # No WAE, o encoder tenta fazer D(E(x)) ser alto (próximo de 1, se D retorna prob de ser real).
            # Ou D(E(x)) ser baixo (próximo de 0, se D retorna prob de ser fake/encoded).
            # Se o seu D retorna P(real), então o encoder quer maximizar log(D(E(x))).
            # A perda do encoder é -log(D(E(x))).
            # A perda do discriminador é -(log(D(z_prior)) + log(1-D(E(x))))
            # No seu código original: loss = args.WAE_regularization_term * -th.mean(loss_real_term + loss_random_term)
            # Onde loss_real_term = th.log(1-discriminator_real)
            # E para o encoder: loss = th.mean(reconstruction_loss - discriminator_loss)
            # onde discriminator_loss = args.WAE_regularization_term * (th.log(discriminator_real_latent))
            # Isso significa que o encoder quer maximizar log(discriminator_real_latent), o que está correto.
            discriminator_loss_for_encoder = args.WAE_regularization_term * (th.log(discriminator_real_latent + 1e-9))

            loss = reconstruction_loss - discriminator_loss_for_encoder # Queremos minimizar esta perda total
            # Se for um batch de ciclos (batch_size > 1), usar th.mean(loss)
            loss = loss.mean() # Se train_tensor_cycle é um batch, ou se é um único ciclo, .mean() não muda nada.

            loss.backward()
            nn.utils.clip_grad_norm_(args.encoder.parameters(), args.clip_norm if hasattr(args, 'clip_norm') else 1.0)
            nn.utils.clip_grad_norm_(args.decoder.parameters(), args.clip_norm if hasattr(args, 'clip_norm') else 1.0)
            optimizer_encoder.step()
            optimizer_decoder.step()
            losses.append(loss.item())
    return np.mean(losses) if losses else 0.0


def train_model(train_tensors_fold, epochs, args):
    optimizer_discriminator = optim.Adam(args.discriminator.parameters(), lr=args.disc_lr, weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 0)
    optimizer_encoder = optim.Adam(args.encoder.parameters(), lr=args.LR, weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 0)
    optimizer_decoder = optim.Adam(args.decoder.parameters(), lr=args.LR, weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 0)

    loss_over_time = {"discriminator": [], "encoder_decoder": []} # Renomeado para clareza
    
    # Usar o mean e covariance do embedding_dim
    multivariate_normal = MultivariateNormal(th.zeros(args.EMBEDDING).to(args.device), 
                                             th.eye(args.EMBEDDING).to(args.device))


    for epoch in range(epochs):
        # Colocar modelos em modo treino
        args.encoder.train()
        args.decoder.train()
        args.discriminator.train()

        mean_discriminator_loss = train_discriminator(optimizer_discriminator, train_tensors_fold,
                                                      multivariate_normal, epoch, args)
        mean_encoder_decoder_loss = train_reconstruction(optimizer_encoder, optimizer_decoder,
                                                         train_tensors_fold, epoch, args)

        loss_over_time['discriminator'].append(mean_discriminator_loss)
        loss_over_time['encoder_decoder'].append(mean_encoder_decoder_loss)

        logging.info(f'Fold {args.fold_id} - Epoch {epoch + 1}/{epochs}: Disc Loss {mean_discriminator_loss:.4f} | Enc/Dec Loss {mean_encoder_decoder_loss:.4f}')
    return loss_over_time


def predict_scores(args_models, data_tensors, description_tqdm):
    """Calcula scores de reconstrução e do crítico para os dados fornecidos."""
    reconstruction_errors = []
    critic_scores = []
    
    # args_models deve ser o objeto 'args' que contém encoder, decoder, discriminator e device
    args_models.encoder.eval()
    args_models.decoder.eval()
    args_models.discriminator.eval()

    with th.no_grad():
        unit_desc = "cycle" if isinstance(data_tensors, list) else "batch"
        with tqdm.tqdm(data_tensors, unit=unit_desc, desc=description_tqdm, disable=args_models.disable_tqdm) as tqdm_data:
            for data_tensor_cycle in tqdm_data:
                # data_tensor_cycle = data_tensor_cycle.to(args_models.device) # Se não estiver

                latent_vector = args_models.encoder(data_tensor_cycle) # (1, embedding_dim)
                
                seq_len_original = data_tensor_cycle.shape[1]
                stacked_lv = latent_vector.unsqueeze(1).repeat(1, seq_len_original, 1) # (1, seq_len_original, embedding_dim)

                reconstruction = args_models.decoder(stacked_lv) # (1, seq_len_original, n_features)
                
                # MSE por ciclo
                # F.mse_loss calcula a média sobre todos os elementos por padrão.
                # Para erro por ciclo, se data_tensor_cycle é (1, seq, feat), a média já é por ciclo.
                # Se fosse um batch, F.mse_loss(reconstruction, data_tensor_cycle, reduction='none').mean(dim=(1,2))
                reconstruction_errors.append(F.mse_loss(reconstruction, data_tensor_cycle).item())
                
                # Score do crítico por ciclo
                critic_out = args_models.discriminator(latent_vector) # (1,1)
                critic_scores.append(critic_out.mean().item()) # .mean() para converter (1,1) para escalar, se necessário
                
    return reconstruction_errors, critic_scores


# --- Funções de Orquestração (Modificadas para CV) ---

def configure_paths_and_models(args_parsed):
    """Configura caminhos, nomes de arquivos e instancia modelos com base nos argumentos."""
    
    # FEATS_TO_NUMBER: Adapte se necessário, especialmente para "mimii_mfcc"
    FEATS_TO_NUMBER = {
        "analog_feats": 8, "digital_feats": 8, "all_feats": 16,
        "noflow_feats": 7, "mimii_mfcc": getattr(args_parsed, 'N_MFCC', N_MFCC) # Usa N_MFCC global se não for arg
    }
    args_parsed.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    logging.info(f"Usando dispositivo: {args_parsed.device}")

    if args_parsed.FEATS not in FEATS_TO_NUMBER:
        raise ValueError(f"Valor de FEATS desconhecido: {args_parsed.FEATS}. Valores esperados: {list(FEATS_TO_NUMBER.keys())}")
    args_parsed.NUMBER_FEATURES = FEATS_TO_NUMBER[args_parsed.FEATS]

    # Validação de argumentos essenciais para CV
    if args_parsed.fold_id is None:
        raise ValueError("'-fold_id' é um argumento obrigatório para o modo de validação cruzada.")
    if not hasattr(args_parsed, 'machine_type') or not args_parsed.machine_type:
        raise ValueError("'-machine_type' é um argumento obrigatório.")
    if not hasattr(args_parsed, 'machine_id') or not args_parsed.machine_id:
        raise ValueError("'-machine_id' é um argumento obrigatório.")

    # Cria diretório de resultados base para CV se não existir
    os.makedirs(args_parsed.cv_results_base_dir, exist_ok=True)
    
    machine_identifier = f"{args_parsed.machine_type}_{args_parsed.machine_id}"
    
    # String base do modelo para nomes de arquivos (mais limpa)
    # Ex: LSTMDiscriminator_slider_id_00_mimii_mfcc_emb4_layers2_reg1.0
    model_config_name = (
        f"{args_parsed.MODEL_NAME}_{machine_identifier}_{args_parsed.FEATS}"
        f"_emb{args_parsed.EMBEDDING}_L{args_parsed.LSTM_LAYERS}"
        f"_Wreg{args_parsed.WAE_regularization_term}"
        f"_Dr{args_parsed.DROPOUT}_Dh{args_parsed.disc_hidden}_Dl{args_parsed.disc_layers}" # Params do Discriminador
        # Adicionar params de TCN se args.ENCODER_NAME ou args.DECODER_NAME for TCN
    )
    if getattr(args_parsed, 'ENCODER_NAME', 'LSTM') == 'TCN' or getattr(args_parsed, 'DECODER_NAME', 'LSTM') == 'TCN':
        model_config_name += f"_TCNk{args_parsed.tcn_kernel}_TCNh{args_parsed.tcn_hidden}_TCNl{args_parsed.tcn_layers}"


    # Diretório de resultados para o fold atual
    args_parsed.current_fold_results_dir = os.path.join(
        args_parsed.cv_results_base_dir,
        args_parsed.machine_type,
        args_parsed.machine_id,
        f"fold_{args_parsed.fold_id}"
    )
    os.makedirs(args_parsed.current_fold_results_dir, exist_ok=True)

    # Nomes dos arquivos de saída
    file_prefix = f"final_{model_config_name}_ep{args_parsed.EPOCHS}_lr{args_parsed.LR}_dlr{args_parsed.disc_lr}"

    args_parsed.model_saving_path_template = lambda model_type_suffix: \
        os.path.join(args_parsed.current_fold_results_dir, f"{file_prefix}_MODEL_{model_type_suffix}.pt")

    args_parsed.training_loss_saving_path = \
        os.path.join(args_parsed.current_fold_results_dir, f"{file_prefix}_training_losses.pkl")

    args_parsed.training_fold_scores_saving_path = \
        os.path.join(args_parsed.current_fold_results_dir, f"{file_prefix}_training_fold_scores.pkl")
    
    args_parsed.test_fold_scores_saving_path = \
        os.path.join(args_parsed.current_fold_results_dir, f"{file_prefix}_test_fold_scores.pkl")


    # Carregamento dos dados do fold
    fold_data_path = os.path.join(args_parsed.cv_data_base_dir,
                                  args_parsed.machine_type,
                                  args_parsed.machine_id,
                                  f"fold_{args_parsed.fold_id}")
    
    train_cycles_path = os.path.join(fold_data_path, f"train_cycles_fold{args_parsed.fold_id}.pkl")
    test_cycles_path = os.path.join(fold_data_path, f"test_cycles_fold{args_parsed.fold_id}.pkl")

    logging.info(f"Carregando dados de treino do Fold {args_parsed.fold_id} de: {train_cycles_path}")
    if not os.path.exists(train_cycles_path): raise FileNotFoundError(f"Arquivo de treino não encontrado: {train_cycles_path}")
    with open(train_cycles_path, "rb") as f:
        train_cycles_list_np = pkl.load(f)
        args_parsed.train_tensors_fold = [th.tensor(cycle, dtype=th.float32).to(args_parsed.device) for cycle in train_cycles_list_np if cycle.ndim == 2 and cycle.shape[0] > 0]
    if not args_parsed.train_tensors_fold: raise ValueError(f"Nenhum ciclo de treino válido carregado de {train_cycles_path}")


    logging.info(f"Carregando dados de teste do Fold {args_parsed.fold_id} de: {test_cycles_path}")
    if not os.path.exists(test_cycles_path): raise FileNotFoundError(f"Arquivo de teste não encontrado: {test_cycles_path}")
    with open(test_cycles_path, "rb") as f:
        test_cycles_list_np = pkl.load(f)
        args_parsed.test_tensors_fold = [th.tensor(cycle, dtype=th.float32).to(args_parsed.device) for cycle in test_cycles_list_np if cycle.ndim == 2 and cycle.shape[0] > 0]
    if not args_parsed.test_tensors_fold: logging.warning(f"Nenhum ciclo de teste válido carregado de {test_cycles_path}")


    # Instanciação dos Modelos (Encoder, Decoder, Discriminator)
    # TODO: Adicionar lógica para escolher entre LSTM e TCN para Encoder/Decoder baseado em args
    # Por enquanto, assume LSTM_AAE
    EncoderClass = Encoder
    DecoderClass = Decoder
    # if getattr(args_parsed, 'ENCODER_NAME', 'LSTM') == 'TCN': EncoderClass = Encoder_TCN
    # if getattr(args_parsed, 'DECODER_NAME', 'LSTM') == 'TCN': DecoderClass = Decoder_TCN
        
    args_parsed.encoder = EncoderClass(args_parsed.NUMBER_FEATURES, args_parsed.EMBEDDING, args_parsed.DROPOUT, args_parsed.LSTM_LAYERS,
                                   hidden_dim=getattr(args_parsed, 'tcn_hidden', 30), kernel_size=getattr(args_parsed, 'tcn_kernel', 5)
                                   ).to(args_parsed.device)
    args_parsed.decoder = DecoderClass(args_parsed.EMBEDDING, args_parsed.NUMBER_FEATURES, args_parsed.DROPOUT, args_parsed.LSTM_LAYERS,
                                   hidden_dim=getattr(args_parsed, 'tcn_hidden', 30), kernel_size=getattr(args_parsed, 'tcn_kernel', 5)
                                   ).to(args_parsed.device)

    discriminator_models_map = {
        "SimpleDiscriminator": SimpleDiscriminator,
        "LSTMDiscriminator": LSTMDiscriminator,
        "ConvDiscriminator": ConvDiscriminator,
        # "SimpleDiscriminator_TCN": SimpleDiscriminator_TCN, # Descomente se usar
        # "ConvDiscriminator_TCN": ConvDiscriminator_TCN
    }
    if args_parsed.MODEL_NAME not in discriminator_models_map:
        raise ValueError(f"Modelo de discriminador desconhecido: {args_parsed.MODEL_NAME}. Disponíveis: {list(discriminator_models_map.keys())}")
    
    DiscriminatorClass = discriminator_models_map[args_parsed.MODEL_NAME]
    args_parsed.discriminator = DiscriminatorClass(
        args_parsed.EMBEDDING, args_parsed.DROPOUT,
        n_layers=args_parsed.disc_layers,
        disc_hidden=args_parsed.disc_hidden,
        kernel_size=getattr(args_parsed, 'tcn_kernel', 5), # Para ConvDiscriminator
        window_size=args_parsed.train_tensors_fold[0].shape[0] # Para ConvDiscriminator_TCN, se usado
    ).to(args_parsed.device)
    
    args_parsed.disable_tqdm = getattr(args_parsed, 'disable_tqdm', False) # Para desabilitar tqdm se necessário
    args_parsed.clip_norm = getattr(args_parsed, 'clip_norm', 1.0) # Valor de clip para gradientes

    logging.info(f"Configuração para Fold {args_parsed.fold_id} ({machine_identifier}) carregada. Modelo: {model_config_name}")
    return args_parsed


def run_training_fold(args):
    """Treina o modelo para o fold configurado em args."""
    logging.info(f"Iniciando treinamento offline para Fold {args.fold_id}...")
    
    loss_over_time = train_model(
        args.train_tensors_fold,
        epochs=args.EPOCHS,
        args=args # Passa o objeto args completo que contém os modelos e outros params
    )

    with open(args.training_loss_saving_path, "wb") as f:
        pkl.dump(loss_over_time, f)
    logging.info(f"Perdas de treinamento do Fold {args.fold_id} salvas em {args.training_loss_saving_path}")

    th.save(args.encoder.state_dict(), args.model_saving_path_template("encoder"))
    th.save(args.decoder.state_dict(), args.model_saving_path_template("decoder"))
    th.save(args.discriminator.state_dict(), args.model_saving_path_template("discriminator"))
    logging.info(f"Modelos do Fold {args.fold_id} salvos.")


def run_scoring_fold(args):
    """Calcula e salva os scores de treino e teste para o fold configurado em args."""
    # Carregar modelos se não estiverem já em args (ex: se rodando scoring separadamente)
    if not (hasattr(args, 'encoder') and hasattr(args, 'decoder') and hasattr(args, 'discriminator')):
        raise ValueError("Modelos não encontrados em args. Carregue ou treine os modelos primeiro.")
    
    logging.info(f"Calculando scores de TREINO para o Fold {args.fold_id}...")
    train_re_scores, train_crit_scores = predict_scores(args, args.train_tensors_fold, f"Scoring Treino Fold {args.fold_id}")
    training_fold_scores_data = {'reconstruction': train_re_scores, 'critic': train_crit_scores}
    with open(args.training_fold_scores_saving_path, "wb") as f:
        pkl.dump(training_fold_scores_data, f)
    logging.info(f"Scores de treinamento do Fold {args.fold_id} salvos em {args.training_fold_scores_saving_path}")

    if args.test_tensors_fold: # Só calcula scores de teste se houver dados de teste
        logging.info(f"Calculando scores de TESTE para o Fold {args.fold_id}...")
        test_re_scores, test_crit_scores = predict_scores(args, args.test_tensors_fold, f"Scoring Teste Fold {args.fold_id}")
        test_fold_scores_data = {
            'reconstruction': test_re_scores,
            'critic': test_crit_scores,
            'train_scores_path': args.training_fold_scores_saving_path # Referência
        }
        with open(args.test_fold_scores_saving_path, "wb") as f:
            pkl.dump(test_fold_scores_data, f)
        logging.info(f"Scores de teste do Fold {args.fold_id} salvos em {args.test_fold_scores_saving_path}")
    else:
        logging.warning(f"Nenhum dado de teste para o Fold {args.fold_id}. Scores de teste não calculados.")


# --- Script Principal de Execução ---
if __name__ == "__main__":
    raw_cli_args = parse_arguments()
    
    try:
        # Configura caminhos, carrega dados do fold, instancia modelos
        args = configure_paths_and_models(raw_cli_args)

        # Verifica se o modelo já foi treinado para este fold
        encoder_path = args.model_saving_path_template("encoder")
        model_already_exists = os.path.exists(encoder_path)

        if model_already_exists and not args.force_training:
            logging.info(f"Modelos para Fold {args.fold_id} ({args.machine_type}/{args.machine_id}) já existem e '-force_training' não está ativo. Carregando...")
            args.encoder.load_state_dict(th.load(args.model_saving_path_template("encoder"), map_location=args.device))
            args.decoder.load_state_dict(th.load(args.model_saving_path_template("decoder"), map_location=args.device))
            args.discriminator.load_state_dict(th.load(args.model_saving_path_template("discriminator"), map_location=args.device))
        else:
            if model_already_exists and args.force_training:
                logging.info(f"'-force_training' ativo. Retreinando modelos para Fold {args.fold_id} ({args.machine_type}/{args.machine_id})...")
            elif not model_already_exists:
                logging.info(f"Nenhum modelo pré-treinado encontrado para Fold {args.fold_id} ({args.machine_type}/{args.machine_id}). Iniciando treinamento...")
            run_training_fold(args)

        # Calcular e salvar scores (treino e teste) para o fold atual
        run_scoring_fold(args)

        logging.info(f"Processamento do Fold {args.fold_id} para {args.machine_type}/{args.machine_id} CONCLUÍDO.")

    except FileNotFoundError as e:
        logging.error(f"Erro de arquivo não encontrado: {e}")
        logging.error("Verifique os caminhos em -cv_data_base_dir e se os dados dos folds foram gerados corretamente.")
    except ValueError as e:
        logging.error(f"Erro de valor: {e}")
    except Exception as e:
        logging.error(f"Ocorreu um erro inesperado durante o processamento do fold {getattr(raw_cli_args, 'fold_id', 'N/A')}: {e}", exc_info=True)