# Nome do arquivo: detect_anomalies_ae.py

import numpy as np
import pickle as pkl
import argparse
import os
from itertools import groupby
from operator import itemgetter
# Removido pandas e scipy por enquanto para simplificar,
# mas podem ser readicionados se necessário para análise de intervalos.

# --- Função para Calcular Limiar de Anomalia (Baseado no IQR) ---
def calculate_anomaly_threshold(train_errors_dist, multiplier=1.5):
    """
    Calcula o limiar de anomalia usando o método IQR (Interquartile Range).
    Q3 + multiplier * (Q3 - Q1)
    """
    if not isinstance(train_errors_dist, np.ndarray):
        train_errors_dist = np.array(train_errors_dist)

    # Ignora NaNs que podem ter vindo de erros de predição
    train_errors_dist = train_errors_dist[~np.isnan(train_errors_dist)]

    if len(train_errors_dist) < 2:
        print("Aviso: Dados de erro de treino insuficientes para calcular o limiar de forma confiável.")
        # Retorna infinito para não classificar nada como anômalo, ou pode definir um fallback.
        return np.inf

    q25, q75 = np.quantile(train_errors_dist, [0.25, 0.75])
    iqr = q75 - q25
    threshold = q75 + multiplier * iqr
    print(f"Limiar de Anomalia Calculado (Baseado nos Erros de Treino):")
    print(f"  Q1={q25:.6f}, Q3={q75:.6f}, IQR={iqr:.6f}")
    print(f"  Limiar = Q3 + {multiplier} * IQR = {threshold:.6f}")
    return threshold

# --- Filtro Suavizador (Passa-Baixa Simples / Média Móvel Exponencial) ---
def simple_lowpass_filter(arr, alpha):
    """Aplica um filtro passa-baixa simples (média móvel exponencial)."""
    if not isinstance(arr, (list, np.ndarray)) or len(arr) == 0:
        return [] # Retorna lista vazia se a entrada for inválida
    y = arr[0] # Inicializa com o primeiro valor
    filtered_arr = [y]
    for elem in arr[1:]:
        y = y + alpha * (float(elem) - y) # Garante que elem seja float
        filtered_arr.append(y)
    return filtered_arr

# --- Detecção de Sequências Anômalas Consecutivas ---
def detect_anomalous_sequences(scores, detection_threshold=0.8, consecutive_cycles=3):
    """
    Identifica sequências de ciclos consecutivos onde o score (após filtro)
    atinge ou excede o limiar de detecção.

    Args:
        scores (list or np.array): Lista de scores (0 a 1) após o filtro passa-baixa.
        detection_threshold (float): Limiar para considerar um ciclo como potencialmente anômalo (ex: 0.8).
        consecutive_cycles (int): Número mínimo de ciclos consecutivos acima do limiar
                                 para marcar uma sequência como anômala.

    Returns:
        list: Uma lista de sets, onde cada set contém os índices dos ciclos
              pertencentes a uma sequência anômala detectada.
    """
    if not isinstance(scores, np.ndarray):
        scores = np.array(scores) # Garante que é array numpy

    anomalous_sequence_list = []
    current_sequence = set()

    for i in range(len(scores)):
        if scores[i] >= detection_threshold:
            current_sequence.add(i) # Adiciona o índice do ciclo atual à sequência
        else:
            # Fim de uma sequência potencial (score caiu abaixo do limiar)
            if len(current_sequence) >= consecutive_cycles:
                # Se a sequência que acabou de terminar era longa o suficiente, salva
                anomalous_sequence_list.append(current_sequence)
            current_sequence = set() # Reseta para a próxima sequência

    # Verifica se a última sequência (que vai até o final dos dados) é longa o suficiente
    if len(current_sequence) >= consecutive_cycles:
        anomalous_sequence_list.append(current_sequence)

    return anomalous_sequence_list

# --- Impressão dos Resultados ---
def print_anomalous_indices(anomalous_sequences):
    """Imprime os índices das sequências anômalas detectadas."""
    if not anomalous_sequences:
        print("\nNenhuma sequência anômala detectada.")
        return

    print(f"\n--- {len(anomalous_sequences)} Sequência(s) Anômala(s) Detectada(s) ---")
    for i, sequence_set in enumerate(anomalous_sequences):
        sequence_list = sorted(list(sequence_set))
        # Imprime ranges para sequências longas, ou índices individuais para curtas
        if len(sequence_list) > 5:
            print(f"  Sequência {i+1}: Ciclos de índice {sequence_list[0]} a {sequence_list[-1]} (Total: {len(sequence_list)} ciclos)")
        else:
            print(f"  Sequência {i+1}: Ciclos de índice {sequence_list}")
    print("----------------------------------------------------")

# --- Função Principal ---
def main(args):

    # 1. Carregar Arquivo de Resultados (.pkl)
    print(f"Carregando resultados de: {args.results_file}")
    if not os.path.exists(args.results_file):
        print(f"Erro: Arquivo de resultados não encontrado em {args.results_file}")
        return
    try:
        with open(args.results_file, "rb") as f:
            results = pkl.load(f)
    except Exception as e:
        print(f"Erro ao carregar arquivo de resultados: {e}")
        return

    # Verifica estrutura básica - deve conter 'train' e 'test' com erros de reconstrução
    if not isinstance(results, dict) or 'train' not in results or 'test' not in results:
        print("Erro: Formato inválido do arquivo de resultados. Esperado dicionário com chaves 'train' e 'test'.")
        # Tenta verificar se o dicionário contém apenas 'test' e 'train' como listas (formato antigo)
        if isinstance(results, dict) and list(results.keys()) == ['test', 'train'] and isinstance(results['test'], list) and isinstance(results['train'], list) :
             print("Formato antigo detectado (dict com 'test'/'train' como listas). Continuando...")
             # Assume que as listas são os erros de reconstrução
        else:
             return

    # Extrai os erros - assume que são listas de floats/ints
    try:
        train_errors = np.array(results['train'], dtype=float)
        test_errors = np.array(results['test'], dtype=float)
    except Exception as e:
        print(f"Erro ao converter listas de erro para arrays numpy: {e}")
        return

    # Lida com possíveis NaNs (gerados se houve erro na predição daquele ciclo)
    original_test_len = len(test_errors)
    train_errors_cleaned = train_errors[~np.isnan(train_errors)]
    test_errors_cleaned = test_errors[~np.isnan(test_errors)]
    valid_test_indices = np.arange(original_test_len)[~np.isnan(test_errors)] # Índices originais dos erros válidos

    if len(train_errors_cleaned) == 0:
        print("Erro: Não foram encontrados erros de treino válidos no arquivo.")
        return
    if len(test_errors_cleaned) == 0:
        print("Erro: Não foram encontrados erros de teste válidos no arquivo.")
        return

    print(f"Carregados {len(train_errors_cleaned)} erros de treino válidos e {len(test_errors_cleaned)} erros de teste válidos (de {original_test_len} originais).")

    # 2. Calcular Limiar de Anomalia (usando apenas erros de treino)
    threshold = calculate_anomaly_threshold(train_errors_cleaned, multiplier=args.iqr_multiplier)

    # 3. Comparar Erros de Teste com o Limiar -> Saída Binária
    # Cria um array binário apenas para os erros válidos
    binary_output_cleaned = np.array(test_errors_cleaned > threshold, dtype=int)

    # Reconstrói o array binário com o tamanho original, marcando NaNs como não-anômalos (0)
    binary_output = np.zeros(original_test_len, dtype=int)
    if len(valid_test_indices) > 0: # Garante que há índices válidos antes de tentar atribuir
        binary_output[valid_test_indices] = binary_output_cleaned

    # 4. Aplicar Filtro Passa-Baixa (Suavização)
    print(f"Aplicando filtro passa-baixa (alpha={args.alpha})...")
    filtered_output = simple_lowpass_filter(binary_output.tolist(), args.alpha)  # Passa como lista

    # 5. Detectar Sequências Anômalas Consecutivas
    print(f"Detectando sequências: score >= {args.detection_threshold} por >= {args.consecutive} ciclos...")
    anomalous_sequences = detect_anomalous_sequences(
        filtered_output, # Usa a saída filtrada
        detection_threshold=args.detection_threshold,
        consecutive_cycles=args.consecutive
    )

    # 6. Reportar Resultados (Índices dos Ciclos Anômalos)
    print_anomalous_indices(anomalous_sequences)

    # --- Opcional: Salvar resultados detalhados ---
    if args.output_file:
         # Cria um nome de arquivo de saída mais descritivo
         output_filename = args.output_file
         if not output_filename.endswith(".pkl"):
             output_filename += ".pkl"
         output_path = os.path.join("results", output_filename) # Salva na pasta results

         detailed_results = {
             "source_results_file": args.results_file,
             "config_iqr_multiplier": args.iqr_multiplier,
             "config_alpha": args.alpha,
             "config_detection_threshold": args.detection_threshold,
             "config_consecutive": args.consecutive,
             "calculated_threshold": threshold,
             "test_errors": test_errors.tolist(), # Erros originais (com NaNs)
             "binary_output": binary_output.tolist(), # Saída binária antes do filtro
             "filtered_output": filtered_output, # Saída após filtro
             "anomalous_sequences_indices": [sorted(list(s)) for s in anomalous_sequences] # Lista de listas de índices
         }
         try:
             os.makedirs("results", exist_ok=True) # Garante que a pasta existe
             with open(output_path, "wb") as f_out:
                 pkl.dump(detailed_results, f_out)
             print(f"\nResultados detalhados da detecção salvos em: {output_path}")
         except Exception as e:
             print(f"\nErro ao salvar resultados detalhados: {e}")

# --- Configuração e Execução via Linha de Comando ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detectar anomalias a partir dos erros de reconstrução de um Autoencoder.")

    # Argumento obrigatório: arquivo de resultados do treino
    parser.add_argument("results_file", type=str,
                        help="Caminho para o arquivo .pkl gerado pelo script de treino, contendo os erros de reconstrução nas chaves 'train' e 'test'.")

    # Argumentos opcionais com valores padrão sensíveis
    parser.add_argument("--iqr_multiplier", type=float, default=3.0,
                        help="Multiplicador IQR para definir o limiar de anomalia (Padrão: 1.5 para outliers). Use 3.0 ou mais para 'extremos'.")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Valor de Alpha para o filtro passa-baixa (0 < alpha <= 1). Valores maiores reagem mais rápido. (Padrão: 0.1)")
    parser.add_argument("--detection_threshold", type=float, default=0.8,
                        help="Limiar (0-1) aplicado *após* o filtro para considerar um ciclo como potencialmente anômalo. (Padrão: 0.8)")
    parser.add_argument("--consecutive", type=int, default=3,
                        help="Número mínimo de ciclos consecutivos acima do limiar de detecção para registrar uma sequência como anômala. (Padrão: 3)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Nome do arquivo .pkl (opcional) para salvar resultados detalhados da detecção na pasta 'results/'.")

    args = parser.parse_args()

    # Validações básicas dos parâmetros
    if not (0 < args.alpha <= 1):
        parser.error("Alpha deve estar entre 0 (exclusivo) e 1 (inclusivo).")
    if not (0 <= args.detection_threshold <= 1): # Permite 0 se desejado
         parser.error("Limiar de detecção deve estar entre 0 e 1 (inclusivo).")
    if args.consecutive < 1:
         parser.error("Número de ciclos consecutivos deve ser pelo menos 1.")

    main(args)