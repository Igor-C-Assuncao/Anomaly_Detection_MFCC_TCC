import numpy as np
import os
import pickle as pkl
import matplotlib.pyplot as plt
from scipy.stats import zscore
import pandas as pd

def calculate_anomaly_threshold(train_errors, multiplier=3.0):
 
  q25, q75 = np.quantile(train_errors, [0.25, 0.75])
  return q75 + multiplier * (q75 - q25)

def detect_failures(anom_indices):
    failure_list = []
    failure = set()
    for i in range(len(anom_indices) - 1):
        if anom_indices[i] >= 0.99 and anom_indices[i + 1] >= 0.99:
            failure.add(i)
            failure.add(i + 1)
        elif len(failure) > 0:
            failure_list.append(failure)
            failure = set()
    if len(failure) > 0:
        failure_list.append(failure)
    return failure_list

def failure_list_to_interval(cycle_indices, failures):
    failure_intervals = []
    for failure in failures:
        failure = sorted(failure)
        start_cycle = cycle_indices[failure[0]]
        end_cycle = cycle_indices[failure[-1]]
        failure_intervals.append((start_cycle, end_cycle))
    return failure_intervals

def main(args):
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

    # Ajuste para acessar os dados gerados por train_cycles_adversarial
    train_losses = np.array(results['train']['reconstruction'], dtype=float)
    test_losses = np.array(results['test']['reconstruction'], dtype=float)
    train_critic = np.array(results['train']['critic'], dtype=float)
    test_critic = np.array(results['test']['critic'], dtype=float)

    # Calcular medianas
    median_train_losses = np.median(train_losses)
    median_test_losses = np.median(test_losses)
    median_train_critic = np.median(train_critic)
    median_test_critic = np.median(test_critic)

    # Calcular limiar de anomalia
    limiar = calculate_anomaly_threshold(train_losses, multiplier=args.iqr_multiplier)
    anomalies = np.where(test_losses > limiar)[0]
    print(f"Anomalias detectadas: {len(anomalies)} em {len(test_losses)}")
    print(f"Limiar de Anomalia: {limiar:.6f}")

    # Carregar ciclos diretamente como índices
    with open(args.train_data_path, "rb") as train_file:
        train_cycles = list(range(len(pkl.load(train_file))))
    with open(args.test_data_path, "rb") as test_file:
        test_cycles = list(range(len(pkl.load(test_file))))

    # Combinar critic scores e reconstruction losses
    combine_critic_reconstruction = []
    for x in range(1, len(test_critic) + 1):
        critic_slice = test_critic[:x]
        if len(critic_slice) > 1:  # Verifica se há elementos suficientes para calcular o zscore
            zscore_value = np.nan_to_num(zscore(critic_slice, ddof=1)[-1])
        else:
            zscore_value = 0  # Define um valor padrão se não for possível calcular o zscore
        combine_critic_reconstruction.append(np.abs(zscore_value) * test_losses[x - 1])
    combine_critic_reconstruction = np.array(combine_critic_reconstruction)

    # Detectar falhas
    anom_threshold = calculate_anomaly_threshold(combine_critic_reconstruction, multiplier=args.iqr_multiplier)
    binary_output = np.array(combine_critic_reconstruction > anom_threshold, dtype=int)
    failures = detect_failures(binary_output)

    # Converter falhas em intervalos de ciclos
    failure_intervals = failure_list_to_interval(test_cycles, failures)
    print("Intervalos de falhas detectados:")
    for interval in failure_intervals:
        print(f"Ciclo {interval[0]} até Ciclo {interval[1]}")

    # Plotar erros de reconstrução
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Treino Errors", color="blue")
    plt.axhline(y=limiar, color="red", linestyle="--", label="Limiar de Anomalia")
    plt.title("Treino Errors")
    plt.xlabel("Ciclo")
    plt.ylabel("Perda")
    plt.legend()

    # Plotar erros de teste
    plt.subplot(1, 2, 2)
    plt.plot(test_losses, label="Teste Errors", color="blue")
    plt.title("Teste Errors com Anomalias")
    plt.xlabel("Ciclo")
    plt.ylabel("Perda")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Anomaly Detection using WAE-GAN")
    parser.add_argument("--results_file", type=str, required=True, help="Path to the results file (.pkl)")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the train data file (.pkl)")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test data file (.pkl)")
    parser.add_argument("-iqr_multiplier", type=float, default=1.5, help="Multiplier for IQR to set the threshold")

    args = parser.parse_args()
    main(args)