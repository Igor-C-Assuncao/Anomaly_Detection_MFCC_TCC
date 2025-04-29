import os
import pickle as pkl
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt  # Biblioteca para gráficos

def calculate_anomaly_threshold(train_losses, multiplier=1.5):
    """
    Calcula o limiar de anomalia usando o método IQR (Interquartile Range).
    Q3 + multiplier * (Q3 - Q1)
    """
    q1 = np.percentile(train_losses, 25)
    q3 = np.percentile(train_losses, 75)
    iqr = q3 - q1
    threshold = q3 + multiplier * iqr
    return threshold, q1, q3, iqr

def load_results(results_file):
    """
    Carrega os resultados de treino e teste de um arquivo .pkl.
    """
    if not os.path.exists(results_file):
        print(f"Erro: Arquivo {results_file} não encontrado.")
        return None

    with open(results_file, "rb") as f:
        results = pkl.load(f)
    return results

def plot_results(train_losses, test_losses, anomaly_threshold, anomalies):
    """
    Gera gráficos para visualizar os resultados de treino, teste e anomalias.
    """
    plt.figure(figsize=(12, 6))

    # Gráfico de perdas de treino
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Perdas de Treino", color="blue")
    plt.axhline(y=anomaly_threshold, color="red", linestyle="--", label="Limiar de Anomalia")
    plt.title("Perdas de Treino")
    plt.xlabel("Índice")
    plt.ylabel("Perda")
    plt.legend()

    # Gráfico de perdas de teste com anomalias destacadas
    plt.subplot(1, 2, 2)
    plt.plot(test_losses, label="Perdas de Teste", color="blue")
    plt.scatter(anomalies, [test_losses[i] for i in anomalies], color="red", label="Anomalias")
    plt.axhline(y=anomaly_threshold, color="red", linestyle="--", label="Limiar de Anomalia")
    plt.title("Perdas de Teste com Anomalias")
    plt.xlabel("Índice")
    plt.ylabel("Perda")
    plt.legend()

    plt.tight_layout()
    plt.show()

def present_results(train_losses, test_losses, anomaly_threshold, q1, q3, iqr):
    """
    Apresenta os resultados do treino e compara com os testes, incluindo a detecção de anomalias.
    """
    print("\n--- Resultados do Treinamento ---")
    print(f"Perda média no treino: {np.mean(train_losses):.6f}")
    print(f"Perda máxima no treino: {np.max(train_losses):.6f}")
    print(f"Perda mínima no treino: {np.min(train_losses):.6f}")

    print("\n--- Resultados do Teste ---")
    print(f"Perda média no teste: {np.mean(test_losses):.6f}")
    print(f"Perda máxima no teste: {np.max(test_losses):.6f}")
    print(f"Perda mínima no teste: {np.min(test_losses):.6f}")

    print("\n--- Detecção de Anomalias ---")
    anomalies = [i for i, loss in enumerate(test_losses) if loss > anomaly_threshold]
    print(f"Limiar de anomalia: {anomaly_threshold:.6f}")
    print(f"Número de anomalias detectadas: {len(anomalies)}")
    if anomalies:
        print(f"Índices das anomalias detectadas: {anomalies}")
    else:
        print("Nenhuma anomalia detectada.")

    # Criar tabela com os quartis e limiar
    table_data = [
        ["Q1 (25%)", f"{q1:.6f}"],
        ["Q3 (75%)", f"{q3:.6f}"],
        ["IQR (Q3 - Q1)", f"{iqr:.6f}"],
        ["Limiar de Anomalia", f"{anomaly_threshold:.6f}"]
    ]
    print("\n--- Estatísticas do Treinamento ---")
    print(tabulate(table_data, headers=["Métrica", "Valor"], tablefmt="grid"))

    # Criar tabela com os resultados de teste e classificação de anomalias
    test_table = [
        [i, loss, "Anômalo" if loss > anomaly_threshold else "Normal"]
        for i, loss in enumerate(test_losses)
    ]
    print("\n--- Resultados do Teste ---")
    print(tabulate(test_table, headers=["Índice", "Perda", "Classificação"], tablefmt="grid"))

    # Gerar gráficos
    plot_results(train_losses, test_losses, anomaly_threshold, anomalies)

def main():
    # Caminho para o arquivo de resultados gerado no treinamento
    results_file = "results/final_complete_losses_lstm_ae_all_feats_4_7_100_0.001.pkl" 
    # Carregar os resultados
    results = load_results(results_file)
    if results is None:
        return

    # Extrair perdas de treino e teste
    train_losses = results.get("train", [])
    test_losses = results.get("test", [])

    if not train_losses or not test_losses:
        print("Erro: Resultados de treino ou teste estão vazios.")
        return

    # Calcular limiar de anomalia
    anomaly_threshold, q1, q3, iqr = calculate_anomaly_threshold(train_losses)

    # Apresentar resultados
    present_results(train_losses, test_losses, anomaly_threshold, q1, q3, iqr)

if __name__ == "__main__":
    main()
