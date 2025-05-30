import numpy as np
import os
import pickle as pkl
import matplotlib.pyplot as plt
from scipy.stats import zscore
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def calculate_anomaly_threshold(train_errors, multiplier=1.5):
 
  q25, q75 = np.quantile(train_errors, [0.25, 0.75])
  return q75 + multiplier * (q75 - q25)





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

    train_losses = np.array(results['train'], dtype=float)
    test_losses = np.array(results['test'], dtype=float)
    # train_critic = np.array(results['train']['critic'], dtype=float)
    # test_critic = np.array(results['test']['critic'], dtype=float)
 

   

    with open(args.test_labels_path, "rb") as labels_file:
        test_labels = pkl.load(labels_file)

 

    limiar = calculate_anomaly_threshold(train_losses, multiplier=args.iqr_multiplier)
   
    anomalies = np.where(test_losses > limiar)[0]
    truth_anomalies = np.where(np.array(test_labels) == 1)[0]
    

    
    print(f"Anomalias detectadas: {len(anomalies)} em {len(truth_anomalies)} reais")
    print(f"Limiar de Anomalia: {limiar:.6f}")
    

    


    test_preds = (test_losses > limiar).astype(int)


    # Calcular métricas
    roc_auc = roc_auc_score(test_labels, test_preds)
    acc = accuracy_score(test_labels, test_preds)
    prec = precision_score(test_labels, test_preds, zero_division=0)
    rec = recall_score(test_labels, test_preds, zero_division=0)
    f1 = f1_score(test_labels, test_preds, zero_division=0)

    plt.figure(figsize=(12, 4))
    plt.plot(test_labels, label="Rótulos Reais", marker='o', linestyle='-', alpha=0.7)
    plt.plot(test_preds, label="Predições", marker='x', linestyle='--', alpha=0.7)
    plt.title("Comparação: Rótulos Reais vs Predições")
    plt.xlabel("Índice da Amostra")
    plt.ylabel("Classe")
    plt.legend()
    plt.tight_layout()
    plt.savefig("labels_vs_preds.png")
    plt.close()

    

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Acurácia: {acc:.4f}")
    print(f"Precisão: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Salvar métricas para uso posterior
    import re
    match = re.search(r'results[\\/](.+?)_(.+?)_Fold(\d+)\.pkl', args.results_file)
    if match:
        machine_type, machine_id, fold = match.group(1), match.group(2), match.group(3)
        metrics_file = f"results/{machine_type}_{machine_id}_Fold{fold}_metrics.pkl"
        metrics_dict = {
            "roc_auc": roc_auc,
            "f1": f1,
            "accuracy": acc,
            "precision": prec,
            "recall": rec
        }
        with open(metrics_file, "wb") as mf:
            pkl.dump(metrics_dict, mf)

    # Plotar erros de reconstrução
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Treino Errors", color="blue")
    plt.axhline(y=limiar, color="red", linestyle="--", label="Limiar de Anomalia")
    plt.title("Treino Errors")
    plt.xlabel("Ciclo")
    plt.ylabel("Perda")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_losses, label="Teste Errors", color="blue")
    plt.axhline(y=limiar, color="red", linestyle="--", label="Limiar de Anomalia")
    plt.title("Teste Errors com Anomalias")
    plt.xlabel("Ciclo")
    plt.ylabel("Perda")
    plt.legend()

    # plt.tight_layout()
    # plt.show()
    plt.savefig("my_plot.png")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Anomaly Detection using WAE-GAN")
    parser.add_argument("--results_file", type=str, required=True, help="Path to the results file (.pkl)")
    parser.add_argument("--test_labels_path", type=str, required=True, help="Path to the test labels file (.pkl)")
    parser.add_argument("-iqr_multiplier", type=float, default=1.5, help="Multiplier for IQR to set the threshold")

    args = parser.parse_args()
    main(args)