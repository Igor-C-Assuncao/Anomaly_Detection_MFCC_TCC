import numpy as np
import os
import pickle as pkl
import matplotlib.pyplot as plt

def calculate_anomaly_threshold(train_errors, multiplier=3.0):
 
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


    train_errors = np.array(results['train'], dtype=float)
    test_errors = np.array(results['test'], dtype=float)


    limiar = calculate_anomaly_threshold(train_errors, multiplier=args.iqr_multiplier)
    anomalies = np.where(test_errors > limiar)[0]
    print(f"Anomalias detectadas: {len(anomalies)} em {len(test_errors)}")
    print(f"Limiar de Anomalia: {limiar:.6f} ")
    

   
    plt.subplot(1, 2, 1)
    plt.plot(train_errors, label="Treino Errors", color="blue")
    plt.axhline(y=limiar, color="red", linestyle="--", label="Limiar de Anomalia")
    plt.title("Treino Errors")
    plt.xlabel("Índice")
    plt.ylabel("Perda")
    plt.legend()

   
    plt.subplot(1, 2, 2)
    plt.plot(test_errors, label="Teste Errors", color="blue")
    plt.axhline(y=limiar, color="red", linestyle="--", label="Limiar de Anomalia")
    plt.title("Teste Errors com Anomalias")
    plt.xlabel("Índice")
    plt.ylabel("Perda")
    plt.legend()



    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Anomaly Detection using WAE-GAN")
    parser.add_argument("--results_file", type=str, required=True, help="Path to the results file (.pkl)")
    parser.add_argument("-iqr_multiplier", type=float, default=1.5, help="Multiplier for IQR to set the threshold")

    args = parser.parse_args()
    main(args)