import os
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from scipy.stats import sem, t
import subprocess

# Caminho para os dados pré-processados
data_dir = "Data"
machine_types = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
n_folds = 10

# Função para rodar o pipeline de K-fold cross-validation
def run_kfold_pipeline(machine_type, machine_id):
    roc_aucs, f1s, accs, precs, recalls = [], [], [], [], []
    # Carregar todos os ciclos de treino e teste completos
    train_path = os.path.join(data_dir, "preprocessed_mimii", f"{machine_type}_{machine_id}_train_cycles.pkl")
    test_path = os.path.join(data_dir, "preprocessed_mimii", f"{machine_type}_{machine_id}_test_cycles.pkl")
    test_labels_path = os.path.join(data_dir, "preprocessed_mimii", f"{machine_type}_{machine_id}_test_labels.pkl")

    with open(train_path, "rb") as f:
        all_train_cycles = pickle.load(f)
    with open(test_path, "rb") as f:
        all_test_cycles = pickle.load(f)
    with open(test_labels_path, "rb") as f:
        all_test_labels = pickle.load(f)

    n_train = len(all_train_cycles)
    fold_size = n_train // n_folds
    indices = np.arange(n_train)

    for fold in range(n_folds):
        # Definir índices do fold de teste e treino
        test_idx = indices[fold*fold_size:(fold+1)*fold_size] if fold < n_folds-1 else indices[fold*fold_size:]
        train_idx = np.setdiff1d(indices, test_idx)
        train_cycles = [all_train_cycles[i] for i in train_idx]
        test_cycles = [all_train_cycles[i] for i in test_idx] + all_test_cycles
        test_labels = [0]*len(test_idx) + all_test_labels

        # Salvar temporariamente para o WAE-GAN
        train_tmp_path = f"temp_data/tmp_{machine_type}_{machine_id}_train.pkl"
        with open(train_tmp_path, "wb") as f:
            pickle.dump(train_cycles, f)
        test_tmp_path = f"temp_data/tmp_{machine_type}_{machine_id}_test.pkl"
        with open(test_tmp_path, "wb") as f:
            pickle.dump(test_cycles, f)
        test_labels_tmp_path = f"temp_data/tmp_{machine_type}_{machine_id}_test_labels.pkl"
        with open(test_labels_tmp_path, "wb") as f:
            pickle.dump(test_labels, f)

        # Use subprocess para acompanhar os processos no terminal
        subprocess.run(f"python WAE/WAE_hyperparams.py all {machine_type} {machine_id} {fold}", shell=True)
        subprocess.run(f"python anomaly_detection.py --results_file results/{machine_type}_{machine_id}_Fold{fold}.pkl  --test_labels_path {test_labels_tmp_path}", shell=True)

        # Carregar resultados do fold
        results_file = f"results/{machine_type}_{machine_id}_Fold{fold}_metrics.pkl"
        if os.path.exists(results_file):
            with open(results_file, "rb") as f:
                metrics = pickle.load(f)
            roc_aucs.append(metrics["roc_auc"])
            f1s.append(metrics["f1"])
            accs.append(metrics["accuracy"])
            precs.append(metrics["precision"])
            recalls.append(metrics["recall"])

        # Limpar arquivos temporários
        os.remove(train_tmp_path)
        os.remove(test_tmp_path)
        os.remove(test_labels_tmp_path)

    # Calcular média, desvio padrão e intervalo de confiança para ROC-AUC
    roc_auc_mean = np.mean(roc_aucs)
    roc_auc_std = np.std(roc_aucs)
    conf_int = t.interval(0.95, len(roc_aucs)-1, loc=roc_auc_mean, scale=sem(roc_aucs))
    print(f"{machine_type}/{machine_id} - ROC-AUC: {roc_auc_mean:.4f} ± {roc_auc_std:.4f} (95% CI: {conf_int})")
    print(f"F1-score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"Acurácia: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Precisão: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    print(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")

# Exemplo para rodar para todos os tipos/ids de máquina
for machine_type in ["valve"]:  
    for machine_id in ["id_00"]: 
        run_kfold_pipeline(machine_type, machine_id)