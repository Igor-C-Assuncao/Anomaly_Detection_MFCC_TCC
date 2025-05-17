import argparse
import os
import pickle
import numpy as np
import torch as th
import shap

from lime import lime_tabular

from models.LSTMAE import LSTM_AE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--machine_type", required=True)
    parser.add_argument("--machine_id", required=True)
    parser.add_argument("--num_instances", type=int, default=5)
    parser.add_argument("--model_name", type=str, default="lstm_ae")
    parser.add_argument("--model_filename", type=str, required=True)

    # parâmetros do modelo
    parser.add_argument("--embedding", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--feats", type=str, default="all_feats")
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def main(args):
    device = th.device(args.device if th.cuda.is_available() else "cpu")

    # carregar os dados
    with open(args.data_path, "rb") as f:
        data = pickle.load(f)
    X_test = data["X"]
    X_test = th.tensor(X_test, dtype=th.float32).to(device)

    num_features = X_test.shape[2]

    # instanciar o modelo
    model = LSTM_AE(
        n_features=num_features,
        emb_dim=args.embedding,
        dropout=args.dropout,
        lstm_layers=args.lstm_layers,
        device=device
    )

    # carregar os pesos
    model_path = os.path.join(args.model_dir, args.model_filename)
    state_dict = th.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # selecionar instâncias para explicar
    selected_indices = np.random.choice(len(X_test), size=args.num_instances, replace=False)
    selected_instances = X_test[selected_indices]

    # função wrapper para SHAP
    def model_forward(x):
        x = x.to(device)
        loss, output = model(x)
        return output.detach().cpu().numpy()

    # aplicar SHAP
    explainer = shap.DeepExplainer(model_forward, selected_instances)
    shap_values = explainer.shap_values(selected_instances)

    # salvar as explicações
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.model_name}_explanations.pkl")
    with open(output_file, "wb") as f:
        pickle.dump({
            "shap_values": shap_values,
            "instances": selected_instances.detach().cpu().numpy()
        }, f)

    

    lime_output_dir = os.path.join(args.output_dir, "lime")
    os.makedirs(lime_output_dir, exist_ok=True)

    def predict_fn(inputs_np):
        """
        Função de predição para o LIME.
        inputs_np: array 2D (n_instances, seq_len * n_features)
        """
        inputs = th.tensor(inputs_np.reshape(-1, X_test.shape[1], X_test.shape[2]), dtype=th.float32).to(device)
        with th.no_grad():
            _, outputs = model(inputs)
        return outputs.detach().cpu().numpy().reshape(inputs_np.shape)

    # reshape para LIME: (num_instances, seq_len * n_features)
    X_reshaped = X_test.cpu().numpy().reshape(X_test.shape[0], -1)

    # inicializa o LIME
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_reshaped,
        mode='regression',
        feature_names=[f't_{t}_f_{f}' for t in range(X_test.shape[1]) for f in range(X_test.shape[2])],
        verbose=True
    )

    lime_explanations = []

    for idx in selected_indices:
        instance = X_reshaped[idx]
        exp = explainer.explain_instance(instance, predict_fn, num_features=10)
        lime_explanations.append(exp)

        # salvar visualmente cada explicação em HTML
        exp.save_to_file(os.path.join(lime_output_dir, f"lime_exp_{idx}.html"))

    # salvar objetos brutos se quiser reusar
    with open(os.path.join(lime_output_dir, "lime_explanations.pkl"), "wb") as f:
        pickle.dump(lime_explanations, f)

    print(f"LIME explicações salvas em: {lime_output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
