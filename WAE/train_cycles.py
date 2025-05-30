import os
import numpy as np
import torch as th
from torch import nn, optim
import pickle as pkl
from models.LSTMAE import LSTM_AE
from models.LSTM_SAE import LSTM_SAE
from models.LSTM_SAE_multi_encoder import LSTM_SAE_MultiEncoder
from models.LSTM_AE_multi_encoder import LSTM_AE_MultiEncoder
from models.LSTM_AE_diff_comp import LSTM_AE_MultiComp
from models.LSTM_SAE_diff_comp import LSTM_SAE_MultiComp
from models.TCN_AE import TCN_AE
import tqdm
from ArgumentParser import parse_arguments


def train_model(model,
                train_tensors,
                epochs,
                lr,
                args):

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    loss_over_time = {"train": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        with tqdm.tqdm(train_tensors, unit="cycles") as tqdm_epoch:
            for train_tensor in tqdm_epoch:
                tqdm_epoch.set_description(f"Epoch {epoch+1}")
                optimizer.zero_grad()
                loss, _ = model(train_tensor)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        loss_over_time['train'].append(train_loss)

        print(f'Epoch {epoch+1}: train loss {train_loss}')

    return model, loss_over_time


def predict(model, test_tensors, tqdm_desc):
    test_losses = []
    with th.no_grad():
        model.eval()
        with tqdm.tqdm(test_tensors, unit="cycles") as tqdm_epoch:
            for test_tensor in tqdm_epoch:
                tqdm_epoch.set_description(tqdm_desc)
                loss, _ = model(test_tensor)
                test_losses.append(loss.item())
    return test_losses


def calculate_train_losses(model, args):
    # *** CHANGE: Use the new train data path ***
    with open(args.train_data_path, "rb") as tensor_pkl:
        train_tensors_np = pkl.load(tensor_pkl)
        train_tensors = [th.tensor(tensor_np, dtype=th.float32).to(args.device) for tensor_np in train_tensors_np]

    train_losses = predict(model, train_tensors, "Calculating training error distribution")

    return train_losses


def offline_train(model, args):

    print(f"Starting offline training   Fold {args.fold} ")
    # *** CHANGE: Use the new train data path ***
    print(f"Offline training: loading data from {args.train_data_path}")
    with open(args.train_data_path, "rb") as tensor_pkl:
        train_tensors_np = pkl.load(tensor_pkl)
        train_tensors = [th.tensor(tensor_np, dtype=th.float32).to(args.device) for tensor_np in train_tensors_np]

  
    model, loss_over_time = train_model(model,
                                        train_tensors,
                                        epochs=args.EPOCHS,
                                        lr=args.LR,
                                        args=args)

    train_losses = predict(model, train_tensors, "Calculating training error distribution")

    with open(args.results_string("offline"), "wb") as loss_file:
        pkl.dump(loss_over_time, loss_file)

    th.save(model.state_dict(), args.model_saving_string)

    return model, train_losses


def calculate_test_losses(model, args):

    # *** CHANGE: Use the new train data path ***
    with open(args.test_data_path, "rb") as tensor_pkl:
        test_tensors_np = pkl.load(tensor_pkl)
        test_tensors = [th.tensor(tensor_np, dtype=th.float32).to(args.device) for tensor_np in test_tensors_np]

    test_losses = predict(model, test_tensors, "Testing on new data")

    losses_over_time = {"test": test_losses, "train": args.train_losses}

    
    with open(args.results_string("complete"), "wb") as loss_file:
        pkl.dump(losses_over_time, loss_file)

    return model



def load_parameters(arguments):
    # *** CHANGE: all freats 313 ***
    FEATS_TO_NUMBER = {"analog_f6eats": 8, "digital_feats": 8, "all_feats": 313} 

    arguments.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    arguments.FEATS = f"{arguments.FEATS}_feats"
    arguments.NUMBER_FEATURES = FEATS_TO_NUMBER[arguments.FEATS]

 
    # arguments.results_folder = "results/"
    # data_base_path = "Data/mimi_prepr"

    
    # arguments.train_data_path = os.path.join(data_base_path, f"{arguments.machine_type}_{arguments.machine_id}_train_cycles.pkl")
    # arguments.test_data_path = os.path.join(data_base_path, f"{arguments.machine_type}_{arguments.machine_id}_test_cycles.pkl")

    # k-fold implementation 
    arguments.results_folder = "results/"
    data_base_path = "temp_data/"
    arguments.train_data_path = os.path.join(data_base_path, f"tmp_{arguments.machine_type}_{arguments.machine_id}_train.pkl")
    arguments.test_data_path = os.path.join(data_base_path, f"tmp_{arguments.machine_type}_{arguments.machine_id}_test.pkl")


    
    
    if not os.path.exists(arguments.train_data_path): print(f"Warning: {arguments.train_data_path} not found")
    if not os.path.exists(arguments.test_data_path): print(f"Warning: {arguments.test_data_path} not found")

    
    if "tcn" in arguments.MODEL_NAME:
        arguments.model_string = f"{arguments.MODEL_NAME}_{arguments.FEATS}_{arguments.EMBEDDING}_{arguments.tcn_layers}_{arguments.tcn_hidden}_{arguments.tcn_kernel}"
    else:
        
        arguments.model_string = f"{arguments.MODEL_NAME}_{arguments.FEATS}_{arguments.EMBEDDING}_{arguments.LSTM_LAYERS}"

   
    # *** CHANGE: f"{arguments.results_folder}final_{loop_no}_losses_{arguments.model_string}_{arguments.machine_type}_{arguments.machine_id}_{arguments.EPOCHS}_{arguments.LR}.pkl" ***
    os.makedirs(arguments.results_folder, exist_ok=True)

    
    
    
    # arguments.results_string = lambda loop_no: f"{arguments.results_folder}final_{loop_no}_losses_{arguments.model_string}_{arguments.machine_type}_{arguments.machine_id}_{arguments.EPOCHS}_{arguments.LR}.pkl"
    # arguments.model_saving_string = f"{arguments.results_folder}final_offline_{arguments.model_string}_{arguments.machine_type}_{arguments.machine_id}_{arguments.EPOCHS}_{arguments.LR}.pt"



    # K-FOLD USE
    arguments.results_string = lambda loop_no: f"{arguments.results_folder}{arguments.machine_type}_{arguments.machine_id}_Fold{arguments.fold}.pkl"
    arguments.model_saving_string = f"{arguments.results_folder}final_offline_{arguments.model_string}_{arguments.machine_type}_{arguments.machine_id}_Fold{arguments.fold}_{arguments.EPOCHS}_{arguments.LR}.pt"
   
    return arguments

def main(arguments):

    MODELS = {"lstm_ae": LSTM_AE, "lstm_sae": LSTM_SAE,
              "multi_enc_sae": LSTM_SAE_MultiEncoder, "multi_enc_ae": LSTM_AE_MultiEncoder,
              "diff_comp_sae": LSTM_SAE_MultiComp, "diff_comp_ae": LSTM_AE_MultiComp,
              "tcn_ae": TCN_AE}

  
    if "tcn" in arguments.MODEL_NAME:
        model = MODELS[arguments.MODEL_NAME](arguments.NUMBER_FEATURES,
                                             arguments.EMBEDDING,
                                             arguments.DROPOUT,
                                             arguments.tcn_layers,
                                             arguments.device,
                                             arguments.tcn_hidden,
                                             arguments.tcn_kernel).to(arguments.device)
    else:
        
        model = MODELS[arguments.MODEL_NAME](arguments.NUMBER_FEATURES,
                                             arguments.EMBEDDING,
                                             arguments.DROPOUT,
                                             arguments.LSTM_LAYERS,
                                             arguments.device,
                                             arguments.sparsity_weight,
                                             arguments.sparsity_parameter).to(arguments.device)

   
    if os.path.exists(arguments.model_saving_string) and not arguments.force_training:
        model.load_state_dict(th.load(arguments.model_saving_string))
        arguments.train_losses = calculate_train_losses(model, arguments)
    else:           
        model, arguments.train_losses = offline_train(model, arguments)

    
    calculate_test_losses(model, arguments)
    print("Execution finished.")




if __name__ == "__main__":    
    argument_dict = parse_arguments()    
    argument_dict = load_parameters(argument_dict)
    main(argument_dict)
