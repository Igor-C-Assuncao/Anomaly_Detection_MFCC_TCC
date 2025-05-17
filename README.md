# MIMII Dataset Analysis with Wasserstein Autoencoders (WAE)

This project implements a pipeline for preprocessing, training, and evaluating Wasserstein Autoencoders (WAE) on the MIMII dataset. The pipeline includes data preprocessing, cycle generation, model training, and evaluation.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The MIMII dataset contains audio recordings of industrial machines under normal and abnormal conditions. This project uses Wasserstein Autoencoders (WAE) to detect anomalies in these recordings. The pipeline includes:
- Preprocessing `.wav` files into MFCC features.
- Generating cycles for training and testing.
- Training WAE models with adversarial and non-adversarial approaches.
- Evaluating the models on separate test sets.

This implementation is inspired by the research paper:
> [Predictive Maintenance: Adversarial Autoencoders and Explainability](https://www.researchgate.net/publication/373987684_Predictive_Maintenance_Adversarial_Autoencoders_and_Explainability)

---

## Features

- **Preprocessing**: Converts `.wav` files to MFCC features and normalizes them.
- **Cycle Generation**: Splits data into cycles for training and testing.
- **Model Training**: Supports various WAE architectures, including LSTM and TCN-based models.
- **Evaluation**: Calculates reconstruction errors and critic scores for anomaly detection.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Igor-C-Assuncao/Anomaly_Detection_MFCC_TCC
   
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the `mtsa` module:
   ```bash
   pip install mtsa
   ```

---

## Usage

### 1. Preprocessing the Data
Run the preprocessing script to convert `.wav` files into normalized MFCC cycles(check the path of your Data)  :
```bash
python \mtsa\preprocessing_mimii.py
```

### 2. Training the Model
Train the WAE model using the generated cycles:
```bash
python .\WAE\WAE_hyperparams.py all {machine type} id_{id number}
```

Inside the WAE_hyperparams.py chose the parameters of the model: 

[Parameters Guide](Parameters.md)

Example for non adversarial training:
```python
base_string = lambda discriminator_lr, disc_params, enc_dec_params, lr: f"python WAE/train_cycles.py \
-encoder TCN -decoder TCN -use_discriminator -model {model} -embedding {emb_size} \
-epochs {epochs} -lr {lr}  -disc_lr {discriminator_lr} {disc_params} {enc_dec_params} -batch_size 64 -feats {feats} \
-machine_type {machine_type} -machine_id {machine_id} -hidden 30 -tcn_layers 10 \
-tcn_hidden 30  -WAEreg 10 -force-training -dropout 0.3"
```

Example for adversarial training:

```python
base_string = lambda discriminator_lr, disc_params, enc_dec_params, lr: f"python WAE/train_cycles_adversarial.py \
-encoder LSTM -decoder LSTM -use_discriminator -model {model} -embedding {emb_size} \
-epochs {epochs} -lr {lr}  -disc_lr {discriminator_lr} {disc_params} {enc_dec_params} -batch_size 64 -feats {feats} \
-machine_type {machine_type} -machine_id {machine_id} -hidden 30 -tcn_layers 10 \
-tcn_hidden 30 -WAEreg 10 -force-training -dropout 0.3"

``` 

### 3. Evaluating the Model
Evaluate the trained model on the test set:
```bash
python .\anomaly_detection.py --results_file {.plk Result Path)
```

---

## Project Structure

```
Implementação TCC/
├── MTSA/
│   ├── preprocessing_mimii.py       # Preprocessing script for MIMII dataset
│   ├── utils.py                     # Utility functions for data handling
│   ├── models/                      # Model definitions
│   └── features/                    # Feature extraction modules
├── WAE/
│   ├── train_cycles.py              # Training script for WAE models
│   ├── train_cycles_adversarial.py  # Adversarial training script
│   ├── train_chunks.py              # Training script for chunk-based models (not used in this implementation)
│   ├── WAE_hyperparams.py           # Hyperparameter configuration
│   └── ArgumentParser.py            # Argument parsing for scripts
├── README.md                        # Project documentation
├── preprocessing_mimii.py           # Pre-Processing Data
├── anomaly_detectionAdversarial.py  # Results of training process 
├── anomaly_detection.py             # Results of adversarial training process 
└── Data/
    ├── preprocessed_mimii         # Preprocessed data
    └── MIMII/                     # Raw MIMII dataset
```

---

## References

This implementation is based on the following research paper:
- [Predictive Maintenance: Adversarial Autoencoders and Explainability](https://www.researchgate.net/publication/373987684_Predictive_Maintenance_Adversarial_Autoencoders_and_Explainability)

Dataset used in this implementation 
- [MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspectiony](https://zenodo.org/records/3384388)

---

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push them to your fork.
4. Submit a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
