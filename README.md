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
   git clone https://github.com/your-repo/mimii-wae.git
   cd mimii-wae
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
Run the preprocessing script to convert `.wav` files into normalized MFCC cycles:
```bash
python \MTSA\preprocessing_mimii.py
```

### 2. Training the Model
Train the WAE model using the generated cycles:
```bash
python \WAE\train_cycles.py
```

For adversarial training:
```bash
python \WAE\train_cycles_adversarial.py
```

### 3. Evaluating the Model
Evaluate the trained model on the test set:
```bash
python \WAE\train_chunks.py
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
│   ├── train_chunks.py              # Training script for chunk-based models
│   ├── WAE_hyperparams.py           # Hyperparameter configuration
│   └── ArgumentParser.py            # Argument parsing for scripts
├── README.md                        # Project documentation
└── Data/
    ├── preprocessed_mimii/          # Preprocessed data
    └── raw/                         # Raw MIMII dataset
```

---

## References

This implementation is based on the following research paper:
- [Predictive Maintenance: Adversarial Autoencoders and Explainability](https://www.researchgate.net/publication/373987684_Predictive_Maintenance_Adversarial_Autoencoders_and_Explainability)

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