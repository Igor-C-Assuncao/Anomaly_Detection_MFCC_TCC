### Script Parameters

Here is a list of command-line arguments that can be used in the \WAE_hyperparams.py:

* **`-lr`**:
    * Variable Name (Destination): `LR`
    * Type: `float`
    * Default: `1e-3`
    * Description: Learning rate.

* **`-disc_lr`**:
    * Variable Name (Destination): `disc_lr`
    * Type: `float`
    * Default: `1e-3`
    * Description: Discriminator learning rate.

* **`-epochs`**:
    * Variable Name (Destination): `EPOCHS`
    * Type: `int`
    * Default: `1000`
    * Description: Number of epochs for offline training.

* **`-l2reg`**:
    * Variable Name (Destination): `weight_decay`
    * Type: `float`
    * Default: `0`
    * Description: L2 regularization (weight decay).

* **`-critic_iterations`**:
    * Variable Name (Destination): `critic_iterations`
    * Type: `int`
    * Default: `5`
    * Description: Number of critic iterations.

* **`-gradient_penalty`**:
    * Variable Name (Destination): `GP_hyperparam`
    * Type: `float`
    * Default: `10.0`
    * Description: Hyperparameter for the gradient penalty.

* **`-WAEreg`**:
    * Variable Name (Destination): `WAE_regularization_term`
    * Type: `float`
    * Default: `1.0`
    * Description: Regularization term for WAE (Wasserstein Autoencoder).

* **`-dropout`**:
    * Variable Name (Destination): `DROPOUT`
    * Type: `float`
    * Default: `0.2`
    * Description: Dropout rate.

* **`-embedding`**:
    * Variable Name (Destination): `EMBEDDING`
    * Type: `int`
    * Default: `4`
    * Description: Embedding dimension.

* **`-hidden`**:
    * Variable Name (Destination): `HIDDEN_DIMS`
    * Type: `int`
    * Default: `30`
    * Description: Hidden layer dimensions.

* **`-n_layers`**:
    * Variable Name (Destination): `LSTM_LAYERS`
    * Type: `int`
    * Default: `2`
    * Description: Number of LSTM layers.

* **`-batch_size`**:
    * Variable Name (Destination): `BATCH_SIZE`
    * Type: `int`
    * Default: `32`
    * Description: Batch size.

* **`-disc_hidden`**:
    * Variable Name (Destination): `disc_hidden`
    * Type: `int`
    * Default: `64`
    * Description: Discriminator hidden layer dimensions.

* **`-disc_layers`**:
    * Variable Name (Destination): `disc_layers`
    * Type: `int`
    * Default: `2`
    * Description: Number of discriminator layers.

* **`-tcn_hidden`**:
    * Variable Name (Destination): `tcn_hidden`
    * Type: `int`
    * Default: `30`
    * Description: Hidden layer dimensions for TCN (Temporal Convolutional Network).

* **`-tcn_layers`**:
    * Variable Name (Destination): `tcn_layers`
    * Type: `int`
    * Default: `8`
    * Description: Number of TCN layers.

* **`-tcn_kernel`**:
    * Variable Name (Destination): `tcn_kernel`
    * Type: `int`
    * Default: `5`
    * Description: Kernel size for TCN.

* **`-sw`**:
    * Variable Name (Destination): `sparsity_weight`
    * Type: `float`
    * Default: `1.0`
    * Description: Sparsity weight for Sparse AE (Sparse Autoencoder).

* **`-sp`**:
    * Variable Name (Destination): `sparsity_parameter`
    * Type: `float`
    * Default: `0.05`
    * Description: Sparsity parameter for Sparse AE.

* **`-att_heads`**:
    * Variable Name (Destination): `NHEADS`
    * Type: `int`
    * Default: `8`
    * Description: Number of attention heads.

* **`-feats`**:
    * Variable Name (Destination): `FEATS`
    * Choices: `["analog", "digital", "all", "noflow"]`
    * Default: `"analog"`
    * Description: Which sensors to use.

* **`-SI`**:
    * Variable Name (Destination): `successive_iters`
    * Type: `int`
    * Default: `10`
    * Description: Number of successive iterations.

* **`-delta_worse`**:
    * Variable Name (Destination): `delta_worse`
    * Type: `float`
    * Default: `0.02`
    * Description: Delta for worse condition.

* **`-delta_better`**:
    * Variable Name (Destination): `delta_better`
    * Type: `float`
    * Default: `0.001`
    * Description: Delta for better condition.

* **`-model`**:
    * Variable Name (Destination): `MODEL_NAME`
    * **Required**: Yes
    * Description: Name of the model to be used.
    * Choices:
        * `lstm_ae`
        * `lstm_sae`
        * `multi_enc_sae`
        * `multi_enc_ae`
        * `lstm_all_layer_sae`
        * `diff_comp_sae`
        * `diff_comp_ae`
        * `GAN`
        * `SimpleDiscriminator`
        * `LSTMDiscriminator`
        * `ConvDiscriminator`
        * `tcn_ae`
        * `alt_lstm_ae`
        * `SimpleDiscriminator_TCN`
        * `LSTMDiscriminator_TCN`
        * `ConvDiscriminator_TCN`

* **`-encoder`**:
    * Variable Name (Destination): `ENCODER_NAME`
    * Choices: `["LSTM", "TCN"]`
    * Default: None (will be `None` if not specified)
    * Description: Name of the encoder.

* **`-decoder`**:
    * Variable Name (Destination): `DECODER_NAME`
    * Choices: `["LSTM", "TCN"]`
    * Default: None (will be `None` if not specified)
    * Description: Name of the decoder.

* **`-recons_error`**:
    * Variable Name (Destination): `reconstruction_error_metric`
    * Choices: `["dtw", "mse"]`
    * Default: `"mse"`
    * Description: Reconstruction error metric.

* **`-dtw_local`**:
    * Variable Name (Destination): `dtw_local_size`
    * Type: `int`
    * Default: `5`
    * Description: Local size for DTW (Dynamic Time Warping).

* **`-separate_comp`**:
    * Variable Name (Destination): `separate_comp`
    * Action: `store_true` (sets to `True` if the argument is passed, otherwise `False`)
    * Default: `False`
    * Description: Use separate components.

* **`-init`**:
    * Variable Name (Destination): `INIT_LOOP`
    * Type: `int`
    * Default: `0`
    * Description: Start of the loop.

* **`-end`**:
    * Variable Name (Destination): `END_LOOP`
    * Type: `int`
    * Default: `17`
    * Description: End of the loop.

* **`-force-training`**:
    * Variable Name (Destination): `force_training`
    * Action: `store_true`
    * Default: `False`
    * Description: Force training.

* **`-sensor`**:
    * Variable Name (Destination): `sensor`
    * Type: (inferred as `str` by `argparse`)
    * Default: `"tp2"`
    * Description: Sensor to be used.

* **`-train_tensor`**:
    * Variable Name (Destination): `train_tensor`
    * Type: (inferred as `str` by `argparse`)
    * Default: None (will be `None` if not specified)
    * Description: Path to the training tensor.

* **`-test_tensor`**:
    * Variable Name (Destination): `test_tensor`
    * Type: (inferred as `str` by `argparse`)
    * Default: None (will be `None` if not specified)
    * Description: Path to the test tensor.

* **`-use_discriminator`**:
    * Variable Name (Destination): `use_discriminator`
    * Action: `store_true`
    * Default: `False`
    * Description: Use a discriminator.

* **`-machine_type`**:
    * Variable Name (Destination): `machine_type`
    * Type: (inferred as `str` by `argparse`)
    * Default: None (will be `None` if not specified)
    * Description: Type of machine.

* **`-machine_id`**:
    * Variable Name (Destination): `machine_id`
    * Type: (inferred as `str` by `argparse`)
    * Default: None (will be `None` if not specified)
    * Description: ID of the machine.
 

Inside the WAE_hyperparams.py chose the parameters of the model: 


For adversarial training:
```python
base_string = lambda discriminator_lr, disc_params, enc_dec_params, lr: f"python WAE/train_cycles.py \
-encoder TCN -decoder TCN -use_discriminator -model {model} -embedding {emb_size} \
-epochs {epochs} -lr {lr}  -disc_lr {discriminator_lr} {disc_params} {enc_dec_params} -batch_size 64 -feats {feats} \
-machine_type {machine_type} -machine_id {machine_id} -hidden 30 -tcn_layers 10 \
-tcn_hidden 30  -WAEreg 10 -force-training -dropout 0.3"
```

For non adversarial training:

```python
base_string = lambda discriminator_lr, disc_params, enc_dec_params, lr: f"python WAE/train_cycles_adversarial.py \
-encoder LSTM -decoder LSTM -use_discriminator -model {model} -embedding {emb_size} \
-epochs {epochs} -lr {lr}  -disc_lr {discriminator_lr} {disc_params} {enc_dec_params} -batch_size 64 -feats {feats} \
-machine_type {machine_type} -machine_id {machine_id} -hidden 30 -tcn_layers 10 \
-tcn_hidden 30 -WAEreg 10 -force-training -dropout 0.3"

``` 
