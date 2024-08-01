# DREAMix
This repository contains the code to reproduce the results from the 2024 DREAM challenge for Olfactory Mixtures by the **who_nose** team.

## Setup
``

## Dataset
We gathered several single compound odor and odor mixture datasets for pre-training: 
- for the primary odor map (POM) graph neural network: 
    - GoodScents/Leffingwell
    - Leffingwell
    - Mayhew
    - Abraham
    - Arctander
    - AromaDB
    - Flavornet
    - Sharma
    - Sigma
- for the transformer neural network (CheMix):
    - Snitz
    - Bushdid
    - Ravia

### Pre-processing
First, the identity of each compound was converted from CID to IsomericSMILES. Although some of the CIDs were parsed incorrectly, we corrected them (check our dataset building code to reproduce it). We removed compounds with elements exclusive to the POM and CheMix datasets when compared with the DREAM challenge dataset.

Wherever possible, the rdkit descriptors are appended to the input representation.

The data is split 80/20 training/validation using 10 random seeds.

## Modelling
We first pre-trained the POM and CheMix to get a warm-start for each of the models. The models are concatenated together for fine-tuning on the mixture datasets end-to-end. Hyperparameter optimization is used throughout all the training steps.

### POM
The POM has a graph attention neural network architecture. It uses the Principal Neighbourhood Aggregator to generate the global node embedding.

### CheMix
The CheMix has an attention transformer neural network architecture where it uses self-attention and cross-attention across the molecules in a mixture.

### GLM
The generalized linear model is appended at the end of the model to handle the correct loss function and post-processing for each dataset and task type (i.e. regression, binary, multiclass, mutlilabel, and zero-inflated)

### End-to-end Prediction
The DREAM challenge dataset is used to fine-tune the concatenated model. This model is used to generate the final submissions for the DREAM challenge.
