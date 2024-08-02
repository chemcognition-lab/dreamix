# DreaMix: From molecules to mixtures in one model
### One model to mix them all, and in the latent space bind them. 

This repository contains the code to reproduce the results from the 2024 DREAM challenge for Olfactory Mixtures by the **who_nose** team.

Our report can be found at: www.synapse.org/dreamix

## Setup
```
git clone https://github.com/chemcognition-lab/dreamix.git

pip install -r colab_requirements.txt
```

## Summary Sentence
Our hierarchical model, DreaMix, combines graph neural networks, attention mechanisms, primary odour maps, and similarity measures to predict olfactory similarity of chemical mixtures, leveraging permutation invariance and multi-scale data sources.

## Background/Introduction
Predicting olfactory properties is essentially the same task whether we're dealing with single molecules or mixtures. A robust olfactory model should handle both inputs seamlessly. In previous work, Lee et al. built a Primary Odour Map (POM) to predict odour labels, odour similarity, and odour intensity of a single molecule. We extend this work to tackle the mixture similarity problem by jointly training a POM with an attention-based mixture model to predict similarity of mixtures. This approach also allows us to combine monomolecular datasets (up to 10,000 data points) and more limited mixture data sources (up to 1,000 data points).

## Methods
### Data
Our datasets come in two modalities: monomolecular datasets and multimolecular (mixture) datasets. Monomolecular datasets covered various regression, classification, and multi-classification tasks. We used the following monomolecular datasets from Pyrfume: Abraham, Arctander, AromaDB, Flavornet, IFRA, Keller, Leffingwell, Mayhew, and Sigma, supplemented with GoodScents and Leffingwell datasets. We created a consistent cross-dataset ontology by translating all molecule names, CIDs, SMILES, etc., to a canonical SMILES representation, enabling us to remove duplicate entries and unsuitable molecules.

Multimolecular datasets were independently compiled from previous publications by Snitz, Ravia, and Bushdid. We found additional data points in the original publication that were excluded or incorrectly parsed in the organizer-provided datasets.

### Modeling
Our model architecture, DreaMix, has three stages:
1. A GNN processes molecules and generates POMs, vector representations of molecules optimized for olfactory properties.
2. CheMix combines a set of POM embeddings into a single mixture vector (POM-Mix).
3. Two mixtures' POM-Mix embeddings are converted into a similarity score.

### Training and Optimization
The model uses batches of the training set to make predictions. We use a differentiable loss function to compute the difference between ground truth labels and predictions, tracking a non-differentiable metric on the validation set for early stopping. Regularization techniques, including layer normalization, dropout, and fewer nodes in linear layers, were used to prevent overfitting.

## Code and Reproducibility
Our code, model, and scripts are available at [GitHub](https://github.com/chemcognition-lab/dreamix). This repository expands on all aspects of the model, training, hyperparameters, and dataset curation. Additionally, we provide thorough analysis and corrections to some input data, presented in the competition forum.

### Explainer Colab Notebooks
1. **Exploratory Data Analysis**: Basic visualization and statistics of the different datasets.
2. **Model**: Step-by-step tutorial on POM, CheMix, inputs, outputs, inference, and results.

## Results
We report results across three metrics: Pearson r, root-mean-squared error (RMSE), and the Kendall ranking tau. Scores are on the DREAM leaderboard set reconstructed from the mixture datasets. Detailed results, including performance metrics and confidence intervals, are provided.

