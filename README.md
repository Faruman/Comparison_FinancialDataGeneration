# Generative AI for Banks: Benchmarks and Algorithms for Synthetic Financial Transaction Data

The following repository contains the online appendix for the paper "Generative AI for Banks: Benchmarks and Algorithms for Synthetic Financial Transaction Data". It not only provides the python code for all experiments conducted but also background information for the literature review.

## [Literature Review: Financial Synthetic Data Generation](literatureReview/README.MD)
This section provides the results of a systematic literature review of synthetic data generation techniques specific to financial transactions. The review follows the PRISMA methodology and ideintifies the best algorithm for synthetic financial transaction data generation as well as suitable evaluation metrics.

This resulted in the selection of the following algorithms to be investigated more closely:
- Conditional Tabular GAN (CTGAN)
- DoppelGANger (DGAN)
- Wasserstein GAN with Discriminator Rejection Sampling (WGANDRS)
- Temporal Variational Auto Encoder (TVAE)
- Financial Diffusion Model (FinDiff)

## Algorithm Evaluation: Synthetic Financial Transaction Data

This section contains the code for tuning, training and evaluating the previously identified algorithms for synthetic financial transaction data generation. This is done in two steps, first the optimal hyperparameters for each synthetic data generation algorithm are identified and secondly the algorithm in combination with the best hyper paramaters is trained and then evaluated by comparing its generated data to the training dataset.
### [Hyperparameter Selection](parameterSearch/README.MD)
Detailed steps and code used for hyperparameter optimization of different generative models introduced previously. This section outlines how the best-performing hyperparameters were selected for each model to maximize quality of the synthetic data and similarity to real data.

### [Model Training and Evaluation](evaluation/README.MD)
This section provides scripts and instructions for training and evaluating the performance of synthetic data generation algorithms. Evaluating evaluating fidelity, synthesis, privacy, and graph structure similarity.
