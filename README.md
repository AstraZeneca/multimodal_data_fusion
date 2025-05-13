# Multiomics Survival Pipeline

## Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Contributing](../CONTRIBUTING.md)

## About <a name = "about"></a>

A python library for multimodal feature integration and survival prediction.
This pipeline has been developped by the AstraZeneca Oncology Biometrics R&D ML/AI Team. It can be used to preprocess and reduce the dimensionality of tabular datasets (unimodal or multimodal) and train & evaluate survival models on them. Its functionalities include several preprocessing & imputation options, flexibility regarding when to integrate modalities (in the case of multimodal data), a range of feature reduction approaches and survival modelling methods, rigorous evaluation including reporting the models' feature importance.

## Getting Started <a name = "getting_started"></a>

Fork the current repository to get a copy of the repo that can be modified. 
Instructions on how to fork a repo can be found at: https://support.atlassian.com/bitbucket-cloud/docs/fork-a-repository

### Prerequisites

Several python packages are required for the utilization of this library and are listed in requirements.txt. 
Alterately, use the environment.yml to create a conda environment with all required packages installed.

```
conda env create -p /path/to/location -f environment_new.yml
```
if installing to a different location, it can be symbolically linked via
```
ln -s /path/to/env_name ~/.conda/envs/env_name
```
Then activate the environment and install scikit-survival package
```
conda install -c sebp scikit-survival
```
## Usage <a name = "usage"></a>

The pipeline can be run on the terminal via:
```
python src/pipeline.py -e experiment_name -c /path/to/input/csv
```

Input File Format:

The columns must include the targets of the survival analysis task (event & times columns) along with a set of associated
features for each entry, the name of which should follow the convention: 'featurename_modalityname'. Additional subsample identifiers (cancer_type, cancer_stage, subject_id) can also be included.


Functionalities offered (see Execution ReadMe for details):
1. Subset selection:
The user can select the appropriate subsample of the available data to conduct analysis on -e.g. specific subset of modalities, specific combination of identifiers (for instance cancer stages & types).

2. Train/Test Splitting:
The user can either specify training/validation/test cohorts, or randomly split the data by specifying split ratios

3. Modality Integration:
In the case of multimodal data, the user can opt for early or intermediate modality fusion (late to be added in future versions). This can be achieved by performing the following steps on a per-modality basis, or jointly.

4. Data Preprocessing:
The user can apply normalization steps (Standardization, Robust Scaling, Min-Max Normalization) or handle missing data by either (i) only keeping full entries, (ii) imputation (mean, median, mode)

5. Feature reduction:
Several Dimensionality Reduction methods are provided:
Unsupervised - Feature Selection: The user can (i) drop/group correlated features, (ii) drop low variance features
Unsupervised - Feature Extraction: Autoencoders [To be included in the next version]
Supervised - Feature Selection - Linear: Univariate Cox PH Models
Supervised - Feature Selection - Nonlinear: Spearman Correlation, Mutual Information Maximization, Joint Mutual Information, Conditional Mutual Information Maximization.
User specified list

Feature selection supports the option of being performed on multiple on bootstraps of the training data to select stable features.

6. Survival Modelling:
Several Survival Modelling Methods are provided:
Linear: Cox PH (L2 or Elastic Net Regularization)
Nonlinear: Gradient Boosted Cox PH, Gradient Boosted Componentwise Least Squares, Random Survival Forests, Weighted Heterogeneous Ensembles.

Cross-validation is performed for model selection. Once optimal hyperparameters are identified, the models are retrained using this on the full training set.


7. Evaluation
Option for multiple runs, option for different train/test splits, in each of which test set data are only used during evaluation.
Detailed results can include TODO...

The pipeline currently supports only survival analysis (regression) tasks.
The pipeline currently supports only early & intermediate modality fusion approaches.
