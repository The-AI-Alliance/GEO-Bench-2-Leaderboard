---
title: GEO-Bench Leaderboard
emoji: 🏆
colorFrom: purple
colorTo: green
sdk: docker
pinned: false
---

# 🏆 GEO-Bench Leaderboard

The [GEO-Bench leaderboard](https://huggingface.co/spaces/aialliance/GEO-Bench-Leaderboard) tracks performance of geospatial foundation models on various benchmark datasets using the GEO-Bench benchmarking framework. 

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Language: Python](https://img.shields.io/badge/language-Python%203.10%2B-green?logo=python&logoColor=green)](https://www.python.org)

## 1. How to Submit New Results

### 1.1. Create New Submission Directory
Create a new folder in the `new_submission` top directory:
```bash
geobench_leaderboard/
└── new_submission/
    ├── results_and_parameters.csv
    ├── additional_info.json
```

### 1.2. Add Results and Parameters Details
Add a CSV file (`results_and_parameters.csv`) with the columns below. Please note that if terratorch-iterate is used for experiments, this table may be created automatically upon completion of an experiment. Please see the `examples/results_and_parameters.csv`  for an example.
 - `backbone`: backbone used for experiment, (e.g. Prithvi-EO-V2 600M)
 - `dataset`: some or all of the GEO-bench datasets. Please see Info page to learn more.
 - `Metric`: the type of metric used for evaluation. Depending on the dataset, this may be one of the following: `Overall_Accuracy`, `Multilabel_F1_Score`, `Multiclass_Jaccard_Index`
 - `experiment_name`: if terratorch-iterate used, this will the experiment_name used in mlflow. Otherwise, a unique name may be used for all results relating to a single backbone
 - `batch_size_selection`: denotes whether the batch size was fixed during hyperparameter optimization. May be `fixed` or `optimized`
 - `early_stop_patience`: early stopping patience using for trainer
 - `n_trials`: number of trials used for hyperparameter optimization
 - `Seed`: random seed used for repeated experiment. At least 5 random seeds must be used for each backbone
 - `batch_size`: batch size used for repeated experiments for each backbone/dataset combination.
 - `weight_decay`: weight decay experiments for each backbone/dataset combination.
 - `lr`: learning rate used for repeated experiments for each backbone/dataset combination. Obtained from hyperparameter optimization (HPO)
 - `test metric`: metric obtained from running backbone on the dataset during repeated experiment. Please see Info page to learn more. 


### 1.3. Add Additional Information
Create a JSON file (`additional_info.json`) with information about your submission and any new models that will be included.
The JSON file MUST have the same file name and contain the same keys as the `examples/additional_info.json` file. 


### 1.4. Submit PR

 - Fork the repository
 - Add your results following the structure above and in the PR comments add more details about your submission
 - Create a pull request to main


## 2. Benchmarking with Terratorch-Iterate
The [TerraTorch-Iterate](https://github.com/IBM/terratorch-iterate) library, based on [TerraTorch](https://github.com/IBM/terratorch), leverages MLFlow for experiment logging, optuna for hyperparameter optimization and ray for parallelization. It includes functionality to easily perform both hyperparameter tuning and re-repeated experiments in the manner prescribed by the GEO-Bench protocol. The `summarize` feature of `TerraTorch-Iterate` can be used to automatically create a `results_and_parameters.csv` file for submission, once benchmarking is complete.


### 2.1 Installation
Please see [TerraTorch-Iterate](https://github.com/IBM/terratorch-iterate) for installation instructions

### 2.2 Running benchmark experiments
**On existing models**:  To run experiments on an existing model, a custom config file specifying the model and dataset parameters should be prepared. To compare performance of multiple models, define a config file with unique experiment name for each model being comapred. Please see the `examples` folder for sample config files. Each config file (experiment) can then be executed with the following command:

```
terratorch iterate --hpo --repeat --config <config-file>
```

**On new models**: New models can be evaluated by first onboarding them to the [TerraTorch](https://github.com/IBM/terratorch/) library. Once onboarded, benchmarking may be conducted as outlined above.


### 2.3 Summarizing and plotting results 
**Extract results and parameters**: The command below can be used to extract results and hyperparameters file for submission to the leaderboard. Please see details at the following link: https://github.com/terrastackai/iterate?tab=readme-ov-file#summarizing-results. 
```
terratorch iterate --summarize --config <summarize-config-file>
```

 **NOTE: Please use the `scale_rmse` function as shown in `normalization_example.ipynb` to convert the biomassters `rmse` value to the corresponding de-normalized score before submission to the leaderboard.**
