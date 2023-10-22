# 🌾 Digital Green Crop Yield Estimate Challenge

<img src='assets/banner.png'>

Objective of this challenge is to create a machine learning solution to predict the crop yield per acre of rice or wheat
crops in India.

## 🏁 Getting started

1 - Create the conda environment:

```bash
conda env create -f environment.yml
```

2 - Activate the conda environment:

```bash
conda activate crop-yield-estimate-env
```

## 🏁 Optimize models hyperparameters

```bash
nohup python src/models/train_model.py --estimator_name XGBoost --nb_agents 10 </dev/null &>/dev/null &
```


