# QNQCDE

## Introduction
This is the repository of our paper 'QNQCDE: Simple Contrastive Learning of Dialogue Embeddings using Question and Non-Question pairs'.

## Quick Start
We prepare a set of shell scripts for convenient usage.

Model Options: `bert`,`roberta`,`t5`,`todbert`,`blender`,`plato`,`bge`

Dataset Options: `metalwoz`,`mwoz`,`selfdialog`,`sgd`, `mediasum`, `multiuserwoz`

### Data
The dataset required for this project can be downloaded from the following link:
[Download Dataset](https://drive.google.com/drive/folders/1K--FFa1ogWVKVi0SzU_eXJPWedoFNcJM?usp=drive_link)

### Train
Train QNQCDE on `${dataset}`, we can use the following script:

```shell
sh scripts/train/run_plato.sh 'sgd' 'train' 5 100  # train QNQCDE on sgd dataset
```

### Inference
Evaluate the performance of `${model}` on `${dataset}`, we can use the following script:

```shell
sh scripts/inference/run_roberta.sh 'mwoz'  # evaluate roberta on MultiWOZ dataset.
```

#### Quick inference
If you want to inference several models on one dataset in one line, then you can receive results for all models.

```shell
sh scripts/inference/run_dataset.sh 'mediasum'  # evaluate all models including QNQCDE and baselines on mediasum dataset.
```

If you want to inference one model on several datasets in one line, then you can receive results for all datasets.

```shell
sh scripts/inference/run_model.sh 'todbert'   # evaluate TOD-BERT on six datasets.
```

## Citation
