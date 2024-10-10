# QNQCDE

## Introduction
This is the repository of our paper 'QNQCDE: Simple Contrastive Learning of Dialogue Embeddings using Question and Non-Question pairs'.

## Quick Start
We prepare a set of shell scripts for convenient usage.

Model Options: `bert`,`roberta`,`t5`,`todbert`,`blender`,`plato`,`bge`

Dataset Options: `bitod`,`doc2dial`,`metalwoz`,`mwoz`,`selfdialog`,`sgd`

### Installation
```shell
git clone https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/QNQCDE
cd QNQCDE

# conda create -n qnqcde python=3.8
pip3 -r install requirements.txt

## download all datasets and move them to ./QNQCDE/datasets/
## url -> https://drive.google.com/file/d/1KpxQGXg9gvH-2u21bAMykL5N-tpYU2Dr/view?usp=sharing

## download all trained checkpoints and move them to ./QNQCDE/output/
## url -> https://drive.google.com/file/d/1JVod0OLyiVeIRVxvA-uk1TKa_zMn-sZK/view?usp=sharing

## download useful PLM and move them to ./QNQCDE/model/
## url -> https://drive.google.com/file/d/1Xq_nj-le_Mm6iUUHjltPtJZYd6gmSvNb/view?usp=sharing
```

### Data
The dataset required for this project can be downloaded from the following link:
[Download Dataset](https://drive.google.com/drive/folders/1sNowWiejo_Hwf1y1HSl9w2PDLK1BUxRj?usp=sharing)

### Train
Train QNQCDE on `${dataset}`, we can use the following script:

```shell
sh scripts/train/run_plato.sh 'doc2dial' 'train' 5 100  # train QNQCDE on doc2dial dataset
```

### Inference
Evaluate the performance of `${model}` on `${dataset}`, we can use the following script:

```shell
sh scripts/inference/run_roberta.sh 'mwoz'  # evaluate roberta on MultiWOZ dataset.
```

#### Quick inference
If you want to inference several models on one dataset in one line, then you can receive results for all models.

```shell
sh scripts/inference/run_dataset.sh 'bitod'  # evaluate all models including QNQCDE and baselines on BiTOD dataset.
```

If you want to inference one model on several datasets in one line, then you can receive results for all datasets.

```shell
sh scripts/inference/run_model.sh 'todbert'   # evaluate TOD-BERT on six datasets.
```

## Citation
