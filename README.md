# PLM with reduced amino acid alphabets

This repository contains the implementation of various protein language models trained on reduced amino acid alphabets, along with the notebooks to recreate the figures found in the paper.

**For more details, see:** [Link after publishing](https://doi.org/10.1093/bioinformatics/). 

![Alt Text](https://github.com/Ieremie/reduced-alph-PLM/blob/main/proemb/embeddings.gif)

## About
**Motivation**: Protein Language Models (PLM), which borrowed ideas for modelling and inference from
Natural Language Processing, have demonstrated the ability to extract meaningful representations in
an unsupervised way. This led to significant performance improvement in several downstream tasks.
Clustering amino acids based on their physical-chemical properties to achieve reduced alphabets has
been of interest in past research, but their application to PLMs or folding models is unexplored.

**Results**: Here, we investigate the efficacy of PLMs trained on reduced amino acid alphabets in capturing
evolutionary information, and we explore how much loss of fidelity in the space of sequence impacts
learned representations and downstream task performance. Our empirical work shows that PLMs trained
on the full alphabet and a large number of sequences capture fine details that are lost in alphabet reduction
methods. We further show the ability of ESMFold to fold CASP14 protein sequences translated using a
reduced alphabet. For 10 proteins out of the 50 targets, reduced alphabets improve structural predictions
with LDDT-Cα differences of up to 19%.


## Datasets
The model is trained and evaluated using publicly available datasets:
- PLM pretraining dataset: [Uniref90](https://www.uniprot.org/help/downloads)
- Structure prediction datasest: [CASP14](https://predictioncenter.org/download_area/CASP14/) 
- Enzyme Commission (EC) dataset: [IEConv_proteins](https://github.com/phermosilla/IEConv_proteins)
- Fold recognition dataset: [TAPE](https://github.com/songlab-cal/tape)
- FLIP benchmark datasests: [FLIP](https://github.com/J-SNACKKB/FLIP)


## Pretraining PLMs on reduced alphabets
To pretrain the protein language model you can run [`train_prose_multitask.py`](./proemb/train_prose_multitask.py).
The implementation uses multiple GPUs and can be run on a single machine or on a cluster. The scripts for running the
file on a cluster can be found at [`iridis-scripts`](./proemb/iridis-scripts/multitask). The progress of the training
can be monitored using [`tensorboard.sh`](./proemb/iridis-scripts/tensorboard.sh).

## Finetuning on downstream tasks
After pretraining the protein language model, you can finetune it on downstream tasks. You can do this by running
the following python files:
- [`train_enzyme.py`](./proemb/train_enzyme.py) for the EC dataset
- [`train_fold.py`](./proemb/train_fold.py) for the Fold recognition dataset
- [`train_flip.py`](./proemb/train_flip.py) for the FLIP benchmark datasets

If you want to run these experiments on a cluster, take a look in the folder: [`iridis-scripts`](./proemb/iridis-scripts)


## Reproducing plots from the paper
To reproduce the plots for the amino acid embedding projection using PCA, use the notebook [`aa_embeddings.ipynb`](./proemb/media/aa_embeddings.ipynb).
For experiments involving protein structure prediction using reduced amino acid alphabets, use the notebook [`esm-structure-prediction.ipynb`](./proemb/media/esm-structure-prediction.ipynb).
This notebook contains code for generating the structures with ESMFold and everything else needed to recreate the results.

For more information on the steps taken to create the WASS13 alphabet, take a look at: [`surface_plots.ipynb`](./proemb/media/surface_plots.ipynb)



#### This code contains various bits of code taken from other sources. If you find the repo useful, please cite the following work too:

- Surface generation code: [MASIF](https://github.com/LPDI-EPFL/masif)
- LDDT calculation: [AlphaFold](https://github.com/deepmind/alphafold)
- Model archiecture and uniprot tokenization: [Prose](https://github.com/tbepler/prose)
- MSA plot generation [ColabFold](https://github.com/sokrypton/ColabFold)

## Authors
Ioan Ieremie, Rob M. Ewing, Mahesan Niranjan

## Citation
```
to be added
```

## Contact
ii1g17 [at] soton [dot] ac [dot] uk
