# Integrated analysis of spatial multi-omics with SpatialGlue

[![DOI](https://zenodo.org/badge/631763850.svg)](https://zenodo.org/badge/latestdoi/631763850)

![](https://github.com/JinmiaoChenLab/GraphST/blob/main/SpatialGlue.png)

## Overview
SpatialGlue is a novel deep learning method for integrating spatial multi-omics data in a spatially informed manner. It utilizes a cycle graph neural network with a dual-attention mechanism to learn the significance of each modality at cross-omics and intra-omics integration. The method can accurately aggregate cell types or cell states at a higher resolution on different tissue types and technology platforms. Besides, it can provide interpretable insights into cross-modality spatial correlations. SpatialGlue is computationally efficient and it only requires about 5 mins for spatial multi-omics data at single-cell resolution (e.g., Spatial-ATAC-RNA-seq data, ~10,000 spots). 

## Requirements
You'll need to install the following packages in order to run the codes.
* python==3.8
* torch>=1.8.0
* cudnn>=10.2
* numpy==1.22.3
* scanpy==1.9.1
* anndata==0.8.0
* rpy2==3.4.1
* pandas==1.4.2
* scipy==1.8.1
* scikit-learn==1.1.1
* tqdm==4.64.0
* matplotlib==3.4.2
* R==4.0.3

## Tutorial
For the step-by-step tutorial, we would release it soon.

## Citation
Yahui Long, Kok Siong Ang, Sha Liao, Raman Sethi, Yang Heng, Chengwei Zhong, Hang Xu, Nazihah Husna, Min Jian, Lai Guan Ng, Ao Chen, Nicholas RJ Gascoigne, Xun Xu, Jinmiao Chen. Integrated analysis of spatial multi-omics with SpatialGlue. bioRxiv. 2023.
