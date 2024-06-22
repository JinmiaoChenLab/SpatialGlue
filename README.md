# Deciphering spatial domains from spatial multi-omics with SpatialGlue 
This repository contains SpatialGlue script and jupyter notebooks essential for reproducing the benchmarking outcomes shown in the paper. We provide experimental data in each case with details available within the notebook. All experiment data is reproducible through the provided scripts of various methods (https://github.com/JinmiaoChenLab/SpatialGlue_notebook). Some notebooks will be uploaded shortly to complement the available resources. 

[![DOI](https://zenodo.org/badge/631763850.svg)](https://zenodo.org/badge/latestdoi/631763850)

![](https://github.com/JinmiaoChenLab/SpatialGlue/blob/main/Workflow.jpg)

## Overview
Integration of multiple data modalities in a spatially informed manner remains an unmet need for exploiting spatial multi-omics data. Here, we introduce SpatialGlue, a novel graph neural network with dual-attention mechanism, to decipher spatial domains by intra-omics integration of spatial location and omics measurement followed by cross-omics integration. We demonstrate that SpatialGlue can more accurately resolve spatial domains at a higher resolution across different tissue types and technology platforms, to enable biological insights into cross-modality spatial correlations. SpatialGlue is computation resource efficient and can be applied for data from various spatial multi-omics technological platforms, including Spatial-epigenome-transcriptome, Stereo-CITE-seq, SPOTS, and 10x Visium. Next, we will extend SpatialGlue to more platforms, such as 10x Genomics Xenium and Nanostring CosMx. 

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
* scikit-misc==0.2.0
* tqdm==4.64.0
* matplotlib==3.4.2
* R==4.0.3

## Tutorial
For the step-by-step tutorial, please refer to:
[https://spatialglue-tutorials.readthedocs.io/en/latest/](https://spatialglue-tutorials.readthedocs.io/en/latest/)

## Data
In this paper, we tested 5 simulation datasets and 12 experimental datasets including 2 mouse spleen datasets acquired with SPOTS (Ben-Chetrit et al., 2023), 4 mouse thymus datasets from Stereo-CITE-seq (unpublished), and 4 mouse brain spatial-epigenome-transcriptome datasets (Zhang et al. 2023), and 2 in-house human lymph node datasets acquired with 10x Visium CytAssist. The SPOTS mouse spleen data was obtained from the GEO repository (accession no. GSE198353, https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE198353), the Stereo-CITE-seq mouse thymus data from BGI and the spatial-epigenome-transcriptome mouse brain data from AtlasXplore (https://web.atlasxomics.com/visualization/Fan). The details of all datasets used are available in the Methods section. The data used as input to the methods tested in this study, inclusive of the Stereo-CITE-seq and the in-house human lymph node data have been uploaded to Zenodo and is freely available at https://zenodo.org/records/10362607.

## Benchmarking and notebooks
In the paper, we compared SpatialGlue with 7 state-of-the-art single-cell multi-omics integration methods, including Seurat, totalVI, MultiVI, MOFA+, MEFISTO, scMM, and StabMap. Jupyter
notebooks covering the benchmarking analysis in this paper are available at https://github.com/JinmiaoChenLab/SpatialGlue/tree/main/notebooks.

## Citation
[Yahui Long](https://longyahui.github.io/Home//) et al. Deciphering spatial domains from spatial multi-omics with SpatialGlue. **Nature Methods**. 2024.
