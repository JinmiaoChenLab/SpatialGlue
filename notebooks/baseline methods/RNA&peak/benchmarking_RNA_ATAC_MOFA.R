reticulate::use_condaenv('seurat_lyh', required=TRUE)

library(data.table)
library(purrr)
library(Seurat)
library(ggplot2)
library(gridExtra)
library(Signac)
library(MOFA2)

setwd('../data/Dataset9_Mouse_Brain3/')

# rna
#Convert("adata_RNA_seurat.h5ad", "adata_RNA.h5seurat", overwrite = TRUE)
seurat_rna <- LoadH5Seurat("adata_RNA.h5seurat")

seurat_rna <- CreateSeuratObject(counts = seurat_rna@assays$RNA@counts, project = "rna")
seurat_rna[["percent.mt"]] <- PercentageFeatureSet(seurat_rna, pattern = "^mt-")

# atac
#Convert("adata_peaks_seurat.h5ad", "adata_peaks.h5seurat", overwrite = TRUE)
seurat_atac <- LoadH5Seurat("adata_peaks.h5seurat")

seurat_atac <- CreateSeuratObject(counts = seurat_atac@assays$RNA@counts, project = 'atac')

# RNA
seurat_rna <- FindVariableFeatures(seurat_rna, 
                                   selection.method = "vst", 
                                   nfeatures = 2500
)
rna.features <- seurat_rna@assays$RNA@var.features

# ATAC
seurat_atac <- FindVariableFeatures(seurat_atac, 
                                    selection.method = "disp", 
                                    nfeatures = 5000
)
#atac.features <- seurat_atac@assays$peaks@var.features
atac.features <- seurat_atac@assays$RNA@var.features

# creat MOFA object
#mofa <- create_mofa(list(
#  "RNA" = as.matrix( seurat_rna@assays$RNA@data[rna.features,] ),
#  "ATAC" = as.matrix( seurat_atac@assays$peaks@data[atac.features,] )
#))

mofa <- create_mofa(list(
  "RNA" = as.matrix( seurat_rna@assays$RNA@data[rna.features,] ),
  "ATAC" = as.matrix( seurat_atac@assays$RNA@data[atac.features,] )
))

mofa

# Model options: let's use only 10 factors, should be enough to distinguish the four cell lines.
model_opts <- get_default_model_options(mofa)
model_opts$num_factors <- 10

# Training options: let's use default options
train_opts <- get_default_training_options(mofa)
train_opts$seed <- 42

# Prepare the MOFA object
mofa <- prepare_mofa(
  object = mofa,
  model_options = model_opts,
  training_options = train_opts
)

# Run MOFA
mofa <- run_mofa(mofa)

cluster <- cluster_samples(mofa, k=16, factors="all")
clusters <- cluster$cluster

samples_metadata(mofa) <- as.data.frame(clusters) %>% tibble::rownames_to_column("sample") %>% as.data.table
#write.csv(mofa@samples_metadata, file='MOFA_metadata.csv', sep='\t', row.names = FALSE)

#saveRDS(mofa,file = 'mofa.rds')
