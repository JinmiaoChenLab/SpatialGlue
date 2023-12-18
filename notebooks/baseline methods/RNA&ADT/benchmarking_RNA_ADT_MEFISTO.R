reticulate::use_condaenv('seurat_lyh', required=TRUE)
library(data.table)
library(purrr)
library(Seurat)
library(ggplot2)
library(gridExtra)
library(MOFA2)

setwd('../data/Dataset1_Mouse_Spleen1/')

# rna
Convert("adata_RNA_seurat.h5ad", "adata_RNA.h5seurat", overwrite = TRUE)
seurat_rna <- LoadH5Seurat("adata_RNA.h5seurat")

seurat_rna <- CreateSeuratObject(counts = seurat_rna@assays$RNA@counts, project = "rna")
#seurat_rna[["percent.mt"]] <- PercentageFeatureSet(seurat_rna, pattern = "^mt-")

# adt
Convert("adata_Pro_seurat.h5ad", "adata_Pro.h5seurat", overwrite = TRUE)
seurat_adt <- LoadH5Seurat("adata_Pro.h5seurat")

seurat_adt <- CreateSeuratObject(counts = seurat_adt@assays$RNA@counts, project = "adt")

sample_metadata <- read.csv('sample_metadata.csv')

barcodes <- intersect(colnames(seurat_adt), colnames(seurat_rna))
seurat_adt <- seurat_adt[,barcodes]
seurat_rna <- seurat_rna[,barcodes]

# RNA
seurat_rna <- FindVariableFeatures(seurat_rna, 
                                   selection.method = "vst", 
                                   nfeatures = 2500
)
rna.features <- seurat_rna@assays$RNA@var.features

# ADT
#seurat_adt <- FindVariableFeatures(seurat_adt, 
#                                    selection.method = "disp", 
#                                    nfeatures = 18
#)
#adt.features <- seurat_adt@assays$RNA@var.features

adt.features <- seurat_adt@assays$RNA@counts

# creat mefisto object
#mefisto <- create_mofa(list(
#  "RNA" = as.matrix( seurat_rna@assays$RNA@data[rna.features,] ),
#  "ADT" = as.matrix( seurat_adt@assays$RNA@data[adt.features,] )
#))

mefisto <- create_mofa(list(
  "RNA" = as.matrix( seurat_rna@assays$RNA@data[rna.features,] ),
  "ADT" = as.matrix( seurat_adt@assays$RNA@counts )
))

mefisto

mefisto@samples_metadata$spatial1 <- sample_metadata$spatial1
mefisto@samples_metadata$spatial2 <- sample_metadata$spatial2

# set covariates
mefisto <- set_covariates(mefisto, c("spatial1","spatial2"))  # add UMAP as coordinates

# Model options: let's use only 10 factors
model_opts <- get_default_model_options(mefisto)
model_opts$num_factors <- 10  

# Training options: let's use default options
train_opts <- get_default_training_options(mefisto)
train_opts$seed <- 42

# Prepare the MOFA object
mefisto <- prepare_mofa(
  object = mefisto,
  model_options = model_opts,
  training_options = train_opts
)

# Run MOFA
mefisto <- run_mofa(mefisto)

cluster <- cluster_samples(mefisto, k=5, factors="all")
clusters <- cluster$cluster

samples_metadata(mefisto) <- as.data.frame(clusters) %>% tibble::rownames_to_column("sample") %>% as.data.table

#setwd('/home/yahui/anaconda3/work/SpatialGlue_revision/result/Dataset2_Mouse_Spleen2/MEFISTO/')
#write.csv(mefisto@samples_metadata, file='MEFISTO_metadata_5clusters.csv', sep='\t', row.names = FALSE)

#saveRDS(mefisto,file = 'mefisto.rds')
#latent <- mefisto@expectations$Z$group1
#df_latent <- as.data.frame(latent)
#write.csv(df_latent, 'latent.csv')
