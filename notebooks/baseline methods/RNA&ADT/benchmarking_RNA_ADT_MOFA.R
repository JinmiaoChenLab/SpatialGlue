reticulate::use_condaenv('seurat_lyh', required=TRUE)

library(sceasy)
library(Seurat)
library(dplyr)
library(SeuratDisk)
library(Matrix)
library(devtools)
library(data.table)
library(MOFA2)

setwd('../data/Dataset1_Mouse_Spleen1/')

# rna
#Convert("adata_RNA_MOFA.h5ad", "adata_RNA.h5seurat", overwrite = TRUE)  
rna <- LoadH5Seurat("adata_RNA.h5seurat")

rna <- CreateSeuratObject(counts = rna@assays$RNA@counts, project = "rna")
rna[["percent.mt"]] <- PercentageFeatureSet(rna, pattern = "^mt-")

# protein
#Convert("adata_protein_MOFA.h5ad", "adata_protein.h5seurat", overwrite = TRUE)
adt <- LoadH5Seurat("adata_Pro.h5seurat")

cite1 <- CreateAssayObject(counts = adt@assays$RNA@counts)
rna[["ADT"]] <- cite1

rna <- NormalizeData(rna, normalization.method = "LogNormalize", assay = "RNA")
rna <- ScaleData(rna, do.center = TRUE, do.scale = FALSE, assay = "RNA")
#rna <- NormalizeData(rna, normalization.method = "LogNormalize", assay = "ADT")
rna <- NormalizeData(rna, normalization.method = "CLR", assay = "ADT")
rna <- ScaleData(rna, do.center = TRUE, do.scale = FALSE, assay = "ADT")
rna <- FindVariableFeatures(rna, selection.method = "vst", nfeatures = 2000, assay = "RNA", verbose = FALSE)
rna <- FindVariableFeatures(rna, selection.method = "vst", nfeatures = 2000, assay = "ADT", verbose = FALSE)

mofa <- create_mofa(rna, assays = c("RNA","ADT"))
mofa

plot_data_overview(mofa)

model_opts <- get_default_model_options(mofa)
model_opts$num_factors <- 10
mofa <- prepare_mofa(mofa, model_options = model_opts)

# run mofa

mofa <- run_mofa(mofa)

plot_factor_cor(mofa)
plot_variance_explained(mofa, max_r2=15)
plot_variance_explained(mofa, plot_total = T)[[2]]

cluster <- cluster_samples(mofa, k=5, factors="all")
clusters <- cluster$cluster

#samples_metadata(mofa) <- as.data.frame(clusters) %>% tibble::rownames_to_column("sample") %>% as.data.table
#rite.csv(mofa@samples_metadata, file='MOFA_metadata_5clusters.csv', sep='\t', row.names = FALSE)

#saveRDS(mofa,file = 'mofa.rds')
#latent <- mofa@expectations$Z$group1
#df_latent <- as.data.frame(latent)
#write.csv(df_latent, 'latent.csv')
