library(sceasy)
library(Seurat)
library(dplyr)
library(SeuratDisk)
library(anndata)

setwd("../data/Dataset1_Mouse_Spleen1/")

Convert("adata_RNA_seurat.h5ad", "adata_RNA.h5seurat", overwrite = TRUE)
rna <- LoadH5Seurat("adata_RNA.h5seurat")
Convert("adata_ADT_seurat.h5ad", "adata_ADT.h5seurat")
adt <- LoadH5Seurat("adata_protein.h5seurat")

rna <- CreateSeuratObject(counts = rna@assays$RNA@counts, project = "rna")
rna[["percent.mt"]] <- PercentageFeatureSet(rna, pattern = "^mt-")

rna <- NormalizeData(rna, normalization.method = "LogNormalize", scale.factor = 10000)
rna <- FindVariableFeatures(rna, selection.method = "vst", nfeatures = 2000)

all.genes <- rownames(rna)
rna <- ScaleData(rna, features = all.genes)
rna <- RunPCA(rna, features = VariableFeatures(object = rna))
rna <- FindNeighbors(rna, dims = 1:100)
rna <- FindClusters(rna, resolution = 1.09)
DimPlot(rna, reduction = "umap", label=T)
rna <- RunUMAP(rna, dims = 1:100)
write.csv(rna@meta.data, 'rna_meta.csv')

adt <- CreateSeuratObject(counts = adt@assays$RNA@counts, project = "adt")
adt[["percent.mt"]] <- PercentageFeatureSet(adt, pattern = "^mt-")
adt <- NormalizeData(adt, normalization.method = "LogNormalize", scale.factor = 10000)
adt <- FindVariableFeatures(adt, selection.method = "vst", nfeatures = 2000)
all.genes <- rownames(adt)
adt <- ScaleData(adt, features = all.genes)
adt <- RunPCA(adt, features = VariableFeatures(object = adt))
adt <- FindNeighbors(adt, dims = 1:18)
adt <- FindClusters(adt, resolution = 1.0) 
adt <- RunUMAP(adt, dims = 1:10)
DimPlot(adt, reduction = "umap", label = T)
write.csv(adt@meta.data, 'adt_meta.csv')

rna[["ADT"]] <- CreateAssayObject(counts = adt@assays$RNA@counts)

DefaultAssay(rna) <- 'RNA'
rna <- NormalizeData(rna) %>% FindVariableFeatures() %>% ScaleData() %>% RunPCA()

DefaultAssay(rna) <- 'ADT'
# we will use all ADT features for dimensional reduction
# we set a dimensional reduction name to avoid overwriting the 
VariableFeatures(rna) <- rownames(rna[["ADT"]])
rna <- NormalizeData(rna, normalization.method = 'CLR', margin = 2) %>% 
  ScaleData() %>% RunPCA(reduction.name = 'apca')

rna <- FindMultiModalNeighbors(
  rna, reduction.list = list("pca", "apca"), 
  dims.list = list(1:30, 1:18), modality.weight.name = "RNA.weight"
)

rna <- RunUMAP(rna, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_")
rna <- FindClusters(rna, graph.name = "wsnn", algorithm = 3, resolution = 0.7, verbose = FALSE)
DimPlot(rna, reduction = 'wnn.umap', label = TRUE, repel = TRUE, label.size = 2.5)

write.csv(rna@meta.data, 'wnn_meta.csv')
saveRDS(rna,file = 'Seurat.rds')