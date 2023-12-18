reticulate::use_condaenv('seurat_lyh', required=TRUE)
library(dplyr)
library(Seurat)
library(patchwork)
library(Signac)

setwd('../Mouse_Brain/Mouse_Brain_RNA_H3K27ac/')

P22mousebrain <- readRDS('P22mousebrain_spatial_RNA_H3K27ac.rds')

DefaultAssay(P22mousebrain) -> "Spatial"
P22mousebrain <- FindVariableFeatures(P22mousebrain, selection.method = "vst", nfeatures = 3000)
P22mousebrain <- SCTransform(P22mousebrain, assay = "Spatial", verbose = FALSE)
P22mousebrain <- RunPCA(P22mousebrain, assay = "SCT", verbose = FALSE)
P22mousebrain <- FindNeighbors(P22mousebrain, reduction = "pca", dims = 1:10, assay = "Spatial")
P22mousebrain <- FindClusters(P22mousebrain, verbose = FALSE, resolution = 1.35)   
P22mousebrain <- RunUMAP(P22mousebrain, reduction = "pca", dims = 1:10, assay = "Spatial")
SpatialDimPlot(P22mousebrain)
DimPlot(P22mousebrain, label = T)
write.csv(P22mousebrain@meta.data, 'rna_meta.csv')

DefaultAssay(P22mousebrain) <- "peaks"
P22mousebrain <- RunTFIDF(P22mousebrain, assay = "peaks")  #normalization
P22mousebrain <- FindTopFeatures(P22mousebrain, min.cutoff = 'q0', assay = "peaks")
P22mousebrain <- RunSVD(P22mousebrain, assay = "peaks")

P22mousebrain <- RunUMAP(object = P22mousebrain, reduction = 'lsi', dims = 2:10)
P22mousebrain <- FindNeighbors(object = P22mousebrain, reduction = 'lsi', dims = 2:10)
P22mousebrain <- FindClusters(object = P22mousebrain, verbose = FALSE, resolution = 0.8) 
DimPlot(object = P22mousebrain, label = TRUE) 
write.csv(P22mousebrain@meta.data, 'atac_meta.csv')
SpatialDimPlot(P22mousebrain)

P22mousebrain <- FindMultiModalNeighbors(P22mousebrain, reduction.list = list("pca", "lsi"), dims.list = list(1:10, 2:10))
P22mousebrain <- RunUMAP(P22mousebrain, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_")
P22mousebrain <- FindClusters(P22mousebrain, graph.name = "wsnn", verbose = FALSE, resolution = 0.7)
write.csv(P22mousebrain@meta.data, 'wnn_meta.csv')

#p1 <- DimPlot(P22mousebrain, reduction = "umap.rna", label = T) + ggtitle("RNA")
#p2 <- DimPlot(P22mousebrain , reduction = "umap.atac", label = T) + ggtitle("ATAC")
#p3 <- DimPlot(P22mousebrain , reduction = "wnn.umap", label = T) + ggtitle("WNN")
#p1 + p2 + p3 & NoLegend() & theme(plot.title = element_text(hjust = 0.5))
DimPlot(P22mousebrain , reduction = "wnn.umap", label = T) #+ ggtitle("WNN")