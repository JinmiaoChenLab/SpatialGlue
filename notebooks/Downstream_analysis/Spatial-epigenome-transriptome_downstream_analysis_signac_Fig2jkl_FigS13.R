library(reticulate)
library(data.table)
library(purrr)
library(SeuratDisk)
library(Seurat)
library(ggplot2)
library(gridExtra)
library(MOFA2)

setwd('../spatial_atac/')

so <- readRDS('../Spatial/scMultiomics/data/Dataset9_Mouse_Brain3/P22mousebrain_spatial_RNA_H3K27ac.rds')
meta <- read.csv('obs_dataset9.csv', header = T, row.names = 1)
so1 <- AddMetaData(so, metadata = meta)

so1


data9 = readH5AD("data/Dataset9_Mouse_Brain3/adata_combined_new.h5ad")

Convert("adata_new.h5ad", dest = "h5seurat", overwrite = TRUE)
data <- LoadH5Seurat("adata_new.h5seurat")

Convert("rna_data9.h5ad", dest = "h5seurat", overwrite = TRUE)
rna <- LoadH5Seurat("rna_data9.h5seurat")

meta <- read.csv('obs_dataset9.csv', header = T, row.names = 1)
so1 <- AddMetaData(so, metadata = meta)

so1a <- subset(so1, subset = label_new_combined != 'NA')

rna_markers <- FindAllMarkers(so1a, logfc.threshold = 0.25, min.pct = 0.25, assay = "SCT", test.use = "wilcox")
rna_markers %>% group_by(cluster) %>% top_n(n = 10, wt = avg_log2FC) -> top10
pdf('Heatmap_DEGs_all.pdf', height = 12, width = 15)
r <- DoHeatmap(so1a, features = top10$gene, size = 4, slot = "scale.data") + theme(axis.text.y = element_text(size = 7)) + scale_fill_gradientn(colors = c("blue", "white", "red")) 
print(r)
dev.off()

write.csv(rna_markers, 'DEGs_0.25_minpct_logFC.csv')
rna_markers_filtered <- subset(rna_markers, subset = p_val <= 0.05 & p_val_adj <= 0.05)
write.csv(rna_markers_filtered, 'DEGs_0.25_minpct_logFC_filtered.csv')
rna_markers_filtered1 <- subset(rna_markers, subset = p_val <= 0.05 & p_val_adj <= 0.05 & avg_log2FC > 0)
write.csv(rna_markers_filtered1, 'DEGs_0.25_minpct_logFC_filtered1.csv')

DefaultAssay(so1) <- "H3K27ac"

atac_markers <- FindAllMarkers(so1, logfc.threshold = 0.25, min.pct = 0.25, assay = "H3K27ac", test.use = "wilcox")
atac_markers %>% group_by(cluster) %>% top_n(n = 10, wt = avg_log2FC) -> top10
pdf('Heatmap_GAS_all.pdf', height = 12, width = 15)
r <- DoHeatmap(so1, features = top10$gene, size = 4, slot = "data") + theme(axis.text.y = element_text(size = 7)) + scale_fill_gradientn(colors = c("blue", "white", "red")) 
print(r)
dev.off()

pdf('FeaturePlot_GAS.pdf', height = 12, width = 15)
y <- SpatialFeaturePlot(so1a, features = c('Reln','Cux1','Cux2','Rorb','Scnn1a','Etv1','Fezf2','Tle4','Foxp2','Ntsr1','Pou3f2','Bcl11b','Tspan2'))
print(y)
dev.off()

SpatialFeaturePlot(so1a, features = c('Reln','Cux1','Cux2','Rorb','Scnn1a','Etv1','Fezf2','Ctip2','Tle4','Foxp2','Ntsr1','Pou3f2','Bcl11b','Tspan2'))

so1 <- AddMetaData(so, metadata = meta)
DefaultAssay(so1) <- "peaks"
so1 <- RunTFIDF(so1)
so1 <- FindTopFeatures(so1, min.cutoff = 'q5')
so1 <- RunSVD(so1)
all.genes <- rownames(so1)
so1 <- ScaleData(so1, features = all.genes)

so1a <- subset(so1, subset = label_new_combined != 'NA')
so1a$label_new_combined -> Idents(so1a)

peak_markers <- FindAllMarkers(so1a, logfc.threshold = 0.1, min.pct = 0.1, assay = "peaks", test.use = "wilcox")
peak_markers %>% group_by(cluster) %>% top_n(n = 10, wt = avg_log2FC) -> top10
pdf('Heatmap_DEPeaks_all4.pdf', height = 14, width = 14)
r <- DoHeatmap(so1a, features = top10$gene, size = 5, slot = "scale.data") + theme(axis.text.y = element_text(size = 7)) + scale_fill_gradientn(colors = c("lightblue", "white", "red"))
print(r)
dev.off()

write.csv(peak_markers, 'DEPeaks_minpct_logFC.csv')
peak_markers_filtered <- subset(peak_markers, subset = p_val <= 0.05 & p_val_adj <= 0.05)
write.csv(peak_markers_filtered, 'DEPeaks_minpct_logFC_filtered.csv')
peak_markers_filtered1 <- subset(peak_markers, subset = p_val <= 0.05 & p_val_adj <= 0.05 & avg_log2FC > 0)
write.csv(peak_markers_filtered1, 'DEPeaks_minpct_logFC_filtered1.csv')

#sceasy::convertFormat('data/Dataset9_Mouse_Brain3/adata_RNA.h5ad', from="anndata", to="seurat", outFile='rna_data9.rds')
