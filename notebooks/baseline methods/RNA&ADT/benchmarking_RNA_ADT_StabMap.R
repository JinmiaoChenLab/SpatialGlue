library(devtools)
library(StabMap)
library(SingleCellMultiModal)
library(scran)
library(Seurat)
library(SeuratDisk)
library(MultiAssayExperiment)
library(Matrix)
library(rhdf5)
library(scater)
library(SummarizedExperiment)
library(zellkonverter)
library(scuttle)

# Display the data name we need
# Dataset1_Mouse_Spleen1
# Dataset2_Mouse_Spleen2
# Dataset3_Mouse_Thymus1
# Dataset4_Mouse_Thymus2
# Dataset5_Mouse_Thymus3
# Dataset6_Mouse_Thymus4
# Dataset7_Mouse_Brain1
# Dataset8_Mouse_Brain2
# Dataset9_Mouse_Brain3
# Dataset10_Mouse_Brain4
# Dataset11_Lymph_Node_A1
# Dataset12_Lymph_Node_D1


###################### Load data ##########################################################

setwd('../data')

genes <- read.csv('d7_genes.csv')
genes <- genes[which(genes$highly_variable=='True'),]
genes <- genes$X

RNA <- readH5AD(file = "Dataset1_Mouse_Spleen1/adata_RNA.h5ad")
ADT <- readH5AD(file = "Dataset1_Mouse_Spleen1/adata_ADT.h5ad")


###################### Reprocessing ###################################################

rna_data <- RNA
adt_data <- ADT
mae <- MultiAssayExperiment::MultiAssayExperiment(experiments = list(rna = rna_data, atac = adt_data))

sce.rna <- experiments(mae)[["rna"]]
# Normalization
names(assays(sce.rna)) <- 'counts'
sce.rna <- logNormCounts(sce.rna)
zero_count_cols <- which(colSums(counts(sce.rna)) == 0)

# Replace the first row of counts for these columns with a very small number
if (length(zero_count_cols) > 0) {
  counts(sce.rna)[1, zero_count_cols] <- 0.000000001
}

# Feature selection
hvgs <- genes
length(hvgs)
hvgs <- toupper(hvgs)
rownames(sce.rna) <- toupper(rownames(sce.rna))
valid_hvgs <- intersect(hvgs, rownames(sce.rna))
length(valid_hvgs)
sce.rna <- sce.rna[valid_hvgs,]

sce.atac <- experiments(mae)[["atac"]]
names(assays(sce.atac)) <- 'counts'
zero_count_cols <- which(colSums(counts(sce.atac)) == 0)

# Replace the first row of counts for these columns with a very small number
if (length(zero_count_cols) > 0) {
  counts(sce.atac)[1, zero_count_cols] <- 0.000000001
}
sce.atac <- logNormCounts(sce.atac)
decomp <- modelGeneVar(sce.atac)

# for 7-10
# hvgs <- rownames(decomp)[decomp$mean>1
#                         | decomp$p.value <= 0.0000001]
# length(hvgs)
# sce.atac <- sce.atac[hvgs,]

# find the intersection
sce.atac <- sce.atac[,intersect(colnames(sce.rna),colnames(sce.atac))]
sce.rna <- sce.rna[,intersect(colnames(sce.rna),colnames(sce.atac))]
logcounts_all = rbind(logcounts(sce.rna), logcounts(sce.atac))
dim(logcounts_all)
assayType = ifelse(rownames(logcounts_all) %in% rownames(sce.rna),
                   "rna", "atac")
table(assayType)



################ Indirect mosaic data integration with StabMap #####################################

dataTypeIndirect = setNames(sample(c("RNA", "Multiome", "ATAC"), ncol(logcounts_all),
                                   prob = c(0.3,0.3, 0.3), replace = TRUE),
                            colnames(logcounts_all))
table(dataTypeIndirect)
assay_list_indirect = list(
  RNA = logcounts_all[assayType %in% c("rna"), dataTypeIndirect %in% c("RNA")],
  Multiome = logcounts_all[assayType %in% c("rna", "atac"), dataTypeIndirect %in% c("Multiome")],
  ATAC = logcounts_all[assayType %in% c("atac"), dataTypeIndirect %in% c("ATAC")]
)

lapply(assay_list_indirect, dim)
lapply(assay_list_indirect, class)

mdt_indirect = mosaicDataTopology(assay_list_indirect)
mdt_indirect

stab_indirect = stabMap(assay_list_indirect,
                        reference_list = c("Multiome"),
                        plot = FALSE,
                        maxFeatures = 5000)
dim(stab_indirect)
stab_indirect[1:5,1:5]
write.csv(stab_indirect,'d14_stabmap.csv')



############################## Seurat clustering #######################################

minVal <- min(stab_indirect)
if (minVal < 0) {
  stab_indirect <- stab_indirect - minVal + 0.1
}
seurat_obj <- CreateSeuratObject(counts = t(stab_indirect))

seurat_obj <- NormalizeData(seurat_obj)
seurat_obj <- ScaleData(seurat_obj)
your_feature_list <- rownames(seurat_obj)
seurat_obj <- RunPCA(seurat_obj, features = your_feature_list)
seurat_obj <- FindNeighbors(seurat_obj)
seurat_obj <- FindClusters(seurat_obj, resolution = 1.78) #change the resolution
Idents(seurat_obj) <- seurat_obj$seurat_clusters
summary(as.factor(seurat_obj$seurat_clusters))

saveRDS(seurat_obj, file = 'd14_Clustering_result.rds')


