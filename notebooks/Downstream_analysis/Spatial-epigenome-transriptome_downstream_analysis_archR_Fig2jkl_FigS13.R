
library(ArchR)
library(Seurat)
library(patchwork)
library(gridExtra)
library(dplyr)
library(tibble)
library(clusterProfiler)
library(org.Mm.eg.db)
library(repr)
library(purrr)
library(presto)
library(ggseqlogo)
library(chromVARmotifs)

setwd('../spatial_atac/Dataset9/')

data_species <- 'mm10'
num_threads <- 1
tile_size <- 5000
genomeSize = 3.0e+09
min_TSS <- 0
min_Frags <- 0
set.seed(1234)
#Path to fragments.tsv.gz located in <sample>_cellranger_outs/
inputFile <- "../spatial_atac/test/GSM6704979_MouseBrain_20um_100barcodes_H3K27ac_fragments.tsv.gz"
project_name <- 'Mouse_Brain'
# Path to spatial folder located in <sample>
spatialFolder <- '../spatial_atac/test/MouseBrain_20um_100barcodes_H3K27ac_spatial/'

addArchRGenome(data_species)
geneAnnotation <- getGeneAnnotation()
genomeAnnotation <- getGenomeAnnotation()
addArchRThreads(threads = num_threads)

ArrowFiles <- createArrowFiles(
  inputFiles = inputFile,
  sampleNames = project_name,
  geneAnnotation = geneAnnotation,
  genomeAnnotation = genomeAnnotation,
  minTSS = min_TSS,
  minFrags = min_Frags,
  maxFrags = 1e+07,
  addTileMat = TRUE,
  addGeneScoreMat = TRUE,
  offsetPlus = 0,
  offsetMinus = 0,
  force = TRUE,
  TileMatParams = list(tileSize = tile_size)
)

proj <- ArchRProject(
  ArrowFiles = ArrowFiles,
  outputDirectory = project_name,
  geneAnnotation = geneAnnotation,
  genomeAnnotation = genomeAnnotation,
  copyArrows = TRUE
)

############### Prepare meta.data
meta.data <- as.data.frame(getCellColData(ArchRProj = proj))
meta.data['cellID_archr'] <- row.names(meta.data)
new_row_names <- row.names(meta.data)
new_row_names <- unlist(lapply(new_row_names, function(x) gsub(".*#","", x)))
new_row_names <- unlist(lapply(new_row_names, function(x) gsub("-.*","", x)))
row.names(meta.data) <- new_row_names

############### Filtering off-tissue tixels using image data
image = Read10X_Image(image.dir = spatialFolder, filter.matrix = TRUE)
sequenced_tixels <- row.names(meta.data)
image <- image[sequenced_tixels, ]
meta.data.spatial <- meta.data[row.names(image@coordinates), ]
proj_in_tissue <- proj[meta.data.spatial$cellID_archr, ]

############### Dimension reduction, clustering, and add UMAP embedding
proj_in_tissue <- addIterativeLSI(
  ArchRProj = proj_in_tissue,
  useMatrix = "TileMatrix",
  name = "IterativeLSI",
  iterations = 2,
  clusterParams = list(
    resolution = c(0.2),
    sampleCells = 10000,
    n.start = 10
  ),
  varFeatures = 25000,
  dimsToUse = 1:30,
  force = TRUE
)

proj_in_tissue <- addClusters(
  input = proj_in_tissue,
  reducedDims = "IterativeLSI",
  method = "Seurat",
  name = "Clusters",
  resolution = 0.5,
  force = TRUE
)

s <- read.csv('te1.csv', header = T, row.names = 1)
as.vector(s$SpatialGlue_new)-> g
as.vector(rownames(s))-> g1
proj_in_tissue1 <- ArchR::addCellColData(ArchRProj = proj_in_tissue, data = g, name = "SpatialGlue", cells = g1)
idxPass <- which(proj_in_tissue1b$SpatialGlue != "NA")
cellsPass <- proj_in_tissue1b$cellNames[idxPass]
proj_in_tissue1b <- proj_in_tissue1b[cellsPass, ]

cM <- confusionMatrix(paste0(proj_in_tissue1a$SpatialGlue), paste0(proj_in_tissue1a$Sample))
cM

library(pheatmap)
cM <- cM / Matrix::rowSums(cM)
p <- pheatmap::pheatmap(
  mat = as.matrix(cM), 
  color = paletteContinuous("whiteBlue"), 
  border_color = "black"
)
p

proj_in_tissue1a <- addUMAP(
  ArchRProj = proj_in_tissue1a,
  reducedDims = "IterativeLSI",
  name = "UMAP",
  nNeighbors = 30,
  minDist = 0.5,
  metric = "cosine",
  force = TRUE
)

p1 <- plotEmbedding(ArchRProj = proj_in_tissue1a, colorBy = "cellColData", name = "SpatialGlue", embedding = "UMAP")

############## Creating Seurat object
gene_score <- getMatrixFromProject(proj_in_tissue1a)
rownames(gene_score) <- rowData(gene_score)$name
proj_in_tissue1a <- addImputeWeights(proj_in_tissue1a)
gene_score <- imputeMatrix(assay(gene_score), getImputeWeights(proj_in_tissue1a))
gene_score <- log(gene_score+1, base = 2)
colnames(gene_score) <- gsub(pattern = paste0(project_name, "#|-1"), replacement = "", x= colnames(gene_score))

object <- CreateSeuratObject(counts = gene_score, assay = "Spatial", meta.data = meta.data)

image <- image[Cells(x = object)]
DefaultAssay(object = image) <- "Spatial"
object[["slice1"]] <- image
spatial_in_tissue.obj <- object

spatial_in_tissue.obj$orig.ident = as.factor(project_name)
Idents(spatial_in_tissue.obj) = 'SpatialGlue'
spatial_in_tissue.obj = AddMetaData(spatial_in_tissue.obj, spatial_in_tissue.obj@images$slice1@coordinates)

############## Define aesthetic parameters
n_clusters <- length(unique(proj_in_tissue1a$SpatialGlue))
palette  = c("navyblue", "turquoise2", "tomato", "tan2", "pink", "mediumpurple1", "steelblue", "springgreen2","violetred", "orange", "violetred", "slateblue1",  "violet", "purple",
             "purple3","blue2",  "pink", "coral2", "palevioletred", "red2", "yellowgreen", "palegreen4",
             "wheat2", "tan", "tan3", "brown",
             "grey70", "grey50", "grey30")
palette1  = c("navyblue", "tomato", "pink", "steelblue", "violetred", "orange",   "violet",
              "purple3",  "pink",  "palevioletred", "red2", 
              "wheat2", "tan", "brown", "grey30")
cols <- palette[seq_len(n_clusters)]
names(cols) <- names(proj_in_tissue1a@sampleMetadata)
names(cols) <- paste0('C', seq_len(n_clusters))
cols_hex <- lapply(X = cols, FUN = function(x){
  do.call(rgb, as.list(col2rgb(x)/255))
})
cols <- unlist(cols_hex)
pt_size_factor <- 1

############## Plotting UMAP/cluster identities to spatial histology
spatial_in_tissue.obj@meta.data$SpatialGlue = proj_in_tissue1a$SpatialGlue
plot_spatial = Seurat::SpatialDimPlot(spatial_in_tissue.obj, pt.size.factor = 1.3)
plot_spatial = Seurat::SpatialDimPlot(
  spatial_in_tissue.obj,
  label = FALSE, label.size = 3,
  group.by = "SpatialGlue",
  pt.size.factor = pt_size_factor, cols = cols, stroke = 2) +
  theme(
    plot.title = element_blank(),
    legend.position = "right",
    text=element_text(size=21)) +
  ggtitle(project_name) + theme(plot.title = element_text(hjust = 0.5), text=element_text(size=21))

plot_spatial$layers[[1]]$aes_params <- c(plot_spatial$layers[[1]]$aes_params, shape=22)

plot_umap = plotEmbedding(
  ArchRProj = proj_in_tissue1a,
  pal = cols,
  colorBy = "cellColData",
  name = "SpatialGlue",
  embedding = "UMAP",
  size = 2) +
  theme(
    plot.title = element_blank(),
    legend.position = "none",
    text=element_text(size=21))

cluster_plots <- plot_spatial + plot_umap
cluster_plots

spatial_in_tissue.obj@meta.data$log10_nFrags <- log10(spatial_in_tissue.obj@meta.data$nFrags)
plot_metadata = SpatialFeaturePlot(
  object = spatial_in_tissue.obj,
  features = c("log10_nFrags", "NucleosomeRatio", "TSSEnrichment"),
  alpha = c(0.2, 1), pt.size.factor = pt_size_factor) +
  theme(plot.title = element_text(hjust = 0.5), text=element_text(size=10))
plot_metadata$layers[[1]]$aes_params <-c(plot_metadata$layers[[1]]$aes_params, shape=22)

plot_metadata

proj_in_tissue1a <- addGroupCoverages(ArchRProj = proj_in_tissue1a, groupBy = "SpatialGlue")

pathToMacs2 <- findMacs2()
proj_in_tissue <- addReproduciblePeakSet(
  ArchRProj = proj_in_tissue1a,
  groupBy = "SpatialGlue",
  pathToMacs2 = pathToMacs2,
  genomeSize = genomeSize,
  force = TRUE
)

proj_in_tissue1a <- addGroupCoverages(ArchRProj = proj_in_tissue1a, groupBy = "SpatialGlue")

markersGS <- getMarkerFeatures(
  ArchRProj = proj_in_tissue1a, 
  useMatrix = "GeneScoreMatrix", 
  groupBy = "SpatialGlue",
  bias = c("TSSEnrichment", "log10(nFrags)"),
  testMethod = "wilcoxon"
)

markerList <- getMarkers(markersGS, cutOff = "FDR <= 0.01 & Log2FC >= 1")
markerList$C6



heatmapGS <- markerHeatmap(
  seMarker = markersGS, 
  cutOff = "FDR <= 0.05 & Log2FC >= 1",
  transpose = TRUE
)

ComplexHeatmap::draw(heatmapGS, heatmap_legend_side = "bot", annotation_legend_side = "bot")
plotPDF(heatmapGS, name = "Heatmap-Marker1", width = 16, height = 8, ArchRProj = proj_in_tissue1a, addDOC = FALSE)



p <- plotEmbedding(
  ArchRProj = proj_in_tissue, 
  colorBy = "GeneScoreMatrix", 
  name = markerList$C1$name[1:5], 
  embedding = "UMAP",
  quantCut = c(0.01, 0.95),
  imputeWeights = NULL
)

p <- plotEmbedding(
  ArchRProj = proj_in_tissue, 
  colorBy = "GeneScoreMatrix", 
  name = markerList$C1$name[1:5], 
  embedding = "UMAP",
  imputeWeights = getImputeWeights(proj_in_tissue)
)

p2 <- lapply(p, function(x){
  x + guides(color = FALSE, fill = FALSE) + 
    theme_ArchR(baseSize = 6.5) +
    theme(plot.margin = unit(c(0, 0, 0, 0), "cm")) +
    theme(
      axis.text.x=element_blank(), 
      axis.ticks.x=element_blank(), 
      axis.text.y=element_blank(), 
      axis.ticks.y=element_blank()
    )
})
do.call(cowplot::plot_grid, c(list(ncol = 3),p2))


plotPDF(plotList = p, 
        name = "Plot-UMAP-Marker-Genes-WO-Imputation.pdf", 
        ArchRProj = proj_in_tissue, 
        addDOC = FALSE, width = 10, height = 10)

plotPDF(plotList = p, 
        name = "Plot-UMAP-Marker-Genes-W-Imputation.pdf", 
        ArchRProj = proj_in_tissue, 
        addDOC = FALSE, width = 10, height = 10)

ArchRBrowser(proj_in_tissue)

pathToMacs2 <- findMacs2()
proj_in_tissue1a <- addReproduciblePeakSet(
  ArchRProj = proj_in_tissue1a,
  groupBy = "SpatialGlue",
  pathToMacs2 = pathToMacs2,
  genomeSize = genomeSize,
  force = TRUE
)

getPeakSet(proj_in_tissue1a)

proj_in_tissue1a <- addPeakMatrix(proj_in_tissue1a)
getAvailableMatrices(proj_in_tissue1a)

saveArchRProject(ArchRProj = proj_in_tissue, outputDirectory = "Save-Proj_in_tissue", load = FALSE)

p <- plotBrowserTrack(
  ArchRProj = proj_in_tissue, 
  groupBy = "SpatialGlue", 
  geneSymbol = markerList$C1$name[1:10], 
  upstream = 50000,
  downstream = 50000
)
p

grid::grid.newpage()
grid::grid.draw(p$CD14)

plotPDF(plotList = p, 
        name = "Plot-Tracks-Marker-Genes.pdf", 
        ArchRProj = proj_in_tissue, 
        addDOC = FALSE, width = 10, height = 10)

markerTest <- getMarkerFeatures(
  ArchRProj = proj_in_tissue1a, 
  useMatrix = "PeakMatrix",
  groupBy = "SpatialGlue",
  testMethod = "wilcoxon",
  bias = c("TSSEnrichment", "log10(nFrags)")
)
markerList <- getMarkers(markerTest, cutOff = "FDR <= 0.01 & Log2FC >= 1")
markerList

markerList <- getMarkers(markerTest, cutOff = "FDR <= 0.01 & Log2FC >= 1", returnGR = TRUE)
markerList

heatmapPeaks <- markerHeatmap(
  seMarker = markerTest, 
  cutOff = "FDR <= 0.1 & Log2FC >= 0.5",
  transpose = FALSE, nLabel = 3
)

draw(heatmapPeaks, heatmap_legend_side = "bot", annotation_legend_side = "bot")
plotPDF(heatmapPeaks, name = "Peak-Marker-Heatmap2a", width = 8, height = 6, ArchRProj = proj_in_tissue1a, addDOC = FALSE)

pma <- markerPlot(seMarker = markerTest, name = "1", cutOff = "FDR <= 0.1 & abs(Log2FC) >= 1", plotAs = "MA")
pma

proj_in_tissue1b <- addPeakMatrix(proj_in_tissue1a)
proj_in_tissue1a <- addMotifAnnotations(ArchRProj = proj_in_tissue1a, motifSet = "encode", name = "Motif", force = TRUE, species = getGenome(ArchRProj = proj_in_tissue1a))
motifsUp <- peakAnnoEnrichment(
  seMarker = markerTest,
  ArchRProj = proj_in_tissue1a,
  peakAnnotation = "Motif",
  cutOff = "FDR <= 0.1 & Log2FC >= 0.5"
)

df <- data.frame(TF = rownames(motifsUp), mlog10Padj = assay(motifsUp)[,1])
df <- df[order(df$mlog10Padj, decreasing = TRUE),]
df$rank <- seq_len(nrow(df))

ggUp <- ggplot(df, aes(rank, mlog10Padj, color = mlog10Padj)) + 
  geom_point(size = 1) +
  ggrepel::geom_label_repel(
    data = df[rev(seq_len(30)), ], aes(x = rank, y = mlog10Padj, label = TF), 
    size = 1.5,
    nudge_x = 2,
    color = "black"
  ) + theme_ArchR() + 
  ylab("-log10(P-adj) Motif Enrichment") + 
  xlab("Rank Sorted TFs Enriched") +
  scale_color_gradientn(colors = paletteContinuous(set = "comet"))

ggUp

proj_in_tissue <- addBgdPeaks(proj_in_tissue, force = TRUE)
proj_in_tissue <- addDeviationsMatrix(
  ArchRProj = proj_in_tissue,
  peakAnnotation = "Motif",
  force = TRUE
)
plotVarDev <- getVarDeviations(proj_in_tissue, name = "MotifMatrix", plot = TRUE)
plotPDF(plotVarDev, name = "Variable-Motif-Deviation-Scores", width = 5, height = 5, ArchRProj = proj_in_tissue, addDOC = FALSE)

motifs <- c("GATA_985", "GATA_986", "IRF_999", "IRF4_997", "SOX10_962", "CEBPA_959")
markersMotifs <- getMarkerFeatures(
  ArchRProj = proj_in_tissue,
  useMatrix = "MotifMatrix",
  groupBy = "SpatialGlue",
  bias = c("TSSEnrichment", "log10(nFrags)"),
  testMethod = "wilcoxon",
  useSeqnames = 'z'
)

markerMotifs <- getFeatures(proj_in_tissue, select = paste(motifs, collapse="|"), useMatrix = "MotifMatrix")

markerMotifs <- grep("z:", markerMotifs, value = TRUE)
markerMotifs <- markerMotifs[markerMotifs %ni% "z:SREBF1_22"]

p <- plotGroups(ArchRProj = proj_in_tissue, 
                groupBy = "Clusters", 
                colorBy = "MotifMatrix", 
                name = markerMotifs,
                imputeWeights = getImputeWeights(proj_in_tissue)
)

p2 <- lapply(p, function(x){
  x + guides(color = FALSE, fill = FALSE) + 
    theme_ArchR(baseSize = 6.5) +
    theme(plot.margin = unit(c(0, 0, 0, 0), "cm")) +
    theme(
      axis.text.x=element_blank(), 
      axis.ticks.x=element_blank(), 
      axis.text.y=element_blank(), 
      axis.ticks.y=element_blank()
    )
})
do.call(cowplot::plot_grid, c(list(ncol = 3),p2))

markerRNA <- getFeatures(proj_in_tissue, select = paste(motifs, collapse="|"), useMatrix = "GeneScoreMatrix")
markerRNA <- markerRNA[markerRNA %ni% c("SREBF1","CEBPA-DT")]
markerRNA

motifPositions <- getPositions(proj_in_tissue)

motifs <- c("GATA1", "CEBPA", "EBF1", "IRF4", "TBX21", "PAX5")
markerMotifs <- unlist(lapply(motifs, function(x) grep(x, names(motifPositions), value = TRUE)))
markerMotifs <- markerMotifs[markerMotifs %ni% "SREBF1_22"]
markerMotifs

#proj_in_tissue <- addGroupCoverages(ArchRProj = proj_in_tissue, groupBy = "Clusters")

seFoot <- getFootprints(
  ArchRProj = proj_in_tissue, 
  positions = motifPositions[markerMotifs], 
  groupBy = "Clusters"
)

plotFootprints(
  seFoot = seFoot,
  ArchRProj = proj_in_tissue, 
  normMethod = "Subtract",
  plotName = "Footprints-Subtract-Bias",
  addDOC = FALSE,
  smoothWindow = 5
)

plotFootprints(
  seFoot = seFoot,
  ArchRProj = proj_in_tissue, 
  normMethod = "Divide",
  plotName = "Footprints-Divide-Bias",
  addDOC = FALSE,
  smoothWindow = 5
)

plotFootprints(
  seFoot = seFoot,
  ArchRProj = proj_in_tissue, 
  normMethod = "None",
  plotName = "Footprints-No-Normalization",
  addDOC = FALSE,
  smoothWindow = 5
)

seTSS <- getFootprints(
  ArchRProj = proj_in_tissue, 
  positions = GRangesList(TSS = getTSS(proj_in_tissue)), 
  groupBy = "Clusters",
  flank = 2000
)

plotFootprints(
  seFoot = seTSS,
  ArchRProj = proj_in_tissue, 
  normMethod = "None",
  plotName = "TSS-No-Normalization",
  addDOC = FALSE,
  flank = 2000,
  flankNorm = 100
)

proj_in_tissue1a <- addCoAccessibility(
  ArchRProj = proj_in_tissue1a,
  reducedDims = "IterativeLSI"
)

cA <- getCoAccessibility(
  ArchRProj = proj_in_tissue1a,
  corCutOff = 0.5,
  resolution = 1,
  returnLoops = T
)

cA

plotBrowserTrack(
  ArchRProj = proj_in_tissue,
  groupBy = "Clusters",
  geneSymbol = markerList$C1$name[1:20],
  upstream = 50000,
  downstream = 50000,
  loops = getCoAccessibility(proj_in_tissue)
)
markerList$C1$name[1:20]
markerList$C1
markerList <- getMarkers(markersGS, cutOff = "FDR <= 0.01 & Log2FC >= 1.25")
markerList$C1$name
plotBrowserTrack(
  ArchRProj = proj_in_tissue,
  groupBy = "Clusters",
  geneSymbol = markerList$C1$name[1:20],
  upstream = 50000,
  downstream = 50000,
  loops = getCoAccessibility(proj_in_tissue)
)
P <- plotBrowserTrack(
  ArchRProj = proj_in_tissue,
  groupBy = "Clusters",
  geneSymbol = markerList$C1$name[1:20],
  upstream = 50000,
  downstream = 50000,
  loops = getCoAccessibility(proj_in_tissue)
)

plotPDF(plotList = P,
        name = "Plot-Tracks-Marker-Genes-with-CoAccessibility.pdf",
        ArchRProj = proj_in_tissue,
        addDOC = FALSE, width = 5, height = 5)

proj_in_tissue1c <- addPeak2GeneLinks(
  ArchRProj = proj_in_tissue1b,
  reducedDims = "IterativeLSI", useMatrix = "GeneScoreMatrix",
)

p2g <- getPeak2GeneLinks(
  ArchRProj = proj_in_tissue1c,
  corCutOff = 0.45,
  resolution = 1000,
  returnLoops = TRUE
)

p2g[[1]]

markerGenes  <- c("Reln","Cux1","Cux2","Rorb","Scnn1a","Etv1","Fezf2","Ctip2","Tle4","Foxp2","Ntsr1","Pou3f2","Bcl11b","Tspan2")

p <- plotBrowserTrack(
  ArchRProj = proj_in_tissue1c, 
  groupBy = "SpatialGlue_new", 
  geneSymbol = markerGenes, 
  upstream = 50000,
  downstream = 50000, 
  loops = getPeak2GeneLinks(proj_in_tissue1c)
)

p <- plotBrowserTrack(
  ArchRProj = proj_in_tissue, 
  groupBy = "Clusters", 
  geneSymbol = markerList$C1$name[1:10], 
  upstream = 50000,
  downstream = 50000,
  loops = getPeak2GeneLinks(proj_in_tissue)
)

plotPDF(p[[11]], name = "Plot-Tracks-Marker-Genes-Pou3f2.pdf", width = 5, height = 5, ArchRProj = proj_in_tissue1c, addDOC = FALSE)
plotPDF(p[[2]], name = "Plot-Tracks-Marker-Genes-Cux1.pdf", width = 5, height = 5, ArchRProj = proj_in_tissue1c, addDOC = FALSE)
plotPDF(p[[3]], name = "Plot-Tracks-Marker-Genes-Cux2.pdf", width = 5, height = 5, ArchRProj = proj_in_tissue1c, addDOC = FALSE)
plotPDF(p[[7]], name = "Plot-Tracks-Marker-Genes-Fezf2.pdf", width = 5, height = 5, ArchRProj = proj_in_tissue1c, addDOC = FALSE)
plotPDF(p[[4]], name = "Plot-Tracks-Marker-Genes-Rorb.pdf", width = 5, height = 5, ArchRProj = proj_in_tissue1c, addDOC = FALSE)
plotPDF(p[[8]], name = "Plot-Tracks-Marker-Genes-Tle4.pdf", width = 5, height = 5, ArchRProj = proj_in_tissue1c, addDOC = FALSE)


plotPDF(plotList = p, 
        name = "Plot-Tracks-Marker-Genes-with-Peak2GeneLinks.pdf", 
        ArchRProj = proj_in_tissue1c, 
        addDOC = FALSE, width = 5, height = 5)

p <- plotPeak2GeneHeatmap(ArchRProj = proj_in_tissue1b, groupBy = "SpatialGlue_new")

seGroupMotif <- getGroupSE(ArchRProj = proj_in_tissue, useMatrix = "MotifMatrix", groupBy = "Clusters")
seGroupMotif

seZ <- seGroupMotif[rowData(seGroupMotif)$seqnames=="z",]

rowData(seZ)$maxDelta <- lapply(seq_len(ncol(seZ)), function(x){
  rowMaxs(assay(seZ) - assay(seZ)[,x])
}) %>% Reduce("cbind", .) %>% rowMaxs

corGSM_MM <- correlateMatrices(
  ArchRProj = proj_in_tissue,
  useMatrix1 = "GeneScoreMatrix",
  useMatrix2 = "MotifMatrix",
  reducedDims = "IterativeLSI"
)

corGSM_MM

corGSM_MM$maxDelta <- rowData(seZ)[match(corGSM_MM$MotifMatrix_name, rowData(seZ)$name), "maxDelta"]

corGSM_MM <- corGSM_MM[order(abs(corGSM_MM$cor), decreasing = TRUE), ]
corGSM_MM <- corGSM_MM[which(!duplicated(gsub("\\-.*","",corGSM_MM[,"MotifMatrix_name"]))), ]
corGSM_MM$TFRegulator <- "NO"
corGSM_MM$TFRegulator[which(corGSM_MM$cor > 0.5 & corGSM_MM$padj < 0.01 & corGSM_MM$maxDelta > quantile(corGSM_MM$maxDelta, 0.75))] <- "YES"
sort(corGSM_MM[corGSM_MM$TFRegulator=="YES",1])

p <- ggplot(data.frame(corGSM_MM), aes(cor, maxDelta, color = TFRegulator)) +
  geom_point() + 
  theme_ArchR() +
  geom_vline(xintercept = 0, lty = "dashed") + 
  scale_color_manual(values = c("NO"="darkgrey", "YES"="firebrick3")) +
  xlab("Correlation To Gene Score") +
  ylab("Max TF Motif Delta") +
  scale_y_continuous(
    expand = c(0,0), 
    limits = c(0, max(corGSM_MM$maxDelta)*1.05)
  )

p

p1 <- plotEmbedding(ArchRProj = proj_in_tissue, colorBy = "cellColData", name = "Clusters", embedding = "UMAP")

trajectory <- c("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10")
trajectory

proj_in_tissue <- addTrajectory(
  ArchRProj = proj_in_tissue, 
  name = "MyeloidU", 
  groupBy = "Clusters",
  trajectory = trajectory, 
  embedding = "UMAP", 
  force = TRUE
)

p2 <- lapply(seq_along(p), function(x){
  if(x != 1){
    p[[x]] + guides(color = FALSE, fill = FALSE) + 
      theme_ArchR(baseSize = 6) +
      theme(plot.margin = unit(c(0.1, 0.1, 0.1, 0.1), "cm")) +
      theme(
        axis.text.y=element_blank(), 
        axis.ticks.y=element_blank(),
        axis.title.y=element_blank()
      ) + ylab("")
  }else{
    p[[x]] + guides(color = FALSE, fill = FALSE) + 
      theme_ArchR(baseSize = 6) +
      theme(plot.margin = unit(c(0.1, 0.1, 0.1, 0.1), "cm")) +
      theme(
        axis.ticks.y=element_blank(),
        axis.title.y=element_blank()
      ) + ylab("")
  }
})
do.call(cowplot::plot_grid, c(list(nrow = 1, rel_widths = c(2, rep(1, length(p2) - 1))),p2))

plotPDF(p, name = "Plot-Groups-Deviations-w-Imputation", width = 10, height = 10, ArchRProj = proj_in_tissue, addDOC = FALSE)

p <- plotEmbedding(
  ArchRProj = proj_in_tissue, 
  colorBy = "MotifMatrix", 
  name = sort(markerMotifs), 
  embedding = "UMAP",
  imputeWeights = getImputeWeights(proj_in_tissue)
)

saveRDS(proj_in_tissue, paste0(project_name, "_spatial_markerMotifs.rds"))

table(proj_in_tissue$Clusters)

markersPeaks <- getMarkerFeatures(
  ArchRProj = proj_in_tissue1a, 
  useMatrix = "PeakMatrix", 
  groupBy = "Clusters",
  bias = c("TSSEnrichment", "log10(nFrags)"),
  testMethod = "wilcoxon"
)

markersPeaks

markerList <- getMarkers(markersPeaks, cutOff = "FDR <= 0.05 & Log2FC >= 1", returnGR = TRUE)
markerList

markerList$C1

heatmapPeaks <- markerHeatmap(
  seMarker = markersPeaks, 
  cutOff = "FDR <= 0.1 & Log2FC >= 1",
  transpose = TRUE
)

draw(heatmapPeaks, heatmap_legend_side = "bot", annotation_legend_side = "bot")

plotPDF(heatmapPeaks, name = "Peak-Marker-Heatmap", width = 14, height = 6, ArchRProj = proj_in_tissue, addDOC = FALSE)

enrichMotifs <- peakAnnoEnrichment(
  seMarker = markersPeaks,
  ArchRProj = proj_in_tissue,
  peakAnnotation = "Motif",
  cutOff = "FDR <= 0.1 & Log2FC >= 0.5"
)

heatmapEM <- plotEnrichHeatmap(enrichMotifs, n = 7, transpose = TRUE)
ComplexHeatmap::draw(heatmapEM, heatmap_legend_side = "bot", annotation_legend_side = "bot")

proj_in_tissue1a <- addArchRAnnotations(ArchRProj = proj_in_tissue1a, collection = "Codex", db = "LOLA")


enrichEncode <- peakAnnoEnrichment(
  seMarker = markerTest,
  ArchRProj = proj_in_tissue1a,
  peakAnnotation = "Codex")

enrichEncode

heatmapEncode <- plotEnrichHeatmap(enrichEncode, n = 30, transpose = TRUE)

ComplexHeatmap::draw(heatmapEncode, heatmap_legend_side = "bot", annotation_legend_side = "bot")

plotPDF(heatmapEncode, name = "EncodeTFBS-Enriched-Marker-Heatmap", width = 14, height = 12, ArchRProj = proj_in_tissue1a, addDOC = FALSE)

plotPDF(heatmapEM, name = "Motifs-Enriched-Marker-Heatmap", width = 8, height = 6, ArchRProj = proj_in_tissue, addDOC = FALSE)

ComplexHeatmap::draw(heatmapEncode, heatmap_legend_side = "bot", annotation_legend_side = "bot")

plotPDF(heatmapEncode, name = "Codex-Enriched-Marker-Heatmap", width = 14, height = 12, ArchRProj = proj_in_tissue1a, addDOC = FALSE)

pma <- markerPlot(seMarker = markersPeaks, name = "C1", cutOff = "FDR <= 0.05 & Log2FC >= 1", plotAs = "MA")
pma

pv <- markerPlot(seMarker = markersPeaks, name = "C1", cutOff = "FDR <= 0.05 & Log2FC >= 1", plotAs = "Volcano")
pv

p <- plotBrowserTrack(
  ArchRProj = proj_in_tissue, 
  groupBy = "Clusters2", 
  geneSymbol = c("Xlrp1"),
  features =  getMarkers(markersPeaks, cutOff = "FDR <= 0.1 & Log2FC >= 1", returnGR = TRUE)["C1"],
  upstream = 50000,
  downstream = 50000
)

plotPDF(p, name = "Plot-Tracks-With-Features", width = 10, height = 10, ArchRProj = proj_in_tissue, addDOC = FALSE)



