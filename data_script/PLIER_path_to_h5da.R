library(PLIER)
library(anndata)
library(dplyr)
PLIER::ListPriors()
# load PLIER pathways
data("canonicalPathways")
data("bloodCellMarkersIRISDMAP")
data("chemgenPathways")
data("immunePathways")
data("oncogenicPathways")
data("svmMarkers")
data("xCell")

setwd("~/OneDrive - The University Of British Columbia/Github/scCLR/data/pathways")

tmp = t(canonicalPathways)
ad <- AnnData(X = tmp)
ad$write_h5ad(filename = "canonicalPathways.h5ad")

tmp = t(bloodCellMarkersIRISDMAP)
ad <- AnnData(X = tmp)
ad$write_h5ad(filename = "bloodCellMarkersIRISDMAP.h5ad")

tmp = t(chemgenPathways)
ad <- AnnData(X = tmp)
ad$write_h5ad(filename = "chemgenPathways.h5ad")

tmp = t(immunePathways)
ad <- AnnData(X = tmp)
ad$write_h5ad(filename = "immunePathways.h5ad")

tmp = t(oncogenicPathways)
ad <- AnnData(X = tmp)
ad$write_h5ad(filename = "oncogenicPathways.h5ad")

tmp = t(svmMarkers)
ad <- AnnData(X = tmp)
ad$write_h5ad(filename = "svmMarkers.h5ad")

tmp = t(xCell)
ad <- AnnData(X = tmp)
ad$write_h5ad(filename = "xCell.h5ad")





