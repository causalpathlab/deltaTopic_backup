library(ggplot2)
library(data.table)
library(gplots)
library(magrittr)

target = "delta"
fileDIR = "models/TotalDeltaETM_allgenes_ep2000_nlv4_bs512_combinebyadd/"

weights = fread(paste0(fileDIR, paste0(target,"_weights.csv")))
weights = weights %>% as.data.frame()
pdf(file = paste0(fileDIR, paste0(target,"_weights.pdf")))
weight_mat = weights[,2:5] %>% as.matrix()
rownames(weight_mat) = weights$V1
weight_mat %>%  gplots::heatmap.2()
dev.off()



K = 15 # top K genes and bottom K for each topic
ordered_weights_matrix = matrix(nrow = 2*K, ncol = ncol(weight_mat))
ordered_genes_matrix = matrix(nrow = 2*K, ncol = ncol(weight_mat))
ngenes = nrow(weight_mat)
#i = 0 
for(i in 1:4){
    print(i)
    weights_order = weights[weights[,paste0('topic_',i-1)] %>% order(decreasing = T),paste0('topic_',i-1)]
    genes_order = weights[weights[,paste0('topic_',i-1)] %>% order(decreasing = T),'V1']
    position = c(1:K, (ngenes-K+1):ngenes)
    ordered_weights_matrix[,i] <- weights_order[position]
    ordered_genes_matrix[,i] <- genes_order[position]

}

colnames(ordered_weights_matrix) = colnames(weight_mat)
colnames(ordered_genes_matrix) = colnames(weight_mat)

pdf(file = paste0(fileDIR, paste0("Heatmap_top_genes_",target,"_weights.pdf")))
myColors <- c(seq(-15,-11.1, length=100),seq(-11,4, length=100),seq(4.1,7.4, length=100))
myPalette <- colorRampPalette(c("red", "black", "green"))(n = 299)
gplots::heatmap.2(ordered_weights_matrix,dendrogram = "none",Rowv = NA,Colv = NA,labRow = F ,srtCol = 0,trace = "none", density = "none", col=myPalette, cellnote =ordered_genes_matrix, notecex = 0.5, notecol = "white", colsep=1:15,sepcolor="white",sepwidth=c(0.001,0.001), key.title = "log(weights)", breaks= myColors, key.xlab ="log(weights)",na.color = "black")
dev.off()

