library(ggplot2)
library(data.table)
library(gplots)
library(magrittr)
library(argparse)

parser <- ArgumentParser()

parser$add_argument("--SavePath",  
    help = "relative path to save folder")
parser$add_argument("--target", default= "delta", 
    help = "weight to plot, delta or rho")

args <- parser$parse_args()    
Save_Path = args$SavePath
target <- args$target
fileDIR = paste0(Save_Path, "/")

target = "delta"
fileDIR = "models/TotalDeltaETM_allgenes_ep1000_nlv32_bs512_combinebyadd_lr0.01_train_size1/"

weights_dt = fread(paste0(fileDIR, paste0(target,"_weights.csv")))
weights = weights_dt %>% as.data.frame()
weight_mat <- weights %>% as.data.frame()
rownames(weight_mat) = weights$V1
weight_mat = as.matrix(weight_mat[,-1])

# subset on meaningful topics
colnames(weight_mat)
subset_topics = paste0("topic_", c(0,1,3,6,10,13,19,23,30))
weight_mat_subset = weight_mat[, subset_topics]
pdf(file = paste0(fileDIR, paste0("Heatmap_",target,"_weights_log_subset.pdf")))
gplots::heatmap.2(log(weight_mat_subset))
dev.off()

pdf(file = paste0(fileDIR, paste0("Heatmap_",target,"_weights_log.pdf")))
gplots::heatmap.2(log(weight_mat))
dev.off()


pdf(file = paste0(fileDIR, paste0("Heatmap_",target,"_weights.pdf")))
gplots::heatmap.2(weight_mat)
dev.off()

pdf(file = paste0(fileDIR, paste0("Heatmap_",target,"_weights_sqrt.pdf")))
gplots::heatmap.2(sqrt(weight_mat))
dev.off()

ngenes = nrow(weight_mat)

ordered_weights_matrix = matrix(nrow = ngenes, ncol = ncol(weight_mat))
ordered_genes_matrix = matrix(nrow = ngenes, ncol = ncol(weight_mat))


for(i in 1:ncol(weight_mat)){
    print(paste0("Topic-",i))
    weights_order <- weights[weights[,paste0('topic_',i-1)] %>% order(decreasing = T),paste0('topic_',i-1)]
    genes_order = weights[weights[,paste0('topic_',i-1)] %>% order(decreasing = T),'V1']
    position = c(1:ngenes)
    ordered_weights_matrix[,i] <- weights_order[position]
    ordered_genes_matrix[,i] <- genes_order[position]

}

colnames(ordered_weights_matrix) = colnames(weight_mat)
colnames(ordered_genes_matrix) = colnames(weight_mat)

# plot heatmap of weight matrix on top genes and selected topics
K = 5
ordered_genes_matrix[1:K,subset_topics] %>% as.character() %>% unique() -> myTopGenes

weight_df = reshape2::melt(weight_mat[myTopGenes,subset_topics])
colnames(weight_df) = c("Gene", "Topic", "value")

library(tidyverse)
weight_df_tb = as.tibble(weight_df)
df_to_plot <- weight_df %>% arrange(Gene)
df_to_plot
'''
xlab = paste("Top",K,"Topic", "genes")
pdf(file = paste0(fileDIR, paste0("Heatmap_",target,'_top',K,"_weights_log_subset.pdf")))
ggplot(df_to_plot, aes(x = factor(Gene), y = Topic, fill = value)) +
    theme(axis.text.x = element_text(angle=70, vjust=1, hjust=1, size=1)) +
    theme(legend.position = "none") +
    xlab(xlab) + ylab("Topics") +
    geom_tile(colour = "black", size = .1) +
    scale_fill_distiller("", palette = "PuRd", direction = 1,trans="log")+
    theme_classic() +
    theme(axis.title = element_text(size=8)) +
    theme(axis.text = element_text(size=6)) +
    theme(legend.spacing = unit(.1, "lines"),
    legend.key.size = unit(.5, "lines"),
    legend.text = element_text(size=5),
    legend.title = element_text(size=6),
    panel.background = element_rect(fill='transparent'),
    plot.background = element_rect(fill='transparent', color=NA),
    legend.background = element_rect(fill='transparent', size=0.05),
    legend.box.background = element_rect(fill='transparent'))
dev.off()'''

pdf(file = paste0(fileDIR, paste0("Heatmap_",target,'_top',K,"_weights_log_subset.pdf")))
gplots::heatmap.2(log((weight_mat[myTopGenes,subset_topics])),cexRow=0.4)
dev.off()
#library(topicmodels)
#library(wordcloud)
#K = 20 # top K genes to plot

#topic = 1
'''for(topic in 1:ncol(ordered_weights_matrix)){
    probabilities = ordered_weights_matrix[1:K,topic]
    genes = ordered_genes_matrix[1:K,topic]

    mycolors <- brewer.pal(8, "Dark2")
    png(file = paste0(fileDIR, paste0("gene_cloud_top",K,"_",colnames(ordered_weights_matrix)[topic],".png")))
    wordcloud(genes, probabilities, 
        random.order = FALSE, color = mycolors,
        main = colnames(ordered_weights_matrix)[topic])
    dev.off()
}
'''

#data.table::fwrite(as.data.frame(ordered_genes_matrix), file = paste0(fileDIR, paste0(target,"_topK_genes.csv")))
#data.table::fwrite(as.data.frame(ordered_weights_matrix), file = paste0(fileDIR, paste0(target,"_topK_weights.csv")))


#K = 10 # top K genes and bottom K for each topic

#weight_mat_topK = ordered_weights_matrix[1:K,]
#gene_mat_topK = ordered_genes_matrix[1:K,]
#pdf(file = paste0(fileDIR, paste0("Heatmap_top_genes_",target,"_weights.pdf")))
#myColors <- seq(0,1, length=100)
#myPalette <- colorRampPalette(c("red", "black", "green"))(n = 99)
#gplots::heatmap.2(weight_mat_topK ,dendrogram = "none",Rowv = NA,Colv = NA,labRow = F ,srtCol = 0,trace = "none", density = "none", cellnote =gene_mat_topK, notecex = 0.5, notecol = "black", colsep=1:15,sepcolor="white",sepwidth=c(0.001,0.001), key.title = "Gene Probability", key.xlab ="Gene Probability",na.color = "black")
#dev.off()

