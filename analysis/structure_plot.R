library(data.table)
library(ggplot2)
library(argparse)
library(magrittr)
library(tidyr)
parser <- ArgumentParser()

parser$add_argument("--SavePath",help = "relative path to save folder")
args <- parser$parse_args()
#SaveFolderPath <- "models/TotalDeltaETM_allgenes_ep1000_nlv32_bs512_combinebyadd_lr0.01_train_size1"
SaveFolderPath = args$SavePath
DT <- fread(paste(SaveFolderPath, "topics.csv", sep = "/")) %>% as.data.frame()

rownames(DT) <- DT$V1
DT <- DT[, -1] 
topics_wide <- DT
topics_wide$max_topic <- colnames(DT)[apply(DT, 1, which.max)]
topics_wide$max_topic_prob <- apply(DT, 1, max)
# only retain the topics with more than 100 cells
topics_to_keep = topics_wide$max_topic %>% table() %>% names()
topics_to_keep = topics_to_keep[(topics_wide$max_topic %>% table() >= 100)]
topics_wide <- topics_wide[,c(topics_to_keep)] 
# re-compute the max_topic_prob and max_topic
topics_wide$max_topic <- colnames(topics_wide)[apply(topics_wide, 1, which.max)]
topics_wide$max_topic_prob <- apply(topics_wide[,-ncol(topics_wide)], 1, max)
# sanity check
table(topics_wide$max_topic)

dt <- data.table(topics_wide, key="max_topic")
dt$cell <- as.factor(rownames(topics_wide))
K <- 1000
dt[, .SD[max_topic_prob %in% head(sort(unique(max_topic_prob)), K)], by=max_topic] %>% as.data.frame() -> topics_wide_topK
gather(topics_wide_topK, topic, probabilities, colnames(topics_wide_topK)[2:(ncol(topics_wide_topK)-2)], factor_key=TRUE) -> topics_long

p <- ggplot(topics_long, aes(x = cell, y = probabilities, fill=topic)) +
  geom_bar(position="stack",stat="identity", size=0) +
  scale_fill_brewer("Latent topic", palette = "Paired")+
  facet_grid(~ max_topic, scales = "free", switch = "x", space = "free")+
  labs(x = "Cells", y = "Topic Proportion")+
  theme(
    legend.position = "top",
    legend.justification = "left", 
    legend.margin = margin(0, 0, 0, 0),
    legend.box.margin = margin(10,10,10,10),
    text = element_text(size=10),
    #panel.spacing.x = unit(0.1, "lines"),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    panel.grid = element_blank(),
    panel.background = element_rect(fill='transparent'),
    plot.background = element_rect(fill='transparent', color=NA))+
  guides(fill = guide_legend(nrow = 1))
#p
ggsave(paste(SaveFolderPath, "struct_plot.pdf", sep = "/"))
