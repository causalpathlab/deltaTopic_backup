#!/bin/bash

#models/TotalDeltaETM_allgenes_ep1000_nlv4_bs512_combinebyadd_lr0.01_train_size1
#models/TotalDeltaETM_allgenes_ep1000_nlv8_bs512_combinebyadd_lr0.01_train_size1
#models/TotalDeltaETM_allgenes_ep2000_nlv16_bs512_combinebyadd_lr0.01_train_size1
#models/TotalDeltaETM_allgenes_ep1000_nlv32_bs512_combinebyadd_lr0.01_train_size1

for nlV in 4 8 32;do
    path="models/TotalDeltaETM_allgenes_ep1000_nlv${nlV}_bs512_combinebyadd_lr0.01_train_size1"
    echo $path
    #python get_latent.py --SavePath $path
    #Rscript analysis/structure_plot.R --SavePath $path
    Rscript analysis/plot_weights.R --SavePath $path --target delta
    Rscript analysis/plot_weights.R --SavePath $path --target rho
    #Rscript analysis/phase_plot.R --SavePath $path
done
  

