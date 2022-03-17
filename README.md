# DeltaETM: Generalized Embedded Topic Model for RNA Velocity 

saved as vector images, no PNG, except no drawing UMAP

- Fig1 Most case-contorl analysis does not capture full velocty dynamic landscape
    - A: scatterplot of represented genes U versus S, color cells according to topics (from DeltaETM), Limitations: velocity only capture short-term dynamics, case-control study 
    - B: Global genes behave the same way
    - C: Model architec

- Fig2 (latent space) learn short-term dynamic 
    - A: H as strcuture plot
    - B: 
        - UMAP on H, seperating
        - UMAP on spliced count (blocky)
        - UMAP on unspliced count (blocky)
    - C: ETM on unspliced only --> H,  H_delta, better resolution, a porpotion of cells in the same topics

- Fig3 (decoder weight matrix) depend on weight matrix
    - A/B: topic-specific genes
    - C

- Fig4 DeltaETM improve statistical (significance, topic-specific DE)
    - case-control DE, more hits
    - interpretability

Plan B: another dataset Fig2/3 deltaETM