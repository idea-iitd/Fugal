# FUGAL: Feature-fortified Unrestricted Graph Alignment
This repository contains the reference implementation for FUGAL for the paper "FUGAL: Feature-fortified Unrestricted Graph Alignment".

## Installation

1. Install PyTorch, numpy, tqdm, networkx, sklearn, and scipy.

## Run FUGAL

1. Import ``helpers/pred.py`` which has the source implementation of FUGAL
2. Call ``predict_alignment()`` function in ``helpers/pred.py`` which takes arguments (queries, targets, mu, niter)
3. queries, targets should be a list of networkx graphs. 
4. Paramters mu, niter for benchmark datasets are provided in the paper.

## Datasets

1. Benchmark Networks:
    1. Graphs with Real Noise: MultiMagna, HighSchool, Voles
    2. Real Graphs: Arenas, inf-euroroad, ca-Netscience, bio-celegans, ACM-DBLP
    3. Synthetic Graphs: Newmann-Watts
2. Real networks data is provided in ``data/`` folder.
3. Noise injected real-world graphs used in the experiments are provided as well.
4. For generating Newmann-Watts graphs (or) to create noisy versions of the real graphs use the [framework](https://github.com/constantinosskitsas/Framework_GraphAlignment)
