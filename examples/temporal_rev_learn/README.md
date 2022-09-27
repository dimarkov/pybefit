# Revealing human sensitivity to a latent temporal structure of changes 
The code accompanying the paper "Revealing human sensitivity to a latent temporal structure of changes" (in pre-print). Here we introduce a semi-Markov formualtion of active inference agent. The behavioural model is than used to analyse the behavioural data of subjects performing a temporally structured reversal learning task. The goal here is to demonstrate subjects' learning of latent temporal strucure of a noisy and dynamic environment.

# Usage
All the figures in the paper can be recreated using the two jupyter notebooks.
To fit the behavioural data availible in directories 'main' and 'pilot' run 
```bash
python run_fits.py --device gpu
```
To generated simulated behaviour for 50 agents in each condition and invert the model for 
estimating confusion matrix run
```bash
python run_sims.py -n 50 --device gpu
```

# Citing
If you use the code or availible behavioural data please consider citing our paper:
```
@article{markovic2022temp,
title = {Revealing human sensitivity to a latent temporal structure of changes},
journal = {Frontiers in Behavioural Neuroscience},
year = {2022},
issn = {},
doi = {https://doi.org/10.3389/fnbeh.2022.962494},
url = {},
author = {Dimitrije Marković and Andrea M.F. Reiter and Stefan J. Kiebel},
keywords = {Decision making, Bayesian inference, Multi-armed bandits, Reversal learning task, Active inference}
}
``` 