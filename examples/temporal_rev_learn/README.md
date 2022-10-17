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
@ARTICLE{markovic2022temprl,
AUTHOR={Marković, Dimitrije and Reiter, Andrea M. F. and Kiebel, Stefan J.},    
TITLE={Revealing human sensitivity to a latent temporal structure of changes},      
JOURNAL={Frontiers in Behavioral Neuroscience},      
VOLUME={16},           
YEAR={2022},      
URL={https://www.frontiersin.org/articles/10.3389/fnbeh.2022.962494},       
DOI={10.3389/fnbeh.2022.962494},      
ISSN={1662-5153},   
}
``` 