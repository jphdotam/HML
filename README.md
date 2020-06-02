# HML

This is the repository for the His pacing machine learning project.
 
![Confusion matrix](cm.png)

`train.py` - Train the 3 CNNs via 3-fold cross-validation across the training ser
`test.py` - Evaluate the ensembled 3 CNNs on the testing dataset
`vis.py` - Create the saliency maps

The Python version used is `3.8` and the following packages are required (all installed by with `conda` via the `conda-forge` channel)
* `matplotlib`
* `pytorch v1.5` & `torchvision` (pytorch conda channel)
* `pyyaml`
* `numpy`
* `pandas`
* `scikit-image`
* `scikit-learn`
* `scipy`
* `statsmodels`
* `tqdm`
* `xlrd`
