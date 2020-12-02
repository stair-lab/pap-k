# pap-k

## General comments:

* The code can be run with python 3.6. Required additional libraries are matplotlib, numpy, pandas, pickle, sklearn, tqdm, umap, and scipy.

* Compressed citation dataset is provided in the 'data/citation' folder. Movielens data can be downloaded from [https://grouplens.org/datasets/movielens/100k/](https://grouplens.org/datasets/movielens/100k/), and Behance data can be downloaded from [https://cseweb.ucsd.edu/~jmcauley/datasets.html\behance](https://cseweb.ucsd.edu/~jmcauley/datasets.html\behance).

* Hyperparameters are passed as arguments into the train function.

* The path for the input and output in the code files are described with respect to the 'code' folder.

* 'utils.py', 'surrogates.py', 'greedy_surrogate.py', and 'preck_surrogate.py' are intermediatery files.

* 'results' and 'plots' folders are initially empty and will contain the output generated from the following files.
    
## Real-world Data Results:

* Processing on Movielens dataset:
    (a) Download 'u.data' file from the above link and save it in the '../data/movielens' folder.
    (b) Run the script 'data_processing_for_movielens.ipynb'. It will save 'train_data_d30.tsv', 'val_data_d30.tsv', and 'test_data_d30.tsv' in the '../data/movielens' folder. 
    
* Processing on Citation dataset:
    (a) Uncompress the '../data/citation/citation.tar.gz' file to get the citation data. 
    (b) Run the script 'data_processing_for_citation.ipynb'. It will save 'train_data_d50.tsv', 'val_data_d50.tsv', and 'test_data_d50.tsv' in the '../data/citation' folder. 
    
* Processing on Behance dataset:
    (a) Download 'Behance_Image_Features.b' and 'Behance_appreciate_1M' files from the above link and save them in the '../data/behance' folder.
    (b) Run the script 'data_processing_for_behance.ipynb'. It will save 'train_data_d50.tsv', 'val_data_d50.tsv', and 'test_data_d50.tsv' in the '../data/behance' folder. 
    
* The command 
```
python3 auc-rel-k-cross-validation.py k dataset surrogate
```
runs cross validation on a particular k-value, dataset, and surrogate. Takes k value, dataset name, and surrogate name as parameters and saves logs on training and validation datasets. Choose the best parameters for each surrogate and use them in 'auc-rel-k-run.py' file.

* The command 
```
python3 auc-rel-k-run.py k dataset
```
trains all surrogates on a particular k-value and dataset (Use the best parameter values in the train function that are obtained from results of auc-rel-k-cross-validation.py for each surrogate and dataset). Takes the k value and dataset as inputs and saves logs on training and testing datasets.

## Citation

If you use this code, please cite the our paper from ICML'20.
```
@inproceedings{hiranandani2020optimization,
  title={Optimization and Analysis of the pAp@ k Metric for Recommender Systems},
  author={Hiranandani, Gaurush and Vijitbenjaronk, Warut and Koyejo, Sanmi and Jain, Prateek},
  booktitle={International Conference on Machine Learning},
  pages={4260--4270},
  year={2020},
  organization={PMLR}
}
```