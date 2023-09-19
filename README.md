# GenotoxNet
===========================================================================
### Genotoxicity prediction model based on multimodal convolutional neural network       
<br>     

GenotoxNet is a multimodal convolutional network for chemical genotoxicity prediction. It takes chemical structure, gene expression data and Toxicity Forecasting (ToxCast) assay data as inputs and predicts the chemical genotoxicity (binary classification).

<!-- TOC START min:1 max:3 link:true asterisk:false update:true -->
- [Requirements](#requirements)
- [User Guide](#user-guide)
  - [Step 1](#step-1)
  - [Step 2](#step-2)
  - [Step 3.](#step-3)
- [Contact Us](#contact-us)
<!-- TOC END -->

# Requirements
**We generated the relevant packages that the project relies on as Requirements.txt files**
These packages can be installed directly in batches using CONDA:
    `conda install --yes --file requirements.txt`

    deepchem=2.6.0.dev20210509222234
    hickle=4.0.4
    Keras=2.4.3
    matplotlib=3.4.0
    networkx=2.5
    numpy=1.19.5
    pandas=1.2.3
    rdkit=2017.09.1
    scikit-learn=0.24.1
    scipy=1.6.2
    seaborn=0.11.1
    sklearn=0.0
    tensorboard=2.4.1
    tensorboard-plugin-wit=1.8.0
    tensorflow-addons=0.13.0
    tensorflow-determinism=0.3.0
    tensorflow-estimator=2.4.0
    tensorflow-gpu=2.4.1
    tensorflow-gpu-estimator=2.3.0

  - It is recommended to execute this project in a linux environment, such as Anaconda3

# User Guide
We provide detailed step-by-step instructions for running GenotoxNet model including data preprocessing, model training, and model test.
## Step 1
**Drug feature representation**

In this project, each chemical will be represented as a graph containing nodes and edges. We collected 244 chemicals from the Carcinogenome Project that have SMILES. Then we put the SMILES and pert ID of chemicals into smiles (for example: ./Dataset/chemicals.smiles) file and run `process_drug.py` script to extract three types of features by [deepchem](https://github.com/deepchem/deepchem) library. The node feature (75 dimension) corresponds to an atom in within a chemical, with includes atom type, degree and hybridization, etc. The adjacent features denote the all the neighboring atoms, and the degree features denote the number of neighboring atoms. The above feature list will be further compressed as pert ID.hkl using hickle library, and placed in drug_graph_feat (for example: ./GenotoxNet_data/drug_graph_feat) folder. 

Please note that we provided the exacted features from the Carcinogenome Project, and gene expression data from CRCGN_ABC dataset (https://clue.io/data/CRCGN) and assay data from ToxCast, just unzip the drug_graph_feat.zip file in GenotoxNet_data/drug_graph_feat folder. 


## Step 2
**GenotoxNet model selection, training and testing**

This project provides a model hyperparameter screening module `Genotoxicity_5foldcrossval.py` and a predictor training module `Genotoxicity_predict.py`. One can run python `Genotoxicity_5foldcrossval.py` to implement the model hyperparameter screening. we use grid search to select different learning rates, batch sizes, dropout coefficients and L2 regularization terms, train models on various hyperparameter combinations, and compare the predictive performance of each model on the validation set. The evaluation metrics of different models on the validation set are put into csv (for example: ./GenotoxNet_data/gridsearch_result.csv) file. 

And for model training and test case, after setting the optimal hyperparameters, one can run python `Genotoxicity_predict` to implement the GenotoxNet classification model. The trained model will be saved in h5 (for example: ./GenotoxNet_data/bestmodel/MyGenotoxNet_highestAUCROC _256_256_256_bn_relu_GMP.h5) file. The overall auROC and auPRC of validation set and external test set will be calculated. And the predicted result of each chemical in external test set will be placed in csv (for example: ./GenotoxNet_data/predict_result/ lr0.0001_batch16_dropout0.1_0.0001l2_extra_res.csv) file.


## Step 3
**Model ablation experiments**

This project also provides a model hyperparameter screening module `Genotoxicity_5foldcrossval_ablation.py` and a predictor training module `Genotoxicity_model_ablation.py`.
One can run python `Genotoxicity_5foldcrossval_ablation.py` to implement the model hyperparameter screening for each combination of different types of data. After setting the optimal hyperparameters, one can run python `Genotoxicity_model_ablation.py` to calculate the chemical genotoxicity probability prediction value for each chemical. The overall auROC and auPRC of validation set and external test set of each combination of different types of data will be calculated and placed in csv (for example: ./GenotoxNet_data/train_validation_test_result_ablation.csv) file.

# Contact Us

Email: hzzhang@rcees.ac.cn

# License
This project is licensed under the MIT License - see the LICENSE.md file for details
