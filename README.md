# MEFFGRN
MEFFGRN is a  supervised deep neural network model for predicting gene regulatory networks in EPC cells infected with SVCV. The MEFFGRN approach employs the matrix enhancement technique to augment the gene adjacency matrix. It integrates the network structural characteristics derived from the enhanced adjacency matrix with gene expression features, utilizing a feature fusion strategy to merge these two types of features for gene representation.

Code is tested using Python 3.8 and R 3.6


# Requirement

-   scikit-learn (Compatible with all versions)
-   Tensorflow 2.9.1
-   Numpy  1.21.6
-   Sklearn  0.0
-   Pandas   1.4.2
-   Scipy       1.8.0



#  Tutorial

## Step 1:Standardized data
**Code**: minmax.py
**Input**:Original gene expression profile
**Output**:Standardized gene expression profile

## Step 2:Generate gene pair list and adjacency matrix
**Code**: dataprocess-1.py
**Input**:gene expression profile and reference network
**Output**:training gene pairs and adjacency matrix between genes

## Step 3:Enhanced adjacency matrix
**Code**: Reward1.m   -->WKNKN2.m

## Step 4:Generate input for MEFFGRN
**Code**: get_histogram_dream4.py
**Input**:Gene expression profile、the enhanced adjacency matrix  and  the benchmark, etc
**Parameters**:
-out_dir: Indicate the path for output.
-expr_file: The file of the gene expression profile.
-pairs_for_predict_file: The file of the training gene pairs and their labels.
-geneName_map_file: The file to map the name of gene in expr_file to the pairs_for_predict_file
-flag_load_from_h5: Is the expr_file is a h5 file. Default is false.
-flag_load_split_batch_pos: Default is false.
**Example**:
python3  -out_dir   network1_representation  
-expr_file  database/data/DREAM100/insilico_size100_1_timeseries.csv
-pairs_for_predict_file   training_pairsnetwork1.txt
-geneName_map_file  network1_geneName_map.txt
-flag_load_from_h5  False
-flag_load_split_batch_pos  False
**Output**:
-   version0 folder: Includes the x file only include the primary image of the gene pair
-   version11 folder:Includes a master image and a neighbor image for each gene pair, used as an input to MEFFGRN.
## Step 5:Train MEFFGRN
**Code**: MEFFGRN-5CV.py
## Step 6:Prediction
**Code**: prediction.py

