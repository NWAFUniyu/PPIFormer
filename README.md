# File Description

## /CompareWithPioneer/
The files in this directory are used to implement the models developed by previous researchers.

#### /CompareWithPioneer/PIPR_PAAE/
The files in this directory directly reproduce the PIPR model and the PAAE encoding method proposed by them.  
>vec5_Rcnn.py: This file is used for training the model.   
>testvec5Rcnn.py: This file is used for test the model.  

#### /CompareWithPioneer/PIPR_onehot/
The files in this folder replicate the PIPR model and utilize the one-hot encoding scheme to encode amino acid sequences.  
>onehot_Rcnn.py: This file is used for training the model.   
>testonehotRcnn.py: This file is used for test the model.

#### /CompareWithPioneer/PIPR_w2v/
The files in this folder replicate the PIPR model and utilize the word2vec encoding scheme to encode amino acid sequences.
>w2v_Rcnn.py: This file is used for training the model.   
>testwevRcnn.py: This file is used for test the model.

## /MyNegativeSampling/
The files in this folder implement the new negative sample generation method proposed by us and 
divide the data into training and testing sets for model training.  

#### /MyNegativeSampling/01_generatingMatrix/
This step was quoted from DeNovo(doi: 10.1093/bioinformatics/btv737). In this step, Global alignment (GlobalAlign) was performed using the 
Needleman-Wunsch algorithm with the BLOSUM30 matrix to capture distant similarities.  

#### /MyNegativeSampling/02_genIntTable/
This step is used to generate the interaction table based on the Hash table.

#### /MyNegativeSampling/03_genNegSamp/
This step is used to generate negative samples.
