# Assignment 3 
## Background
This GitHub Repository houses the coding pipeline to re-implement the "Explainable automated coding of clinical notes using hierarchical label-wise attention networks and label embedding initialisation", as part of UW-Madison's CS 769 Advanced Natural Language Processing for Fall 2024. The underlying code-base to perform data pre-processing, embedding generation and model training are present in the repository, however due to the DUA for MIMIC-III no data is available for reproducing the results of this re-implementation. The underlying folder structure is as follows:

## Folder Structure:
### Data
- Training_results: Model training results for multiple epochs across training, validation and test sets
- notebooks: Notebooks for data pre-processing sourced from the CAML github [CAML](https://github.com/jamesmullenbach/caml-mimic/tree/master). The CAML code was utilized in the original HLAN model paper for pre-processing the data, so we replicated this process using the available CAML code.
- processed_50: Outlying folder path for pre-processed MIMIC-III data (removed)
### Embeddings
- pre-train_embeddings.py: Code to pre-train the word and label CBoW models
- code-emb.mode: Saved pre-trained label model
- word-emb.model: Saved pre-trained word model
- word-emb.model.syn1neg.npy: Additional file for word model
- word-emb.model.wv.vectors.npy: Additional file for the word model
- generate_code_embeddings.py: Code to generate embeddings from the pre-trained label model for the notes corresponding labels
- generate_word_embeddings.py: Code to generate embeddings from the pre-trained word model for the notes
- embed_data_helper.py: Helper functions to generate embeddings

### HLAN_Model
- Development/Model_Comp_Graph.ipynb: Development notebook to complete first pass of computational graph of HLAN
- Development/Model_Train_Development.ipynb: Development notebook to create training pipeline for HLAN
- HLAN.py: Main pytorch model for HLAN
- Plots.ipynb: Notebook for training performance monitoring plots and determining best performing epochs
- main_run.py: Main model training and evaluation code
- model_helper.py: Helper function for model main_run (unused)


