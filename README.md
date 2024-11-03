# Assignment 3 
## Background
This GitHub Repository houses the coding pipeline to re-implement the "Explainable automated coding of clinical notes using hierarchical label-wise attention networks and label embedding initialisation", as part of UW-Madison's CS 769 Advanced Natural Language Processing for Fall 2024. The underlying code-base to perform data pre-processing, embedding generation and model training are present in the repository, however due to the DUA for MIMIC-III data no data is available for reproducing the results of this re-implementation. The underlying folder structure is as follows:

## Folder Structure:
### Data
- Training_results: Model training results for multiple epochs across training, validation and test sets
- notebooks: Notebooks for data pre-processing sourced from the CAML github (https://github.com/jamesmullenbach/caml-mimic/tree/master). The CAML code was utilized in the original HLAN model paper for pre-processing the data, so we replicated this process using the available CAML code.
- processed_50: Outlying folder path for pre-processed MIMIC-III data (removed)
### Embeddings

### HLAN_Model


